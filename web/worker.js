import init, {loadModel, DictationSession} from './speechworks_web.js';

const describe = error => error?.message || String(error);

/** A worker owns model lifetimes and serializes inference, while audio/VAD keeps
 * advancing during asynchronous GPU work. All model fetches are same-origin. */
export function startWorker(options = {}) {
  let settings = {manifestUrl: new URL('./models.json', import.meta.url).href, backend: 'auto', ...options};
  let manifestPromise, recording = null, running = false, activeClip = false, clipQueue = [], queuedSamples = 0;
  const models = new Map();
  const devices = new Map();
  const ready = init();
  const send = data => self.postMessage(data);
  function reportBackend(id,recognizer) {
    const device=recognizer.device(),fallback=recognizer.fallbackReason();
    if(devices.get(id)!==device) {devices.set(id,device);send({type:'backend',model:id,device,fallback});}
  }
  ready.then(() => send({type:'ready'}), error => send({type:'error', message:describe(error)}));

  async function manifest() {
    return manifestPromise ||= (async () => {
      const url = new URL(settings.manifestUrl, self.location.href);
      if (url.origin !== self.location.origin) throw new Error('Model manifests must be served by this application.');
      const response = await fetch(url,{cache:'no-store',redirect:'error'});
      if (!response.ok) throw new Error(`Model manifest: HTTP ${response.status}`);
      const value = await response.json();
      if (value.schema !== 1 || value.sampleRate !== 16000 || !value.models || !value.defaults) throw new Error('Unsupported model manifest.');
      return value;
    })().catch(error=>{manifestPromise=null;throw error;});
  }
  async function choices() {
    const m = await manifest();
    return {...m.defaults, ...settings.models};
  }
  async function model(id) {
    if (models.has(id)) return models.get(id);
    if (models.size >= 2) throw new Error('At most two models may be resident; create another engine to change profiles.');
    const promise = (async () => {
      await ready;
      const spec = (await manifest()).models[id];
      if (!spec || !/^[a-f0-9]{64}$/.test(spec.sha256) || !Number.isSafeInteger(spec.bytes)
          || spec.bytes < 1 || spec.bytes > 512 * 1024 * 1024) throw new Error(`Invalid model specification: ${id}`);
      const url = new URL(spec.url, settings.manifestUrl);
      if (url.origin !== self.location.origin) throw new Error('Models must be served by this application.');
      let cache;
      try { cache = await caches.open('speechworks-models-v1'); } catch { /* Storage may be disabled; inference still works. */ }
      let response = await cache?.match(url.href);
      let fromCache = !!response;
      if (!response) response = await fetch(url, {redirect:'error'});
      if (!response.ok) throw new Error(`${id}: HTTP ${response.status}`);
      const reader = response.body?.getReader();
      let bytes;
      if (reader) {
        bytes = new Uint8Array(spec.bytes); let offset = 0;
        try {
          while (true) {
            const {value,done} = await reader.read(); if (done) break;
            if (offset + value.length > bytes.length) throw new Error(`${id}: model exceeds its pinned size`);
            bytes.set(value,offset); offset += value.length;
            send({type:'progress',model:id,value:offset/spec.bytes});
          }
          if (offset !== bytes.length) throw new Error(`${id}: incomplete model download`);
        } catch(error) {
          if(fromCache) {try {await cache?.delete(url.href);}catch{}}
          throw error;
        } finally { await reader.cancel().catch(() => {}); }
      } else bytes = new Uint8Array(await response.arrayBuffer());
      const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
      if (bytes.length !== spec.bytes || digest !== spec.sha256) {
        await cache?.delete(url.href);
        throw new Error(`${id}: model checksum does not match the pinned manifest`);
      }
      if (!fromCache) {
        try { await cache?.put(url.href,new Response(bytes, {headers:{'Content-Type':'application/octet-stream'}})); } catch { /* Quota does not prevent local inference. */ }
      }
      const recognizer = await loadModel(bytes,settings.backend);
      reportBackend(id,recognizer);
      return recognizer;
    })();
    models.set(id,promise);
    try { return await promise; } catch (error) { models.delete(id); throw error; }
  }
  async function preload() {
    const profile = await choices();
    await model(profile.final);
    if (settings.session?.previews !== false && profile.draft !== profile.final) await model(profile.draft);
  }
  function dispose(record) { record?.session.free(); }
  function failRecording(record,error) {
    if (recording === record) recording = null;
    record.cancelled = true; record.session.cancel();
    send({type:'error',session:record.id,message:describe(error)});
    dispose(record);
  }
  async function pump() {
    if (running) return;
    running = true;
    try {
      while (true) {
        const r = recording;
        if (r) {
          const job = r.session.nextFinal() || (!r.stopping && performance.now() >= r.previewNotBefore && r.session.preview());
          if (job) {
            const isFinal = job.isFinal;
            const phrase = job.phrase;
            try {
              const profile = await choices();
              const recognizer = await model(isFinal ? profile.final : profile.draft);
              const began = performance.now();
              const text = (await recognizer.transcribe(job.samples)).trim();
              reportBackend(isFinal ? profile.final : profile.draft,recognizer);
              // Leave at least as much time for capture/VAD as the preceding preview
              // consumed. Faster GPUs retain the configured audio-time cadence.
              if (!isFinal) r.previewNotBefore = performance.now() + (performance.now()-began);
              if (recording === r && !r.cancelled && (isFinal || (!r.stopping && r.session.isCurrent(phrase)))) {
                send({type:isFinal?'final':'live',session:r.id,phrase,text});
              }
            } catch (error) { if (!r.cancelled) failRecording(r,error); }
            finally { job.free(); }
            continue;
          }
          if (r.stopping) { recording = null; send({type:'done',session:r.id}); dispose(r); continue; }
        }
        const clip = clipQueue.shift();
        if (!clip) break;
        queuedSamples -= clip.samples.length;
        activeClip = true;
        try {
          const profile = await choices();
          const recognizer = await model(clip.model || profile.final);
          let text;
          if (clip.samples.length <= 30*16000) text = (await recognizer.transcribe(clip.samples)).trim();
          else {
            const session = new DictationSession({...settings.session,previews:false});
            const texts=[];
            try {
              for(let offset=0;offset<clip.samples.length;offset+=512) {
                const frame=new Float32Array(512);frame.set(clip.samples.subarray(offset,offset+512));session.push(frame);
                let job;
                while((job=session.nextFinal())) { try{texts.push((await recognizer.transcribe(job.samples)).trim());}finally{job.free();} }
              }
              session.finish();let job;
              while((job=session.nextFinal())) {try{texts.push((await recognizer.transcribe(job.samples)).trim());}finally{job.free();}}
              text=texts.filter(Boolean).join(' ');
            } finally {session.free();}
          }
          reportBackend(clip.model || profile.final,recognizer);
          send({type:'result',id:clip.id,text});
        } catch (error) { send({type:'error',id:clip.id,message:describe(error)}); }
        finally {activeClip=false;}
      }
    } finally { running = false; }
  }
  self.onmessage = async ({data:m}) => {
    try {
      await ready;
      if (m.type === 'configure') {
        if (recording || models.size || clipQueue.length) throw new Error('Configure the engine before loading a model.');
        settings = {...settings,...m.options};
        settings.manifestUrl = new URL(settings.manifestUrl,self.location.href).href;
        manifestPromise = null;
        await manifest(); send({type:'configured',id:m.id});
      } else if (m.type === 'preload') {
        await preload(); send({type:'loaded',id:m.id});
      } else if (m.type === 'start') {
        if (recording || activeClip || clipQueue.length) throw new Error('A transcription is already active.');
        recording = {id:m.session,session:new DictationSession({...settings.session,...m.config}),stopping:false,cancelled:false,previewNotBefore:0};
        send({type:'started',id:m.id,session:m.session});
      } else if (m.type === 'audio' && recording?.id === m.session) {
        const r = recording;
        try { r.session.pushInput(m.samples,m.sampleRate || 16000); send({type:'audio-ack',session:r.id,samples:m.samples.length}); }
        catch (error) { failRecording(r,error); }
        void pump();
      } else if (m.type === 'stop' && recording?.id === m.session) {
        recording.stopping = true; recording.session.finish();
        void pump();
      } else if (m.type === 'cancel' && recording?.id === m.session) {
        const old = recording; recording = null; old.cancelled = true; old.session.cancel();
        // A running job owns a copied audio buffer; session has no borrowed async state.
        dispose(old);
      } else if (m.type === 'transcribe') {
        if (recording) throw new Error('Finish dictation before submitting a clip.');
        if (!(m.samples instanceof Float32Array) || m.samples.length > 600*16000 || !m.samples.length) throw new Error('Clip must be 16 kHz mono PCM, up to 10 minutes.');
        if (clipQueue.length >= 8 || queuedSamples+m.samples.length>600*16000) throw new Error('Transcription queue is full.');
        queuedSamples += m.samples.length;
        if (m.priority) clipQueue.unshift(m); else clipQueue.push(m);
        void pump();
      }
    } catch (error) { send({type:'error',id:m.id,session:m.session,message:describe(error)}); }
  };
}
