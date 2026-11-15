// Isolated benchmark coordinator. Not a production runtime selector.
const describe = e => e?.message || String(e);
async function runtime(threads, injectStartupFailure, forceThreaded) {
  let cleanup=()=>{};
  let fallback=null;
  const nativeWorker=globalThis.Worker;
  const workerEvidence={created:0,terminated:0};
  if(injectStartupFailure) {
    globalThis.Worker=class {
      constructor(...args) {
        if(workerEvidence.created===1)throw new Error('Injected second-worker startup failure');
        const worker=new nativeWorker(...args);workerEvidence.created++;
        const terminate=worker.terminate.bind(worker);
        worker.terminate=()=>{workerEvidence.terminated++;terminate();};
        return worker;
      }
    };
  }
  try {
    if((threads>1 || forceThreaded) && globalThis.crossOriginIsolated && typeof SharedArrayBuffer==='function') {
      try {
        const pool=await import('/threaded/thread-pool.js');cleanup=pool.terminateWorkers;
        await pool.wasm.default();
        const expectRejection=async(call,pattern)=>{
          let rejected=false;try{await call();}catch(error){if(!pattern.test(describe(error)))throw error;rejected=true;}
          if(!rejected)throw new Error('Thread-pool lifecycle guard did not reject');
        };
        await expectRejection(()=>pool.wasm.loadModel(new Uint8Array(),'cpu'),/initCpuThreadPool/);
        await expectRejection(()=>pool.wasm.initCpuThreadPool(0),/between 1 and 32/);
        await pool.wasm.initCpuThreadPool(threads);
        await expectRejection(()=>pool.wasm.initCpuThreadPool(threads),/already attempted/);
        return {wasm:pool.wasm,cleanup,selected:'threaded',fallback,workerEvidence};
      } catch(error) {cleanup();fallback=describe(error);cleanup=()=>{};}
    } else if(threads>1) fallback='cross-origin isolation/shared memory unavailable';
    const wasm=await import('/plain/speechworks_web.js');await wasm.default();
    return {wasm,cleanup,selected:'plain',fallback,workerEvidence};
  } finally {globalThis.Worker=nativeWorker;}
}
async function fixture() {
  const bytes=await(await fetch('/jfk.wav')).arrayBuffer();const view=new DataView(bytes);
  const four=i=>String.fromCharCode(...new Uint8Array(bytes,i,4));
  if(four(0)!=='RIFF'||four(8)!=='WAVE')throw new Error('Expected WAV');
  let format,data;
  for(let at=12;at+8<=bytes.byteLength;) {
    const size=view.getUint32(at+4,true);
    if(four(at)==='fmt ')format=[view.getUint16(at+8,true),view.getUint16(at+10,true),view.getUint32(at+12,true),view.getUint16(at+22,true)];
    if(four(at)==='data')data=[at+8,size];at+=8+size+(size&1);
  }
  if(JSON.stringify(format)!=='[1,1,16000,16]'||!data)throw new Error('Expected signed16-bit mono16kHz');
  const pcm=new Float32Array(data[1]/2);for(let i=0;i<pcm.length;i++)pcm[i]=view.getInt16(data[0]+i*2,true)/32768;
  return pcm;
}
globalThis.onmessage=async({data})=>{
  let engine;
  try {
    const at=performance.now();engine=await runtime(data.threads,data.injectStartupFailure,data.forceThreaded);
    const initMs=performance.now()-at;
    const pcm=await fixture();const manifest=await(await fetch('/models.json')).json();const direct=[];
    for(const id of data.models) {
      const spec=manifest.models[id];const fetchAt=performance.now();
      const bytes=new Uint8Array(await(await fetch('/'+spec.url)).arrayBuffer());
      const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
      if(bytes.length!==spec.bytes||digest!==spec.sha256)throw new Error(`Model integrity ${id}`);
      const fetchMs=performance.now()-fetchAt;const loadAt=performance.now();
      const model=await engine.wasm.loadModel(bytes,'cpu');const loadMs=performance.now()-loadAt;
      try {
        const firstAt=performance.now();const firstText=await model.transcribe(pcm);const firstMs=performance.now()-firstAt;
        const clips=[];
        for(const seconds of data.prefixes) {
          const clip=pcm.slice(0,Math.min(pcm.length,seconds*16000));const timesMs=[];let text;
          for(let n=0;n<data.repeats;n++) {
            const start=performance.now();const next=await model.transcribe(clip);timesMs.push(performance.now()-start);
            if(text!==undefined&&text!==next)throw new Error('Unstable repeat');text=next;
          }
          const sorted=[...timesMs].sort((a,b)=>a-b);
          clips.push({audioSeconds:seconds,timesMs,medianMs:sorted[Math.floor(sorted.length/2)],text});
        }
        direct.push({id,device:model.device(),fetchMs,loadMs,firstMs,firstText,clips});
      } finally {model.free();}
    }
    const result={requestedThreads:data.threads,selected:engine.selected,fallback:engine.fallback,workerEvidence:engine.workerEvidence,initMs,crossOriginIsolated:globalThis.crossOriginIsolated,sharedArrayBuffer:typeof SharedArrayBuffer,hardwareConcurrency:navigator.hardwareConcurrency,direct};
    engine.cleanup();engine=null;postMessage({result});
  } catch(error) {engine?.cleanup();postMessage({error:describe(error),stack:error?.stack});}
};
