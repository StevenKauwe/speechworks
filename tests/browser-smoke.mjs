#!/usr/bin/env node
// Real browser acceptance, using installed runtime/model files and a public 16 kHz mono WAV.
// npm install --no-save playwright
// node tests/browser-smoke.mjs --root /path/to/installed-runtime --fixture /path/to/jfk.wav
// PLAYWRIGHT_MODULE and CHROME_EXECUTABLE may select an existing local installation.
// --cpu-only works without WebGPU. --direct-only, --runtime-only and --fallback-only
// select a stage; HEADLESS=0 opens an isolated visible Chrome window for GPU acceptance.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=Object.fromEntries(process.argv.slice(2).map((x,i,a)=>x.startsWith('--')?[x,a[i+1] && !a[i+1].startsWith('--')?a[i+1]:true]:null).filter(Boolean));
const root=path.resolve(String(args['--root'] || path.join(repo,'dist')));
const fixture=path.resolve(String(args['--fixture'] || process.env.SPEECHWORKS_TEST_AUDIO || 'jfk.wav'));
const output=path.resolve(String(args['--output'] || path.join(repo,'test-results/browser.json')));
const cpuOnly=!!args['--cpu-only'];
const directOnly=!!args['--direct-only'];
const runtimeOnly=!!args['--runtime-only'];
const fallbackOnly=!!args['--fallback-only'];
assert.ok(Number(directOnly)+Number(runtimeOnly)+Number(fallbackOnly)<=1,'direct-only, runtime-only and fallback-only are mutually exclusive');
const headless=process.env.HEADLESS !== '0';
const repeats=Number(args['--repeats'] || 3);
assert.ok(Number.isSafeInteger(repeats) && repeats>0 && repeats<=10,'repeats must be 1..10');
const manifest=JSON.parse(await fs.readFile(path.join(root,'models.json'),'utf8'));
const ids=['tiny','base'].map(name=>Object.keys(manifest.models).find(id=>id.includes(name)));
assert.ok(ids.every(Boolean),'installed manifest must contain tiny and base');
await fs.access(fixture);
await fs.access(path.join(root,'speechworks_web_bg.wasm'));
const contentTypes={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.wasm':'application/wasm','.json':'application/json','.wav':'audio/wav'};
const csp="script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; connect-src 'self'; worker-src 'self'";
const server=http.createServer(async(req,res)=>{
  res.setHeader('Content-Security-Policy',csp);
  try {
    const route=new URL(req.url,'http://localhost').pathname;
    if(route==='/__harness.html') {res.setHeader('Content-Type','text/html');res.end('<!doctype html><meta charset="utf-8"><title>Speechworks browser acceptance</title><p>Testing local speech recognition.</p>');return;}
    if(route==='/__bad-models.json') {
      const bad=structuredClone(manifest);const id=bad.defaults.final;
      bad.models[id].sha256='0'.repeat(64);res.setHeader('Content-Type','application/json');res.end(JSON.stringify(bad));return;
    }
    const target=route==='/__jfk.wav'?fixture:path.resolve(root,'.'+decodeURIComponent(route));
    if(target!==fixture && !target.startsWith(root+path.sep)) {res.writeHead(403);res.end();return;}
    res.setHeader('Content-Type',contentTypes[path.extname(target)] || 'application/octet-stream');
    res.setHeader('Cache-Control','no-cache');
    res.end(await fs.readFile(target));
  } catch(error) {res.writeHead(404);res.end(String(error));}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`;
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const results={recordedAt:new Date().toISOString(),environment:{headless,browser:null,platform:os.platform(),architecture:os.arch(),cpu:os.cpus()[0]?.model,csp},artifacts:{wasmSha256:hash(await fs.readFile(path.join(root,'speechworks_web_bg.wasm'))),fixtureSha256:hash(await fs.readFile(fixture)),models:Object.fromEntries(ids.map(id=>[id,{sha256:manifest.models[id].sha256,bytes:manifest.models[id].bytes}]))},direct:[],fallback:[],worker:[],client:[],externalRequests:[],errors:[]};
let browser;
async function save(){await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,JSON.stringify(results,null,2)+'\n');}
const note=text=>process.stdout.write(text+'\n');
const serializeError=e=>({message:e.message,stack:e.stack});
async function pageForTest(){
  const context=await browser.newContext({permissions:['microphone']});
  context.on('request',request=>{if(!request.url().startsWith(origin+'/') && !request.url().startsWith('blob:') && !request.url().startsWith('data:'))results.externalRequests.push(request.url());});
  const page=await context.newPage();
  page.on('pageerror',e=>results.errors.push({pageError:e.message}));
  await page.goto(origin+'/__harness.html');
  await page.evaluate(async()=>{
    window.readFixture=async()=>{
      const bytes=await(await fetch('/__jfk.wav')).arrayBuffer();const v=new DataView(bytes);
      const four=i=>String.fromCharCode(...new Uint8Array(bytes,i,4));
      if(four(0)!=='RIFF' || four(8)!=='WAVE')throw new Error('Expected a RIFF/WAVE fixture');
      let format,data;
      for(let at=12;at+8<=bytes.byteLength;){const name=four(at),size=v.getUint32(at+4,true);if(name==='fmt ')format={type:v.getUint16(at+8,true),channels:v.getUint16(at+10,true),rate:v.getUint32(at+12,true),bits:v.getUint16(at+22,true)};if(name==='data')data=[at+8,size];at+=8+size+(size&1);}
      if(!format || format.type!==1 || format.channels!==1 || format.rate!==16000 || format.bits!==16 || !data)throw new Error('Fixture must be signed 16-bit mono at 16 kHz');
      const pcm=new Float32Array(data[1]/2);for(let i=0;i<pcm.length;i++)pcm[i]=v.getInt16(data[0]+i*2,true)/32768;return pcm;
    };
    window.sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  });
  return {context,page};
}
function normalized(s){return s.toLowerCase().replace(/[^a-z0-9\s]/g,'').replace(/\s+/g,' ').trim();}
try {
  browser=await chromium.launch({
    headless,
    ...(process.env.CHROME_EXECUTABLE ? {executablePath:process.env.CHROME_EXECUTABLE} : {}),
    args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',`--use-file-for-fake-audio-capture=${fixture}`],
  });
  results.environment.browser=browser.version();
  for(const backend of (runtimeOnly || fallbackOnly?[]:cpuOnly?['cpu']:['cpu','gpu'])) {
    for(const id of ids) {
      note(`Direct ${id} ${backend}`);
      const {context,page}=await pageForTest();
      try {
        const record=await page.evaluate(async({id,backend,repeats})=>{
          const started=performance.now();const wasm=await import('/speechworks_web.js');await wasm.default();
          const wasmMs=performance.now()-started;
          const manifest=await(await fetch('/models.json')).json();const spec=manifest.models[id];
          const fetchAt=performance.now();const bytes=new Uint8Array(await(await fetch(new URL(spec.url,location.href))).arrayBuffer());
          const fetchMs=performance.now()-fetchAt;
          const loadAt=performance.now();const model=await wasm.loadModel(bytes,backend);const loadMs=performance.now()-loadAt;
          const device=model.device();const fallback=model.fallbackReason();
          const adapter=backend==='gpu'?await navigator.gpu.requestAdapter():null;
          const info=adapter?.info;
          const adapterInfo=info?{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,isFallbackAdapter:adapter.isFallbackAdapter??info.isFallbackAdapter??null}:null;
          const pcm=await window.readFixture();const clips=[];
          try {
            // First full phrase includes lazy decoder pipeline setup; warm samples follow.
            const firstAt=performance.now();const firstText=await model.transcribe(pcm);const firstMs=performance.now()-firstAt;
            for(const seconds of [1,3,5,pcm.length/16000,3]) {
              const samples=pcm.slice(0,Math.min(pcm.length,seconds*16000));const times=[];let text;
              for(let n=0;n<repeats;n++){const at=performance.now();const next=await model.transcribe(samples);times.push(performance.now()-at);if(text!==undefined && text!==next)throw new Error(`Unstable repeat ${id}/${backend}/${seconds}`);text=next;}
              const sorted=[...times].sort((a,b)=>a-b);
              clips.push({audioSeconds:seconds,timesMs:times,medianMs:sorted[Math.floor(sorted.length/2)],p95Ms:sorted[Math.ceil(sorted.length*.95)-1],text});
            }
            return {id,backend,wasmMs,fetchMs,loadMs,device,fallback,adapterInfo,firstMs,firstText,clips};
          } finally {model.free();}
        },{id,backend,repeats});
        results.direct.push(record);
        assert.equal(record.fallback,undefined,'explicit backend should not fall back');
        if(backend==='gpu') {
          assert.match(record.device,/WebGPU/i,'actual inference must run through browser WebGPU');
          assert.doesNotMatch(record.device,/software|swiftshader|llvmpipe|CPU \(/i,'hardware GPU required');
          assert.notEqual(record.adapterInfo?.isFallbackAdapter,true,'software adapter is not hardware acceptance');
          const cpu=results.direct.find(x=>x.id===id && x.backend==='cpu');
          assert.equal(record.firstText,cpu.firstText,`${id} full CPU/GPU text`);
          for(let n=0;n<record.clips.length;n++)assert.equal(record.clips[n].text,cpu.clips[n].text,`${id} ${record.clips[n].audioSeconds}s CPU/GPU text`);
        } else assert.match(record.device,/CPU/);
        assert.match(normalized(record.firstText),/my fellow americans/);
        assert.match(normalized(record.firstText),/what you can do for your country/);
        await save();
        note(`${record.device}; first=${record.firstMs.toFixed(1)}ms; 1/3/5s warm medians=${record.clips.slice(0,3).map(x=>x.medianMs.toFixed(1)).join('/')}ms`);
      } finally {await context.close();}
    }
  }
  if(!runtimeOnly) {
    for(const reason of (cpuOnly?['unavailable']:['unavailable','lost'])) {
      note(`Auto fallback: ${reason}`);
      const {context,page}=await pageForTest();
      try {
        const record=await page.evaluate(async({id,reason})=>{
          const devices=[];
          if(reason==='unavailable')Object.defineProperty(navigator,'gpu',{value:undefined});
          else {
            // Capture and destroy the library's actual GPUDevice; inference is never mocked.
            const requestDevice=GPUAdapter.prototype.requestDevice;
            GPUAdapter.prototype.requestDevice=async function(...args){const device=await requestDevice.apply(this,args);devices.push(device);return device;};
          }
          const wasm=await import('/speechworks_web.js');await wasm.default();
          const manifest=await(await fetch('/models.json')).json();
          const bytes=new Uint8Array(await(await fetch(new URL(manifest.models[id].url,location.href))).arrayBuffer());
          const model=await wasm.loadModel(bytes,'auto');
          const samples=(await window.readFixture()).slice(0,48000);
          try {
            const before=model.device();let firstText;
            if(reason==='lost') {
              if(devices.length!==1 || !before.includes('WebGPU'))throw new Error('Auto did not select a real WebGPU device');
              firstText=await model.transcribe(samples);
              devices[0].destroy();await devices[0].lost;
            }
            const text=await model.transcribe(samples);
            return {reason,before,after:model.device(),fallback:model.fallbackReason(),firstText,text};
          } finally {model.free();}
        },{id:ids[0],reason});
        results.fallback.push(record);
        assert.match(record.after,/CPU/);
        assert.ok(record.fallback,'Auto must report why CPU was selected');
        if(reason==='lost')assert.equal(record.text,record.firstText,'CPU must replay the same phrase after real GPU loss');
        assert.match(normalized(record.text),/my fellow americans/);
        await save();
        note(`Auto ${reason}: ${record.after}; real CPU transcription succeeded`);
      } finally {await context.close();}
    }
  }
  if(!directOnly && !fallbackOnly) {
    for(const backend of (cpuOnly?['cpu']:['cpu','gpu'])) {
      note(`Worker ${backend} clip, live/final sessions, cancel, pinned hash rejection`);
      const {context,page}=await pageForTest();
      try {
        const record=await page.evaluate(async({backend,base})=>{
          const pcm=await window.readFixture();
          const wasm=await import('/speechworks_web.js');await wasm.default();
          const probe=new wasm.DictationSession({previews:false});
          const vad={frames:0,voiced:0,maxProbability:0,finals:0,finalSamples:0};
          try {
            const drain=()=>{let job;while((job=probe.nextFinal())){vad.finals++;vad.finalSamples+=job.samples.length;job.free();}};
            for(let at=0;at<pcm.length;at+=512){const frame=new Float32Array(512);frame.set(pcm.subarray(at,at+512));probe.push(frame);const probability=probe.speechProbability();vad.frames++;vad.voiced+=Number(probability>.3);vad.maxProbability=Math.max(vad.maxProbability,probability);drain();}
            probe.finish();drain();
            if(vad.voiced<vad.frames/4 || vad.maxProbability<.9 || !vad.finals || vad.finalSamples<pcm.length/2)throw new Error(`Actual WASM VAD failed: ${JSON.stringify(vad)}`);
          } finally {probe.free();}
          const events=[];let worker;let sequence=0;const requests=new Map();let active;
          const create=()=>{
            worker=new Worker('/worker-entry.js',{type:'module'});
            worker.onerror=e=>{for(const p of requests.values())p.reject(new Error(e.message));active?.reject(new Error(e.message));};
            worker.onmessage=({data:m})=>{
              events.push({...m,at:performance.now()});
              if(m.id && requests.has(m.id)){const p=requests.get(m.id);requests.delete(m.id);m.type==='error'?p.reject(new Error(m.message)):p.resolve(m);}
              if(active && m.session===active.id){if(m.type==='done'){active.resolve();active=null;}else if(m.type==='error'){active.reject(new Error(m.message));active=null;}}
            };
          };
          const call=(type,data={})=>new Promise((resolve,reject)=>{
            const id=++sequence;const timeout=setTimeout(()=>{requests.delete(id);reject(new Error(`${type} timeout`));},90000);
            requests.set(id,{resolve:m=>{clearTimeout(timeout);resolve(m);},reject:e=>{clearTimeout(timeout);reject(e);}});worker.postMessage({type,id,...data});
          });
          const session=async(id,previews,pace)=>{
            const start=performance.now();const completed=new Promise((resolve,reject)=>{active={id,resolve,reject};});
            await call('start',{session:id,config:{previews,soft_limit_ms:15000,hard_limit_ms:15000}});
            const feeding=performance.now();
            for(let at=0;at<pcm.length;at+=512){const samples=pcm.slice(at,at+512);worker.postMessage({type:'audio',session:id,samples,sampleRate:16000},[samples.buffer]);if(pace)await window.sleep(Math.max(0,feeding+(at+512)/16-performance.now()));}
            const stopped=performance.now();worker.postMessage({type:'stop',session:id});
            await Promise.race([completed,new Promise((_,reject)=>setTimeout(()=>reject(new Error('session completion timeout')),90000))]);
            const relevant=events.filter(e=>e.session===id);const finals=relevant.filter(e=>e.type==='final').sort((a,b)=>a.phrase-b.phrase);const live=relevant.filter(e=>e.type==='live' && e.text);
            if(!previews && live.length)throw new Error('Final-only session emitted live text');
            if(!finals.length)throw new Error(`VAD produced no final phrases from real speech: ${JSON.stringify(relevant.filter(e=>e.type!=='audio-ack'))}`);
            if(finals.some((e,n)=>e.phrase!==n))throw new Error('Missing final phrase');
            return {previews,firstLiveMs:live[0]?live[0].at-start:null,liveCount:live.length,finalCount:finals.length,text:finals.map(e=>e.text).filter(Boolean).join(' '),finalizationMs:performance.now()-stopped,events:relevant.filter(e=>['live','final','done','error'].includes(e.type)).map(e=>({...e,at:e.at-start}))};
          };
          try {
            create();await call('configure',{options:{manifestUrl:new URL('/models.json',location.href).href,backend,models:{draft:base,final:base}}});
            const preloadAt=performance.now();await call('preload');const preloadMs=performance.now()-preloadAt;
            const clipAt=performance.now();const clip=await call('transcribe',{samples:pcm,model:base});const clipMs=performance.now()-clipAt;
            const live=await session(101,true,true);
            const finalOnly=await session(102,false,false);
            await call('start',{session:103,config:{previews:true}});
            worker.postMessage({type:'audio',session:103,samples:pcm.slice(0,8000),sampleRate:16000});worker.postMessage({type:'cancel',session:103});
            const afterCancel=await call('transcribe',{samples:pcm,model:base});
            if(events.some(e=>e.session===103 && e.type==='done'))throw new Error('Cancelled session completed');
            worker.terminate();create();
            await call('configure',{options:{manifestUrl:new URL('/__bad-models.json',location.href).href,backend,models:{draft:base,final:base}}});
            let checksumRejected=false;try{await call('preload');}catch(error){checksumRejected=/checksum/.test(error.message);if(!checksumRejected)throw error;}
            if(!checksumRejected)throw new Error('Corrupt manifest hash accepted');
            return {backend,vad,preloadMs,clipMs,clipText:clip.text,live,finalOnly,cancelReplayText:afterCancel.text,checksumRejected};
          } finally{worker?.terminate();}
        },{backend,base:ids[1]});
        results.worker.push(record);
        assert.match(normalized(record.live.text),/what you can do for your country/,'session must preserve final words');
        assert.match(normalized(record.finalOnly.text),/what you can do for your country/);
        assert.equal(record.cancelReplayText,record.clipText);
        assert.ok(record.live.liveCount>0,'live mode must emit tentative text');
        await save();
        note(`Worker ${backend}: live=${record.live.liveCount}, finals=${record.live.finalCount}, stop-to-final=${record.live.finalizationMs.toFixed(1)}ms`);
      } finally{await context.close();}
      note(`Client ${backend} real fake-microphone capture through AudioWorklet`);
      const client=await pageForTest();
      try {
        const record=await client.page.evaluate(async({backend,base})=>{
          const {createDictation}=await import('/speechworks.js');
          const engine=createDictation({backend,models:{draft:base,final:base},session:{previews:false},stopTailMs:0});
          const changes=[],errors=[],status=[];engine.onStatus(s=>status.push(s));
          try {
            const pcm=await window.readFixture();await engine.preload();
            const clip=await engine.transcribe(pcm);const cached=await engine.isCached();
            const start=performance.now();const recording=await engine.record({onText:t=>changes.push({...t,at:performance.now()-start}),onError:e=>errors.push(e.message)});
            await window.sleep(pcm.length/16+250);
            const stop=performance.now();const text=await recording.stop();
            const secondStop=await recording.stop();
            if(text!==secondStop)throw new Error('stop() is not idempotent');
            if(errors.length)throw new Error(errors.join('; '));
            return {backend,clipText:clip,cached,text,stopMs:performance.now()-stop,changes,status};
          } finally {engine.destroy();}
        },{backend,base:ids[1]});
        results.client.push(record);
        assert.ok(record.cached,'verified weights should be cached');
        assert.match(record.status.find(s=>s.ready && s.device)?.device || '',backend==='gpu'?/WebGPU/:/CPU/,'client must report the requested compute backend');
        // Browser microphone processing can change inflection; require the spoken content.
        assert.match(normalized(record.text),/fellow american/);
        assert.match(normalized(record.text),/what you can do for your country/);
        assert.ok(record.changes.every(c=>c.live===''),'final-only microphone capture must not emit previews');
        await save();
        note(`Client ${backend}: capture succeeded; stop=${record.stopMs.toFixed(1)}ms`);
      } finally {await client.context.close();}
    }
  }
  assert.deepEqual(results.externalRequests,[],'browser must make no external requests');
  assert.deepEqual(results.errors,[],'browser must not report unhandled errors');
  results.passed=true;await save();note(`PASS ${output}`);
} catch(error) {results.passed=false;results.errors.push(serializeError(error));await save();throw error;}
finally {await browser?.close();await new Promise(resolve=>server.close(resolve));}
