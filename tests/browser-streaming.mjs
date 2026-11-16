#!/usr/bin/env node
// Real WASM/WebGPU streaming acceptance and reproducible wall-clock timings.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const args=Object.fromEntries(process.argv.slice(2).flatMap((v,i,a)=>v.startsWith('--')?[[v,a[i+1]]]:[]));
const root=path.resolve(args['--root'] || 'dist');
const fixture=path.resolve(args['--fixture'] || 'jfk.wav');
const backend=args['--backend'] || 'cpu';
const variant=args['--variant'] || 'plain';
const threads=Number(args['--threads'] || 4);
assert.ok(['plain','relaxed','threaded'].includes(variant));
const wasmPath=variant==='plain'?'speechworks_web_bg.wasm':`${variant}/speechworks_web_bg.wasm`;
const models=(args['--models'] || 'streaming-medium,streaming-small').split(',');
const output=path.resolve(args['--output'] || 'test-results/browser-streaming.json');
const expected='And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.';
const manifest=JSON.parse(await fs.readFile(path.join(root,'models.json'),'utf8'));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const report={at:new Date().toISOString(),backend,variant,threads:variant==='threaded'?threads:1,platform:os.platform(),cpu:os.cpus()[0]?.model,
  wasmSha256:hash(await fs.readFile(path.join(root,wasmPath))),
  fixtureSha256:hash(await fs.readFile(fixture)),models:[],errors:[],externalRequests:[]};
assert.ok(['cpu','gpu'].includes(backend));
const server=http.createServer(async(req,res)=>{
  try {
    if(variant==='threaded') {res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');}
    const route=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    if(route==='/') { res.setHeader('Content-Type','text/html'); res.end('<!doctype html><title>Streaming acceptance</title>');return; }
    const file=route==='/fixture.wav'?fixture:path.resolve(root,'.'+route);
    if(file!==fixture && !file.startsWith(root+path.sep)) throw new Error('outside test root');
    res.setHeader('Content-Type',({'.js':'text/javascript','.wasm':'application/wasm','.json':'application/json','.wav':'audio/wav'})[path.extname(file)] || 'application/octet-stream');
    res.end(await fs.readFile(file));
  } catch(error) { res.writeHead(404);res.end(String(error)); }
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`;
let browser;
try {
  browser=await chromium.launch({headless:process.env.HEADLESS!=='0',
    ...(process.env.CHROME_EXECUTABLE?{executablePath:process.env.CHROME_EXECUTABLE}:{})});
  report.browser=browser.version();
  for(const id of models) {
    const context=await browser.newContext();
    const page=await context.newPage();
    page.on('pageerror',e=>report.errors.push(e.message));
    page.on('request',r=>{if(!r.url().startsWith(origin)) report.externalRequests.push(r.url());});
    await page.addInitScript(()=>{
      globalThis.testDevices=[];
      if(typeof GPUAdapter!=='undefined') {
        const request=GPUAdapter.prototype.requestDevice;
        GPUAdapter.prototype.requestDevice=async function(...args) { const device=await request.apply(this,args); testDevices.push(device);return device; };
      }
    });
    await page.goto(origin);
    const result=await page.evaluate(async({id,backend,expected,variant,threads})=>{
      const assert=(yes,message)=>{if(!yes) throw new Error(message);};
      const pool=variant==='threaded'?await import('/threaded/thread-pool.js'):null;
      const runtime=pool?.wasm || await import(variant==='relaxed'?'/relaxed/speechworks_web.js':'/speechworks_web.js');
      const exports=await runtime.default();
      if(pool) await runtime.initCpuThreadPool(threads);
      const manifest=await(await fetch('/models.json')).json();
      const bytes=new Uint8Array(await(await fetch(manifest.models[id].url)).arrayBuffer());
      const wav=await(await fetch('/fixture.wav')).arrayBuffer(),view=new DataView(wav);
      const four=i=>String.fromCharCode(...new Uint8Array(wav,i,4));
      let data;
      for(let at=12;at+8<=wav.byteLength;) {
        const size=view.getUint32(at+4,true);
        if(four(at)==='data') {data=new Int16Array(wav,at+8,size/2);break;}
        at+=8+size+(size&1);
      }
      const pcm=Float32Array.from(data,x=>x/32768);
      const load=performance.now();
      const recognizer=await runtime.loadModel(bytes,backend);
      const result={id,loadMs:performance.now()-load,device:recognizer.device(),offline:[],stream:[]};
      assert(backend==='cpu'?result.device.includes('CPU'):result.device.includes('WebGPU'),'requested backend is actually used');
      try {
        for(let run=0;run<2;run++) {
          const offlineAt=performance.now();
          const offline=await recognizer.transcribe(pcm);
          result.offline.push({ms:performance.now()-offlineAt,text:offline});
          assert(offline===expected,`${id} offline transcript: ${offline}`);
          const updates=[];let last='';
          for(let end=5120;end<pcm.length+5120;end+=5120) {
            const samples=pcm.subarray(0,Math.min(end,pcm.length));
            const start=performance.now();
            const text=await recognizer.transcribePhrase(samples,`stream-${run}`,false,samples.length);
            updates.push({audioMs:samples.length/16,ms:performance.now()-start,text});last=text;
          }
          const start=performance.now();
          const text=await recognizer.transcribePhrase(pcm,`stream-${run}`,true,pcm.length);
          const finalMs=performance.now()-start;
          assert(text===expected,`${id} streaming transcript: ${text}`);
          result.stream.push({updates,finalMs,text,lastPreview:last});
        }
        // A hard phrase split may move the final boundary behind a preview.
        // It must reset cached audio, and an unrelated recording must do likewise.
        await recognizer.transcribePhrase(pcm.subarray(0,41003),'rewind',false,41003);
        const short=pcm.subarray(0,32037);
        const wanted=await recognizer.transcribe(short);
        const rewound=await recognizer.transcribePhrase(short,'rewind',true,short.length);
        assert(rewound===wanted,'a shorter final phrase resets cached history');
        await recognizer.transcribePhrase(pcm.subarray(0,32037),'cancelled',false,32037);
        recognizer.resetStream();
        const restarted=await recognizer.transcribePhrase(pcm,'new-recording',true,pcm.length);
        assert(restarted===expected,'cancel/restart does not reuse earlier audio');
        result.rewindAndRestart=true;
      } finally { recognizer.free(); }
      if(backend==='gpu' && id==='streaming-medium') {
        const automatic=await runtime.loadModel(bytes,'auto');
        try {
          assert(automatic.device().includes('WebGPU'),'recovery test starts on GPU');
          await automatic.transcribePhrase(pcm.subarray(0,32000),'lost',false,32000);
          assert(testDevices.length>0,'captured test GPU device');
          testDevices.at(-1).destroy();
          const text=await automatic.transcribePhrase(pcm,'lost',true,pcm.length);
          assert(text===expected,'device loss replays the whole streaming phrase');
          assert(automatic.device().includes('CPU') && automatic.fallbackReason(),'CPU fallback is reported');
          result.recovery={text,device:automatic.device(),reason:automatic.fallbackReason()};
        } finally {automatic.free();}
      }
      result.wasmMemoryBytes=exports.memory.buffer.byteLength;
      if(pool) await pool.terminateWorkers();
      return result;
    },{id,backend,expected,variant,threads});
    result.model=manifest.models[id];report.models.push(result);
    const warm=result.stream.at(-1),times=warm.updates.map(x=>x.ms).sort((a,b)=>a-b);
    console.log(`${id} ${result.device}: offline ${result.offline.at(-1).ms.toFixed(1)} ms; preview median ${times[Math.floor(times.length/2)].toFixed(1)} ms; final ${warm.finalMs.toFixed(1)} ms; transcript/restart/recovery checks passed`);
    await context.close();
  }
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.externalRequests,[]);
} catch(error) { report.failure={message:error.message,stack:error.stack};throw error; }
finally {
  await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
  await browser?.close();await new Promise(resolve=>server.close(resolve));
}
