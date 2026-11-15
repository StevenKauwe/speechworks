#!/usr/bin/env node
// Public API warm CPU latency, not a GPU comparison or idle-core measurement.
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import os from 'node:os';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const args=Object.fromEntries(process.argv.slice(2).flatMap((v,i,a)=>v.startsWith('--')?[[v,a[i+1]]]:[]));
for(const k of ['--root','--fixture','--output'])assert.ok(args[k],`${k} required`);
const root=path.resolve(args['--root']),fixture=path.resolve(args['--fixture']);
const repeats=Number(args['--repeats']||5);assert.ok(Number.isSafeInteger(repeats)&&repeats>0&&repeats<=10);
const variants=(args['--threads']||'auto,1,2,4').split(',').map(x=>x==='auto'?x:Number(x));
assert.ok(variants.every(x=>x==='auto'||Number.isSafeInteger(x)&&x>0&&x<=32));
const report={recordedAt:new Date().toISOString(),methodology:'Actual createDictation CPU API. Fresh browser context per thread setting, sequential runs; each tiny/base model gets a full JFK warmup before 1/3/5/11/3-second prefixes, five repetitions by default. Models remain loaded, initialization/downloads excluded from timed transcribe calls. Main-page heartbeat sampled every10ms during timed calls; logical CPU count is not an idle-core measurement.',environment:{platform:os.platform(),architecture:os.arch(),cpu:os.cpus()[0]?.model},artifactSha256:{},runs:[],errors:[]};
for(const file of ['speechworks.js','pool-owner.js','worker.js','runtime-loader.js','runtime-config.js','speechworks_web_bg.wasm','relaxed/speechworks_web_bg.wasm','threaded/thread-pool.js','threaded/speechworks_web_bg.wasm'])report.artifactSha256[file]=createHash('sha256').update(await fs.readFile(path.join(root,file))).digest('hex');
report.fixtureSha256=createHash('sha256').update(await fs.readFile(fixture)).digest('hex');
const server=http.createServer(async(req,res)=>{try{
 res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');res.setHeader('Content-Security-Policy',"script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; connect-src 'self'");
 const route=new URL(req.url,'http://localhost').pathname;
 if(route==='/'){res.setHeader('Content-Type','text/html');res.end('<title>Speechworks CPU benchmark</title>');return;}
 const file=route==='/fixture.wav'?fixture:path.resolve(root,'.'+decodeURIComponent(route));if(file!==fixture&&!file.startsWith(root+path.sep))throw new Error('Invalid path');
 res.setHeader('Content-Type',({'.js':'text/javascript','.wasm':'application/wasm','.json':'application/json'})[path.extname(file)]||'application/octet-stream');res.end(await fs.readFile(file));
 }catch(e){res.writeHead(404);res.end(String(e));}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin=`http://127.0.0.1:${server.address().port}`;let browser;
try{
 browser=await chromium.launch({headless:true,...(process.env.CHROME_EXECUTABLE?{executablePath:process.env.CHROME_EXECUTABLE}:{})});report.environment.browser=browser.version();
 for(const threads of variants){
  const context=await browser.newContext(),page=await context.newPage();const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
  await context.route('**/*',r=>r.request().url().startsWith(origin+'/')?r.continue():r.abort());await page.goto(origin);
  const result=await page.evaluate(async({threads,repeats})=>{
   const {createDictation}=await import('/speechworks.js');const statuses=[],records=[];const NativeWorker=Worker;
   globalThis.Worker=class extends NativeWorker{constructor(url,options){super(url,options);this.record={url:String(url),name:options?.name||'coordinator',terminated:false};records.push(this.record);}terminate(){this.record.terminated=true;return super.terminate();}};
   const bytes=await(await fetch('/fixture.wav')).arrayBuffer(),view=new DataView(bytes);let pcm;
   for(let at=12;at+8<=bytes.byteLength;){const size=view.getUint32(at+4,true);if(String.fromCharCode(...new Uint8Array(bytes,at,4))==='data'){pcm=new Float32Array(size/2);for(let i=0;i<pcm.length;i++)pcm[i]=view.getInt16(at+8+i*2,true)/32768;}at+=8+size+(size&1);}
   if(!pcm)throw new Error('Missing PCM fixture');
   const engine=createDictation({backend:'cpu',cpuThreads:threads,models:{final:'base',draft:'tiny'},session:{previews:false}});engine.onStatus(s=>statuses.push(s));
   const direct=[];let timing=false,last=performance.now();const gaps=[];const timer=setInterval(()=>{const now=performance.now();if(timing)gaps.push(now-last);last=now;},10);const start=performance.now();
   try{
    await engine.preload();const preloadMs=performance.now()-start;
    for(const id of ['tiny','base']){
     const firstAt=performance.now();const firstText=await engine.transcribe(pcm,{model:id});const firstMs=performance.now()-firstAt;const clips=[];
     for(const audioSeconds of [1,3,5,11,3]){
      const clip=pcm.slice(0,Math.min(pcm.length,audioSeconds*16000));const timesMs=[];let text;
      for(let i=0;i<repeats;i++){
       const at=performance.now();timing=true;const next=await engine.transcribe(clip,{model:id});timing=false;timesMs.push(performance.now()-at);
       if(text!==undefined&&text!==next)throw new Error('Unstable repeat text');text=next;
      }
      clips.push({audioSeconds,timesMs,medianMs:[...timesMs].sort((a,b)=>a-b)[Math.floor(timesMs.length/2)],text});
     }
     direct.push({id,firstMs,firstText,device:[...statuses].reverse().find(s=>s.model===id&&s.device)?.device,clips});
    }
    return {requestedThreads:threads,preloadMs,hardwareConcurrency:navigator.hardwareConcurrency,crossOriginIsolated,records,direct,heartbeat:{samples:gaps.length,maxGapMs:Math.max(0,...gaps),p95GapMs:[...gaps].sort((a,b)=>a-b)[Math.floor(gaps.length*.95)]},statuses};
   }finally{engine.destroy();clearInterval(timer);}
  },{threads,repeats});
  assert.deepEqual(errors,[]);assert.ok(requests.every(u=>u.startsWith(origin+'/')));assert.ok(result.records.every(r=>r.terminated));assert.equal(result.crossOriginIsolated,true);
  const expected=threads==='auto'?Math.max(1,Math.min(4,Math.floor(result.hardwareConcurrency/2))):threads;
  for(const d of result.direct){assert.ok(d.device.includes(`, ${expected} threads`));assert.match(d.firstText,/my fellow Americans/);if(report.runs.length){const ref=report.runs[0].result.direct.find(x=>x.id===d.id);assert.equal(d.firstText,ref.firstText);assert.deepEqual(d.clips.map(c=>c.text),ref.clips.map(c=>c.text));}}
  report.runs.push({result,requests:requests.map(u=>u.replace(origin,'')),errors,passed:true});console.log(`PASS ${threads}: ${result.direct.map(d=>`${d.id}=${d.clips[3].medianMs.toFixed(3)}ms`).join(' ')} heartbeatMax=${result.heartbeat.maxGapMs.toFixed(2)}ms`);await context.close();
 }
 report.passed=true;
}catch(e){report.errors.push(e.stack||String(e));throw e;}
finally{await browser?.close();await new Promise(r=>server.close(r));await fs.mkdir(path.dirname(path.resolve(args['--output'])),{recursive:true});await fs.writeFile(args['--output'],JSON.stringify(report,null,2)+'\n');}
