#!/usr/bin/env node
// Real-browser public API lifecycle acceptance; all fault injection is test-local.
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import os from 'node:os';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const args=Object.fromEntries(process.argv.slice(2).flatMap((v,i,a)=>v.startsWith('--')?[[v,a[i+1]]]:[]));
for(const k of ['--root','--fixture','--output'])assert.ok(args[k],`${k} required`);
const root=path.resolve(args['--root']), fixture=path.resolve(args['--fixture']);
const report={methodology:'Actual createDictation API in fresh isolated Chrome contexts. UI Worker handles instrumented; CDP worker targets observed after teardown. Fault injections are explicit, test-local and not throughput measurements.',cases:[],errors:[]};
report.environment={platform:os.platform(),architecture:os.arch(),cpu:os.cpus()[0]?.model};
report.artifactSha256={};
for(const file of ['speechworks.js','pool-owner.js','worker.js','runtime-loader.js','runtime-config.js','threaded/thread-pool.js','threaded/speechworks_web_bg.wasm'])report.artifactSha256[file]=createHash('sha256').update(await fs.readFile(path.join(root,file))).digest('hex');
report.fixtureSha256=createHash('sha256').update(await fs.readFile(fixture)).digest('hex');
async function server(isolated){const s=http.createServer(async(req,res)=>{try{
 if(isolated){res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');}
 res.setHeader('Content-Security-Policy',"script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; connect-src 'self'");
 const route=new URL(req.url,'http://localhost').pathname;
 if(route==='/'){res.setHeader('Content-Type','text/html');res.end('<title>Speechworks lifecycle acceptance</title>');return;}
 const file=route==='/fixture.wav'?fixture:path.resolve(root,'.'+decodeURIComponent(route));
 if(file!==fixture&&!file.startsWith(root+path.sep))throw new Error('Invalid path');
 res.setHeader('Content-Type',({'.js':'text/javascript','.wasm':'application/wasm','.json':'application/json','.wav':'audio/wav'})[path.extname(file)]||'application/octet-stream');res.end(await fs.readFile(file));
 }catch(e){res.writeHead(404);res.end(String(e));}});await new Promise(r=>s.listen(0,'127.0.0.1',r));return s;}
const servers=[await server(true),await server(false)];let browser;
try{
 browser=await chromium.launch({headless:true,...(process.env.CHROME_EXECUTABLE?{executablePath:process.env.CHROME_EXECUTABLE}:{})});
 report.browser=browser.version();const cdp=await browser.newBrowserCDPSession();const cdpEvents=[],workerIds=new Set();
 cdp.on('Target.targetCreated',({targetInfo:t})=>{if(t.type==='worker'){workerIds.add(t.targetId);cdpEvents.push({event:'created',id:t.targetId,url:t.url});}});
 cdp.on('Target.targetDestroyed',({targetId:id})=>{if(workerIds.has(id))cdpEvents.push({event:'destroyed',id});});
 await cdp.send('Target.setDiscoverTargets',{discover:true});
 const targets=async()=> (await cdp.send('Target.getTargets')).targetInfos.filter(t=>t.type==='worker').map(t=>({id:t.targetId,url:t.url}));
 for(const spec of [{name:'auto-repeat-recreate',threads:'auto'},{name:'explicit-two',threads:2},{name:'explicit-four',threads:4},{name:'nonisolated',threads:4,isolated:false},{name:'destroy-inference',threads:4},{name:'destroy-partial-startup',threads:4},{name:'mixed-ready-failure',threads:4},{name:'late-fatal',threads:4},{name:'watchdog-stall',threads:4}]){
  const cdpStart=cdpEvents.length;const context=await browser.newContext();const page=await context.newPage();const requests=[],errors=[];const s=servers[spec.isolated===false?1:0];const origin=`http://127.0.0.1:${s.address().port}`;
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
  await context.route('**/*',r=>r.request().url().startsWith(origin+'/')?r.continue():r.abort());
  await page.goto(origin);
  const result=await page.evaluate(async spec=>{
   const NativeWorker=Worker,nativeTimeout=setTimeout.bind(globalThis),nativeInterval=setInterval.bind(globalThis);const records=[],events=[],statuses=[],gaps=[],computeGaps=[];let engine,armed=false,action=false,computing=false,last=performance.now();
   const heartbeat=nativeInterval(()=>{const now=performance.now();gaps.push(now-last);if(computing)computeGaps.push(now-last);last=now;},10);
   globalThis.setTimeout=(fn,ms,...rest)=>nativeTimeout(fn,spec.name==='watchdog-stall'&&armed&&ms===30000?120:ms,...rest);
   function destroy(){engine.destroy();events.push({type:'destroy',at:performance.now()});}
   globalThis.Worker=class extends NativeWorker{
    constructor(url,options){super(url,options);const rec={url:String(url),name:options?.name||'coordinator',createdAt:performance.now(),terminated:false,ready:false};records.push(rec);this.record=rec;
     this.addEventListener('message',event=>{
      const type=event.data?.type;if(type==='speechworks:inference-start')computing=true;if(type==='speechworks:inference-end')computing=false;events.push({type,worker:rec.name,at:performance.now()});
      if(type==='speechworks_rayon_ready'){
       rec.ready=true;
       if(['destroy-partial-startup','mixed-ready-failure'].includes(spec.name)){
        if(records.filter(r=>r.name==='speechworks_wasm_rayon'&&r.ready).length>1){rec.readySuppressed=true;event.stopImmediatePropagation();}
        if(!action){action=true;if(spec.name==='destroy-partial-startup')nativeTimeout(destroy,0);else queueMicrotask(()=>this.dispatchEvent(new MessageEvent('message',{data:{type:'speechworks_rayon_error',message:'Injected mixed-readiness startup failure'}})));}
       }
      }
      if(type==='speechworks:inference-start'&&!action&&['destroy-inference','late-fatal','watchdog-stall'].includes(spec.name)){
       action=true;armed=true;
       if(spec.name==='watchdog-stall')for(const child of liveChildren)child.terminate();
       else nativeTimeout(()=>{
        if(spec.name==='destroy-inference')destroy();
        else liveChildren[0].dispatchEvent(new MessageEvent('message',{data:{type:'speechworks_rayon_error',message:'Injected post-ready child failure'}}));
       },20);
      }
     });
     if(rec.name==='speechworks_wasm_rayon')liveChildren.push(this);
    }
    terminate(){this.record.terminated=true;this.record.terminatedAt=performance.now();return super.terminate();}
   };
   const liveChildren=[];const {createDictation}=await import('/speechworks.js');
   const bytes=await(await fetch('/fixture.wav')).arrayBuffer();const v=new DataView(bytes);let pcm;
   for(let at=12;at+8<=bytes.byteLength;){const size=v.getUint32(at+4,true);if(String.fromCharCode(...new Uint8Array(bytes,at,4))==='data'){pcm=new Float32Array(size/2);for(let i=0;i<pcm.length;i++)pcm[i]=v.getInt16(at+8+2*i,true)/32768;}at+=8+size+(size&1);}
   if(!pcm)throw new Error('Missing fixture PCM');
   const make=()=>{engine=createDictation({backend:'cpu',cpuThreads:spec.threads,models:{draft:'base',final:'base'},session:{previews:false}});engine.onStatus(s=>statuses.push(s));};
   const bounded=p=>Promise.race([p,new Promise((_,reject)=>nativeTimeout(()=>reject(new Error('Acceptance operation timeout')),60000))]);
   const texts=[];let rejected=null;
   try{
    make();
    if(spec.name==='destroy-partial-startup'){
     try{await bounded(engine.preload());}catch(e){rejected=e.message;}
    }else{
     await bounded(engine.preload());
     try{texts.push(await bounded(engine.transcribe(pcm)));}catch(e){rejected=e.message;}
     if(spec.name==='auto-repeat-recreate'){
      texts.push(await bounded(engine.transcribe(pcm)));destroy();make();await bounded(engine.preload());texts.push(await bounded(engine.transcribe(pcm)));
     }
    }
   }finally{engine?.destroy();clearInterval(heartbeat);}
   return {records,events,statuses,texts,rejected,injectedAction:action,hardwareConcurrency:navigator.hardwareConcurrency,isolated:crossOriginIsolated,heartbeat:{computeSamples:computeGaps.length,maxComputeGapMs:Math.max(0,...computeGaps),samples:gaps.length,maxGapMs:Math.max(0,...gaps),p95GapMs:[...gaps].sort((a,b)=>a-b)[Math.floor(gaps.length*.95)]}};
  },spec);
  let remaining;for(let n=0;n<100;n++){remaining=await targets();const observed=cdpEvents.slice(cdpStart);const created=observed.filter(e=>e.event==='created');const ended=new Set(observed.filter(e=>e.event==='destroyed').map(e=>e.id));if(!remaining.length&&created.every(e=>ended.has(e.id)))break;await new Promise(r=>setTimeout(r,20));}
  const entry={...spec,result,requests:requests.map(u=>u.replace(origin,'')),errors,cdpWorkerEvents:cdpEvents.slice(cdpStart),remainingWorkerTargets:remaining};report.cases.push(entry);
  assert.equal(result.records.filter(r=>!r.terminated).length,0,`${spec.name} handles leaked`);assert.deepEqual(remaining,[],`${spec.name} CDP workers leaked`);const createdTargets=entry.cdpWorkerEvents.filter(e=>e.event==='created');const destroyedTargets=new Set(entry.cdpWorkerEvents.filter(e=>e.event==='destroyed').map(e=>e.id));assert.equal(createdTargets.length,result.records.length,'CDP observed every instrumented worker');assert.ok(createdTargets.every(e=>destroyedTargets.has(e.id)),'CDP confirmed every worker destruction');assert.deepEqual(errors,[]);
  assert.ok(requests.every(u=>u.startsWith(origin+'/')),'Unexpected external request');
  const children=result.records.filter(r=>r.name==='speechworks_wasm_rayon');
  if(spec.name==='nonisolated'){assert.equal(children.length,0);assert.ok(!requests.some(u=>u.includes('/threaded/')));}
  else if(spec.name==='auto-repeat-recreate'){const auto=Math.max(1,Math.min(4,Math.floor(result.hardwareConcurrency/2)));assert.equal(children.length,auto>1?auto*2:0);}
  else assert.equal(children.length,spec.threads);
  if(['destroy-inference','destroy-partial-startup','late-fatal','watchdog-stall'].includes(spec.name)){
   assert.equal(result.injectedAction,true);assert.ok(result.rejected);
   if(spec.name==='destroy-partial-startup')assert.equal(children.filter(c=>c.ready&&!c.readySuppressed).length,1);assert.notEqual(result.rejected,'Acceptance operation timeout');
   if(spec.name==='late-fatal')assert.match(result.rejected,/Injected/);
   if(spec.name==='watchdog-stall')assert.match(result.rejected,/exceeded/);
  }else{assert.equal(result.rejected,null);const expected=['nonisolated','mixed-ready-failure'].includes(spec.name)?1:spec.threads==='auto'?Math.max(1,Math.min(4,Math.floor(result.hardwareConcurrency/2))):spec.threads;assert.ok(result.statuses.some(s=>s.device?.includes(`, ${expected} threads`)));for(const text of result.texts){assert.match(text,/my fellow Americans/);assert.match(text,/what you can do for your country/);}assert.equal(new Set(result.texts).size,1);if(report.cases[0]!==entry)assert.equal(result.texts[0],report.cases[0].result.texts[0],'Cross-configuration transcript mismatch');}
  entry.passed=true;await context.close();console.log(`PASS ${spec.name} heartbeat max ${result.heartbeat.maxGapMs.toFixed(1)}ms`);
 }
 report.passed=true;
}catch(e){report.errors.push(e.stack||String(e));throw e;}
finally{await browser?.close();for(const s of servers)await new Promise(r=>s.close(r));await fs.mkdir(path.dirname(path.resolve(args['--output'])),{recursive:true});await fs.writeFile(args['--output'],JSON.stringify(report,null,2)+'\n');}
