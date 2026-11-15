#!/usr/bin/env node
// Optional local-only COOP/COEP harness. Never changes a deployed application's headers.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const args=Object.fromEntries(process.argv.slice(2).map((x,i,a)=>x.startsWith('--')?[x,a[i+1]&&!a[i+1].startsWith('--')?a[i+1]:true]:null).filter(Boolean));
for(const key of ['--plain','--threaded','--models','--manifest','--fixture'])assert.equal(typeof args[key],'string',`${key} required`);
const roots={plain:path.resolve(args['--plain']),threaded:path.resolve(args['--threaded']),models:path.resolve(args['--models'])};
const fixture=path.resolve(args['--fixture']),manifest=path.resolve(args['--manifest']);
const threads=Number(args['--threads']||4),repeats=Number(args['--repeats']||5);
assert.ok(Number.isInteger(threads)&&threads>=1&&threads<=32);assert.ok(Number.isInteger(repeats)&&repeats>=1&&repeats<=10);
const isolated=!args['--non-isolated'];const injectStartupFailure=!!args['--inject-startup-failure'];
const output=path.resolve(args['--output']||`test-results/threads-${threads}-${isolated}.json`);
const workerFile=new URL('./threaded-worker.mjs',import.meta.url);
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const csp="script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; connect-src 'self'; style-src 'self'";
const server=http.createServer(async(req,res)=>{
  res.setHeader('Content-Security-Policy',csp);
  if(isolated){res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');}
  try {
    const route=new URL(req.url,'http://localhost').pathname;
    if(route==='/'){res.setHeader('Content-Type','text/html');res.end('<title>Speechworks optional CPU threads</title>');return;}
    let file;
    if(route==='/coordinator.mjs')file=fileURLToPath(workerFile);
    else if(route==='/jfk.wav')file=fixture;
    else if(route==='/models.json')file=manifest;
    else {
      const parts=route.split('/');const root=roots[parts[1]];if(!root)throw new Error('Unknown route');
      file=path.resolve(root,decodeURIComponent(parts.slice(2).join('/')));
      if(!file.startsWith(root+path.sep))throw new Error('Invalid path');
    }
    res.setHeader('Content-Type',({'.js':'text/javascript','.mjs':'text/javascript','.wasm':'application/wasm','.json':'application/json','.wav':'audio/wav'})[path.extname(file)]||'application/octet-stream');
    res.end(await fs.readFile(file));
  }catch(error){res.writeHead(404);res.end(String(error));}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`;let browser;
const results={environment:{platform:os.platform(),architecture:os.arch(),cpu:os.cpus()[0]?.model,isolated,csp},artifacts:{fixtureSha256:hash(await fs.readFile(fixture)),plainSha256:hash(await fs.readFile(path.join(roots.plain,'speechworks_web_bg.wasm'))),threadedSha256:hash(await fs.readFile(path.join(roots.threaded,'speechworks_web_bg.wasm')))},externalRequests:[],errors:[]};
try {
  browser=await chromium.launch({headless:true,...(process.env.CHROME_EXECUTABLE?{executablePath:process.env.CHROME_EXECUTABLE}:{})});
  results.environment.browser=browser.version();
  const page=await browser.newPage();page.on('pageerror',e=>results.errors.push(e.message));
  await page.context().route('**/*',route=>{if(!route.request().url().startsWith(origin+'/')){results.externalRequests.push(route.request().url());return route.abort();}return route.continue();});
  await page.goto(origin+'/');
  results.result=await page.evaluate(options=>new Promise((resolve,reject)=>{
    const worker=new Worker('/coordinator.mjs',{type:'module'});
    const timeout=setTimeout(()=>{worker.terminate();reject(new Error('Threaded harness timeout'));},180000);
    worker.onmessage=({data})=>{clearTimeout(timeout);worker.terminate();data.error?reject(new Error(data.error+'\n'+data.stack)):resolve(data.result);};
    worker.onerror=e=>{clearTimeout(timeout);worker.terminate();reject(new Error(e.message));};
    worker.postMessage(options);
  }),{threads,repeats,injectStartupFailure,forceThreaded:!!args['--force-threaded'],models:args['--model']?[String(args['--model'])]:['tiny','base'],prefixes:args['--full-only']?[11]:[1,3,5,11,3]});
  assert.equal(results.result.crossOriginIsolated,isolated);
  const expectThreaded=isolated&&(threads>1||!!args['--force-threaded'])&&!injectStartupFailure;
  assert.equal(results.result.selected,expectThreaded?'threaded':'plain');
  if(injectStartupFailure){assert.equal(results.result.workerEvidence.created,1);assert.equal(results.result.workerEvidence.terminated,1);assert.match(results.result.fallback,/Injected/);}
  for(const record of results.result.direct){assert.match(record.device,new RegExp(`, ${expectThreaded?threads:1} threads`));if(args['--expected-texts']){const expected=JSON.parse(await fs.readFile(args['--expected-texts'],'utf8'));assert.equal(record.firstText,expected[record.id]);}else{assert.match(record.firstText,/my fellow Americans/);assert.match(record.firstText,/what you can do for your country/);}}
  assert.deepEqual(results.errors,[]);assert.deepEqual(results.externalRequests,[]);results.passed=true;
} catch(error){results.errors.push(error.stack||String(error));throw error;}
finally {await browser?.close();await new Promise(resolve=>server.close(resolve));await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,JSON.stringify(results,null,2)+'\n');}
console.log(JSON.stringify(results.result,null,2));console.log(`PASS ${output}`);
