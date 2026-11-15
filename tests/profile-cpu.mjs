// Separate sampling run. Profiling is not enabled in the latency benchmark.
// node tests/profile-cpu.mjs RUNTIME OUTPUT_PREFIX JFK.wav
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root=path.resolve(process.argv[2] || 'dist');
const out=path.resolve(process.argv[3] || 'test-results/mac-cpu-profile');
const fixture=path.resolve(process.argv[4] || process.env.SPEECHWORKS_TEST_AUDIO || 'jfk.wav');
const server=http.createServer(async(req,res)=>{
  try {
    const route=new URL(req.url,'http://localhost').pathname;
    if(route==='/') {res.setHeader('Content-Type','text/html');res.end('<title>Speechworks CPU profile</title>');return;}
    const file=route==='/fixture.wav'?fixture:path.resolve(root,'.'+decodeURIComponent(route));
    if(file!==fixture && !file.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}
    res.setHeader('Content-Type',({'.js':'text/javascript','.wasm':'application/wasm','.json':'application/json'})[path.extname(file)] || 'application/octet-stream');
    res.end(await fs.readFile(file));
  }catch(e){res.writeHead(404);res.end(String(e));}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
let browser;
try {
  browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_EXECUTABLE});
  for(const id of ['tiny','base']) {
    const page=await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(async id=>{
      const w=await import('/speechworks_web.js');await w.default();
      const manifest=await(await fetch('/models.json')).json();
      const key=Object.keys(manifest.models).find(x=>x.includes(id));
      window.model=await w.loadModel(new Uint8Array(await(await fetch(manifest.models[key].url)).arrayBuffer()),'cpu');
      const bytes=await(await fetch('/fixture.wav')).arrayBuffer();const v=new DataView(bytes);
      let at=12;while(String.fromCharCode(...new Uint8Array(bytes,at,4))!=='data')at+=8+v.getUint32(at+4,true)+(v.getUint32(at+4,true)&1);
      window.pcm=new Float32Array(v.getUint32(at+4,true)/2);for(let i=0;i<pcm.length;i++)pcm[i]=v.getInt16(at+8+i*2,true)/32768;
      await model.transcribe(pcm);await model.transcribe(pcm);
    },id);
    const cdp=await page.context().newCDPSession(page);
    await cdp.send('Profiler.enable');await cdp.send('Profiler.setSamplingInterval',{interval:1000});await cdp.send('Profiler.start');
    const text=await page.evaluate(async()=>{let t;for(let i=0;i<3;i++)t=await model.transcribe(pcm);return t;});
    const {profile}=await cdp.send('Profiler.stop');
    await fs.mkdir(path.dirname(out),{recursive:true});await fs.writeFile(`${out}-${id}.cpuprofile`,JSON.stringify(profile));
    const nodes=new Map(profile.nodes.map(n=>[n.id,n]));const counts=new Map();
    for(let i=0;i<profile.samples.length;i++){const name=nodes.get(profile.samples[i]).callFrame.functionName;counts.set(name,(counts.get(name)||0)+profile.timeDeltas[i]);}
    console.log(JSON.stringify({id,text,environment:await page.evaluate(()=>({crossOriginIsolated,hardwareConcurrency:navigator.hardwareConcurrency,userAgent:navigator.userAgent,device:model.device()})),totalMicros:profile.endTime-profile.startTime,top:[...counts].sort((a,b)=>b[1]-a[1]).slice(0,25)},null,2));
    await page.evaluate(()=>model.free());await page.close();
  }
} finally {await browser?.close();await new Promise(r=>server.close(r));}
