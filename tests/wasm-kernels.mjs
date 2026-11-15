#!/usr/bin/env node
// Executes the production SIMD source with scalar/f64 numerical oracles in actual WASM.
import {execFileSync} from 'node:child_process';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const temp=await mkdtemp(join(tmpdir(),'speechworks-kernels-'));
const relaxed=process.argv.includes('--relaxed');
const traps=relaxed?8:5;
let browser;
try {
  const output=join(temp,'kernels.wasm');
  execFileSync('rustc',['--edition=2024','--target','wasm32-unknown-unknown','--crate-type=cdylib','-O','-C',`target-feature=+simd128,${relaxed?'+':'-'}relaxed-simd`,resolve('tests/wasm-kernels.rs'),'-o',output],{stdio:'inherit'});
  const bytes=await readFile(output);
  const {instance}=await WebAssembly.instantiate(bytes,{});
  const checks=instance.exports.run_tests();assert.ok(checks>800);
  for(let i=0;i<traps;i++)assert.throws(()=>instance.exports.reject_invalid(i),WebAssembly.RuntimeError);
  console.log(`Node WASM ${relaxed?'relaxed':'plain'} SIMD: ${checks} numerical assertions and ${traps} bounds traps passed`);
  if(process.argv.includes('--browser')) {
    const require=createRequire(import.meta.url);
    const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
    browser=await chromium.launch({headless:true,...(process.env.CHROME_EXECUTABLE?{executablePath:process.env.CHROME_EXECUTABLE}:{})});
    const page=await browser.newPage();
    const result=await page.evaluate(async({bytes,traps})=>{
      const {instance}=await WebAssembly.instantiate(new Uint8Array(bytes),{});
      const checks=instance.exports.run_tests();
      for(let i=0;i<traps;i++) {
        let trapped=false;try{instance.exports.reject_invalid(i);}catch(e){if(!(e instanceof WebAssembly.RuntimeError))throw e;trapped=true;}
        if(!trapped)throw new Error(`Invalid input ${i} did not trap`);
      }
      return checks;
    },{bytes:[...bytes],traps});
    assert.equal(result,checks);console.log(`Chrome ${browser.version()}: ${result} numerical assertions and ${traps} bounds traps passed`);
  }
}finally{await browser?.close();await rm(temp,{recursive:true,force:true});}
