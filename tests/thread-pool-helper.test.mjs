import test from 'node:test';
import assert from 'node:assert/strict';
import {startWorkers,terminateWorkers,setWorkerOwner,setWorkerErrorHandler} from '../tools/wasm-rayon-worker.js';

async function scenario(mode,action) {
  const original=globalThis.Worker;const workers=[];let built=0;
  globalThis.Worker=class extends EventTarget {
    constructor(url,options) {
      super();
      if(mode==='constructor' && workers.length===1)throw new Error('constructor failed');
      assert.equal(url.protocol,'file:'); // no blob URL in the same-origin helper
      assert.equal(options.type,'module');
      this.id=workers.length;this.terminated=false;workers.push(this);
    }
    postMessage() {
      if(mode==='post')throw new Error('post failed');
      if(mode==='timeout')return;
      queueMicrotask(()=>{
        if(mode==='worker' && this.id===1) {
          const e=new Event('error');e.message='startup failed';this.dispatchEvent(e);
        } else this.dispatchEvent(new MessageEvent('message',{data:{type:'speechworks_rayon_ready'}}));
      });
    }
    terminate(){this.terminated=true;}
  };
  const builder={numThreads:()=>3,receiver:()=>0,mainJS:()=>import.meta.url,build:()=>{if(mode==='build')throw new Error('build failed');built++;}};
  try {await action(builder,workers,()=>built);}
  finally {await terminateWorkers();setWorkerOwner(null);setWorkerErrorHandler(null);globalThis.Worker=original;}
}

test('same-origin persistent workers initialize once and explicitly shut down',async()=>{
  await scenario('success',async(builder,workers,built)=>{
    await startWorkers({}, {}, builder);assert.equal(built(),1);assert.equal(workers.length,3);
    assert.ok(workers.every(w=>!w.terminated));
    await assert.rejects(startWorkers({}, {}, builder),/already initialized/);
    terminateWorkers();assert.ok(workers.every(w=>w.terminated));
  });
});
for(const mode of ['constructor','post','worker','build']) {
  test(`${mode} startup failure terminates all attempted workers`,async()=>{
    await scenario(mode,async(builder,workers)=>{
      await assert.rejects(startWorkers({}, {}, builder),/failed/);
      assert.ok(workers.length>0);assert.ok(workers.every(w=>w.terminated));
    });
  });
}
test('invalid counts do not spawn workers',async()=>{
  await scenario('success',async(builder,workers)=>{
    for(const count of [0,-1,33,1.5])await assert.rejects(startWorkers({}, {}, {...builder,numThreads:()=>count}),/Invalid/);
    assert.equal(workers.length,0);
  });
});

test('startup timeout terminates all attempted workers',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  try {
    await scenario('timeout',async(builder,workers)=>{
      const pending=startWorkers({}, {}, builder);
      t.mock.timers.tick(30000);
      await assert.rejects(pending,/timed out/);
      assert.equal(workers.length,3);assert.ok(workers.every(w=>w.terminated));
    });
  } finally {t.mock.timers.reset();}
});

for(const type of ['error','messageerror','message']) {
  test(`ready then ${type} reports fatal and terminates entire pool`,async()=>{
    await scenario('success',async(builder,workers)=>{
      const errors=[];setWorkerErrorHandler(e=>errors.push(e.message));
      await startWorkers({}, {}, builder);
      const event=type==='message'?new MessageEvent('message',{data:{type:'speechworks_rayon_error',message:'late failure'}}):new Event(type);
      if(type!=='message')event.message='late failure';
      workers[0].dispatchEvent(event);
      assert.deepEqual(errors,['late failure']);assert.ok(workers.every(w=>w.terminated));
      workers[0].dispatchEvent(event);assert.equal(errors.length,1);
    });
  });
}
test('external owner starts same-origin children and awaits idempotent disposal',async()=>{
  await scenario('success',async(builder,workers,built)=>{
    let callback,disposed=0,release;
    setWorkerOwner({spawn:async p=>{assert.equal(p.count,3);assert.equal(p.init.type,'speechworks_rayon_init');assert.equal(p.init.mainJS,import.meta.url);assert.equal(new URL(p.url).protocol,'file:');},onError:fn=>{callback=fn;return()=>{callback=null;};},dispose:()=>{disposed++;return new Promise(r=>release=r);}});
    await startWorkers({}, {}, builder);assert.equal(workers.length,0);assert.equal(built(),1);
    const first=terminateWorkers();assert.equal(first,terminateWorkers());
    await Promise.resolve();assert.equal(disposed,1);assert.equal(callback,null);
    release();await first;
  });
});
test('external startup rejection waits for acknowledged disposal',async()=>{
  await scenario('success',async builder=>{
    let disposed=false;
    setWorkerOwner({spawn:async()=>{throw new Error('broker failed');},onError:()=>()=>{},dispose:async()=>{await Promise.resolve();disposed=true;}});
    await assert.rejects(startWorkers({}, {}, builder),/broker failed/);assert.equal(disposed,true);
  });
});
test('external late fatal is propagated once and disposes owner',async()=>{
  await scenario('success',async builder=>{
    let callback,disposed=0;const errors=[];
    setWorkerOwner({spawn:async()=>{},onError:fn=>{callback=fn;return()=>{};},dispose:async()=>{disposed++;}});
    setWorkerErrorHandler(e=>errors.push(e.message));await startWorkers({}, {}, builder);
    callback(new Error('late broker failure'));callback(new Error('duplicate'));
    await terminateWorkers();assert.equal(disposed,1);assert.deepEqual(errors,['late broker failure']);
  });
});
test('foreign module origin is rejected before spawning',async()=>{
  await scenario('success',async(builder,workers)=>{
    await assert.rejects(startWorkers({}, {}, {...builder,mainJS:()=> 'https://foreign.invalid/pkg.js'}),/same-origin/);
    assert.equal(workers.length,0);
  });
});
