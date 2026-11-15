/*
 * Adapted from wasm-bindgen-rayon 1.3.0 workerHelpers.no-bundler.js.
 * Copyright 2022 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for details.
 *
 * Speechworks changes: same-origin module Workers (no blob/CSP relaxation),
 * bounded startup, explicit startup error propagation and failed-pool cleanup.
 */
const liveWorkers = new Map();
const pendingWaits = new Set();
const workerName = 'speechworks_wasm_rayon';
const startupTimeoutMs = 30000;
let owner = null;
let active = false;
let disposal = null;
let unsubscribe = null;
let errorHandler = null;

export function setWorkerOwner(next) {
  if(active || disposal) throw new Error('Cannot replace an active CPU worker owner');
  if(next !== null && (!next || typeof next.spawn !== 'function' || typeof next.dispose !== 'function' || typeof next.onError !== 'function')) throw new Error('Invalid CPU worker owner');
  owner = next;
}
export function setWorkerErrorHandler(handler) {
  if(handler !== null && typeof handler !== 'function') throw new Error('Invalid CPU worker error handler');
  errorHandler = handler;
}
function fatal(error) {
  if(!active) return;
  // Notify the coordinator/broker explicitly; it must reject pending inference.
  const failure = error instanceof Error ? error : new Error(String(error));
  const cleanup = terminateWorkers();
  try { errorHandler?.(failure); } finally { cleanup.catch(() => {}); }
}
function waitForReady(worker) {
  return new Promise((resolve, reject) => {
    let ready = false;
    const settle = () => {clearTimeout(timeout);pendingWaits.delete(cancel);};
    const detach = () => {
      settle();
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
      worker.removeEventListener('messageerror', onError);
    };
    const cancel = error => {detach();reject(error);};
    const fail = error => ready ? fatal(error) : cancel(error);
    const onError = event => fail(new Error(event.message || 'CPU worker failed'));
    const onMessage = ({data}) => {
      if(data?.type === 'speechworks_rayon_ready' && !ready) {ready=true;settle();resolve();}
      else if(data?.type === 'speechworks_rayon_error') fail(new Error(data.message));
    };
    const timeout = setTimeout(() => cancel(new Error('CPU worker startup timed out')),startupTimeoutMs);
    pendingWaits.add(cancel);
    liveWorkers.set(worker,detach);
    worker.addEventListener('message',onMessage);
    worker.addEventListener('error',onError);
    worker.addEventListener('messageerror',onError);
  });
}

export function terminateWorkers() {
  if(disposal) return disposal;
  const wasActive = active;
  active = false;
  for(const cancel of [...pendingWaits]) cancel(new Error('CPU worker pool terminated'));
  for(const [worker,detach] of liveWorkers) {detach();worker.terminate();}
  liveWorkers.clear();
  const detachOwner = unsubscribe;unsubscribe=null;
  try {detachOwner?.();} catch { /* Disposal must still reach the actual owner. */ }
  if(!wasActive || !owner) return Promise.resolve();
  // Owner controls real child handles outside the blocking WASM coordinator.
  disposal = Promise.resolve().then(() => owner.dispose()).finally(() => {disposal=null;});
  return disposal;
}

if(globalThis.name === workerName) {
  const onInit = async ({data}) => {
    if(data?.type !== 'speechworks_rayon_init') return;
    globalThis.removeEventListener('message',onInit);
    try {
      if(new URL(data.mainJS).origin !== globalThis.location.origin) throw new Error('CPU worker module must be same-origin');
      const pkg = await import(data.mainJS);
      pkg.initSync(data.init);
      globalThis.postMessage({type:'speechworks_rayon_ready'});
      pkg.wbg_rayon_start_worker(data.receiver);
    } catch(error) {
      globalThis.postMessage({type:'speechworks_rayon_error',message:error?.message || String(error)});
    }
  };
  globalThis.addEventListener('message',onInit);
}

export async function startWorkers(module,memory,builder) {
  if(active || disposal) throw new Error('CPU worker pool already initialized');
  const count=builder.numThreads();
  if(!Number.isInteger(count) || count<1 || count>32) throw new Error('Invalid CPU worker count');
  const url=new URL(import.meta.url);
  const mainJS=new URL(builder.mainJS(),url);
  if(mainJS.origin!==url.origin || (url.protocol!=='file:' && !['http:','https:'].includes(mainJS.protocol))) throw new Error('CPU worker module must be same-origin');
  const init={type:'speechworks_rayon_init',init:{module,memory},receiver:builder.receiver(),mainJS:mainJS.href};
  active=true;
  try {
    if(owner) {
      const subscription=owner.onError(fatal);
      if(subscription != null && typeof subscription !== 'function') throw new Error('Invalid owner error subscription');
      unsubscribe=subscription || null;
      if(!active) throw new Error('CPU worker owner failed before startup');
      await owner.spawn({url:url.href,count,init});
    } else {
      const pending=[];
      for(let n=0;n<count;n++) {
        const worker=new Worker(url,{type:'module',name:workerName});
        const ready=waitForReady(worker);
        pending.push(ready);
        ready.catch(() => {});
        worker.postMessage(init);
      }
      await Promise.all(pending);
    }
    if(!active) throw new Error('CPU worker pool terminated during startup');
    builder.build();
  } catch(error) {
    await terminateWorkers();
    throw error;
  }
}
