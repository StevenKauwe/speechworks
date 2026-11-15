import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createPoolOwner, createWorkerOwner} from '../web/pool-owner.js';
import {createDictation} from '../web/speechworks.js';

const tick = () => new Promise(resolve => setImmediate(resolve));
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const payload = {url: 'https://example.test/app/threaded/thread-pool.js', count: 2,
  init: {type: 'speechworks_rayon_init', mainJS: 'https://example.test/app/threaded/speechworks_web.js'}};
class FakeWorker {
  constructor(url, options) { this.url = String(url); this.options = options; this.listeners = new Map(); this.terminated = 0; }
  addEventListener(type, fn) { this.listeners.set(type, fn); }
  removeEventListener(type, fn) { if (this.listeners.get(type) === fn) this.listeners.delete(type); }
  postMessage(value) { this.init = value; }
  terminate() { this.terminated++; }
  emit(type, data) { this.listeners.get(type)?.(type === 'message' ? {data} : {message: 'child failed', preventDefault() {}}); }
  ready() { this.emit('message', {type: 'speechworks_rayon_ready'}); }
}
function harness(options = {}) {
  const children = [], messages = [], errors = [];
  const owner = createPoolOwner({workerUrl: new URL('https://example.test/app/worker-entry.js'),
    postMessage: value => messages.push(value), onFatal: error => errors.push(error),
    createWorker: (url, opts) => { const child = new FakeWorker(url, opts); children.push(child); return child; },
    ...options});
  const spawn = data => owner.handleMessage({type: 'speechworks:pool-spawn', id: 1, payload: data || payload});
  return {owner, children, messages, errors, spawn};
}
test('owner starts exact module workers and waits for every ready message', async () => {
  const h = harness();
  h.spawn();
  assert.equal(h.owner.size, 2);
  for (const child of h.children) {
    assert.deepEqual(child.options, {type: 'module', name: 'speechworks_wasm_rayon'});
    assert.equal(child.init, payload.init);
  }
  h.children[0].ready(); await tick(); assert.equal(h.messages.length, 0);
  h.children[1].ready(); await tick();
  assert.deepEqual(h.messages, [{type: 'speechworks:pool-result', id: 1, ok: true}]);
  h.owner.dispose(); h.owner.dispose();
  assert.ok(h.children.every(child => child.terminated === 1 && child.listeners.size === 0));
});
test('startup child failure removes all workers before returning an error', async () => {
  const h = harness(); h.spawn(); h.children[0].emit('error'); await tick();
  assert.equal(h.owner.size, 0); assert.ok(h.children.every(child => child.terminated === 1));
  assert.equal(h.messages[0].ok, false); assert.equal(h.errors.length, 0);
});
test('constructor failure terminates already created children', async () => {
  const children = [];
  const h = harness({createWorker: (url, options) => {
    if (children.length) throw new Error('constructor failed');
    const child = new FakeWorker(url, options); children.push(child); return child;
  }});
  h.spawn(); await tick();
  assert.equal(children[0].terminated, 1); assert.match(h.messages[0].message, /constructor failed/);
});
test('postMessage failure terminates all owned workers', async () => {
  const child = new FakeWorker();
  child.postMessage = () => { throw new Error('clone failed'); };
  const h = harness({createWorker: () => child}); h.spawn(); await tick();
  assert.equal(child.terminated, 1); assert.match(h.messages[0].message, /clone failed/);
});
test('startup timeout cleans up before fallback response', async () => {
  const h = harness({startupTimeoutMs: 5}); h.spawn(); await delay(20);
  assert.equal(h.owner.size, 0); assert.match(h.messages[0].message, /startup timed out/);
});
test('destroy during startup immediately cleans children and suppresses stale replies', async () => {
  const h = harness(); h.spawn(); h.owner.dispose(); await tick();
  assert.equal(h.owner.size, 0); assert.equal(h.messages.length, 0);
  assert.ok(h.children.every(child => child.terminated === 1));
});
for (const kind of ['error', 'messageerror', 'message']) {
  test(`ready-then-${kind} is fatal and explicitly disposes every child once`, async () => {
    const h = harness(); h.spawn(); h.children.forEach(child => child.ready()); await tick();
    h.children[0].emit(kind, {type: 'speechworks_rayon_error', message: 'late failure'});
    assert.equal(h.owner.size, 0); assert.equal(h.errors.length, 1);
    h.owner.dispose(); assert.ok(h.children.every(child => child.terminated === 1));
  });
}
test('inference watchdog is independent of the blocked coordinator', async () => {
  const h = harness({inferenceTimeoutMs: 5}); h.spawn(); h.children.forEach(child => child.ready()); await tick();
  await delay(15); assert.equal(h.errors.length, 0); // No timeout on idle/model downloads.
  h.owner.handleMessage({type: 'speechworks:inference-start', token: 1}); await delay(20);
  assert.equal(h.errors.length, 1); assert.match(h.errors[0].message, /operation exceeded/);
  assert.equal(h.owner.size, 0);
});
test('matching compute end disarms watchdog, stale end does not', async () => {
  const h = harness({inferenceTimeoutMs: 5}); h.spawn(); h.children.forEach(child => child.ready()); await tick();
  h.owner.handleMessage({type: 'speechworks:inference-start', token: 1});
  h.owner.handleMessage({type: 'speechworks:inference-end', token: 1});
  await delay(15); assert.equal(h.errors.length, 0);
  h.owner.handleMessage({type: 'speechworks:inference-start', token: 2});
  h.owner.handleMessage({type: 'speechworks:inference-end', token: 1});
  await delay(15); assert.equal(h.errors.length, 1);
});
test('owner rejects foreign paths and invalid counts without creating workers', async () => {
  for (const change of [{url: 'https://other.test/threaded/x.js'}, {count: 0}, {count: 33},
    {init: {...payload.init, mainJS: 'https://other.test/x.js'}}]) {
    const h = harness(); h.spawn({...payload, ...change}); await tick();
    assert.equal(h.children.length, 0); assert.equal(h.messages[0].ok, false);
  }
});
test('coordinator adapter resolves startup and awaits acknowledged disposal', async () => {
  const messages = [];
  const owner = createWorkerOwner(message => messages.push(message));
  const pending = owner.spawn(payload);
  assert.equal(messages[0].type, 'speechworks:pool-spawn');
  owner.handleMessage({type: 'speechworks:pool-result', id: messages[0].id, ok: true});
  await pending;
  const disposed = owner.dispose();
  owner.handleMessage({type: 'speechworks:pool-result', id: messages[1].id, ok: true});
  await disposed;
});
test('client destroy/recreate owns children even while coordinator cannot reply', async t => {
  const previous = globalThis.Worker;
  const children = [], coordinators = [], order = [];
  globalThis.Worker = class extends FakeWorker {
    constructor(url, options) {
      super(url, options);
      if (options.name) children.push(this); else coordinators.push(this);
    }
    postMessage(message) {
      if (this.options.name) { queueMicrotask(() => this.ready()); return; }
      if (message.type === 'configure') queueMicrotask(() => {
        assert.equal(message.options.poolOwner, 'ui-v1');
        this.onmessage({data: {type: 'speechworks:pool-spawn', id: 99, payload}});
        this.onmessage({data: {type: 'configured', id: message.id}});
      });
      // Simulate a coordinator blocked forever in transcribe: no response.
    }
    terminate() { order.push(this.options.name ? 'child' : 'coordinator'); super.terminate(); }
  };
  t.after(() => { globalThis.Worker = previous; });
  const engine = createDictation({workerUrl: 'https://example.test/app/worker-entry.js'});
  const first = engine.transcribe(new Float32Array(16));
  const firstRejected = assert.rejects(first, /closed/);
  await tick(); engine.destroy(); await firstRejected;
  assert.deepEqual(order, ['child', 'child', 'coordinator']);
  const second = engine.transcribe(new Float32Array(16));
  const secondRejected = assert.rejects(second, /child failed/);
  await tick(); children[2].emit('error'); await secondRejected;
  assert.equal(coordinators.length, 2);
  assert.ok([...children, ...coordinators].every(worker => worker.terminated === 1));
});

test('ready child failing while another starts rejects startup rather than killing engine', async () => {
  const h = harness(); h.spawn();
  h.children[0].ready();
  h.children[0].emit('error');
  await tick();
  assert.equal(h.errors.length, 0);
  assert.equal(h.owner.size, 0);
  assert.equal(h.messages[0].ok, false);
  assert.ok(h.children.every(child => child.terminated === 1));
});
test('same-turn destroy/recreate cannot move old transcribe onto new worker', async t => {
  const previous = globalThis.Worker, workers = [];
  globalThis.Worker = class {
    constructor() { workers.push(this); this.messages = []; }
    postMessage(message) {
      this.messages.push(message);
      queueMicrotask(() => this.onmessage({data: {id: message.id,
        type: message.type === 'configure' ? 'configured' : message.type === 'preload' ? 'loaded' : 'result',
        text: 'fresh'}}));
    }
    terminate() {}
  };
  t.after(() => { globalThis.Worker = previous; });
  const engine = createDictation();
  await engine.preload();
  const old = engine.transcribe(new Float32Array([1]));
  const oldRejected = assert.rejects(old, /closed or replaced/);
  engine.destroy();
  const fresh = engine.transcribe(new Float32Array([2]));
  await oldRejected;
  assert.equal(await fresh, 'fresh');
  assert.equal(workers[1].messages.filter(x => x.type === 'transcribe').length, 1);
  assert.equal(workers[1].messages.find(x => x.type === 'transcribe').samples[0], 2);
  engine.destroy();
});
test('stale preload cannot send work or clear new generation readiness', async t => {
  const previous = globalThis.Worker, workers = [];
  globalThis.Worker = class {
    constructor() { workers.push(this); this.messages = []; }
    postMessage(message) {
      this.messages.push(message);
      queueMicrotask(() => this.onmessage({data: {id: message.id, type: message.type === 'configure' ? 'configured' : 'loaded'}}));
    }
    terminate() {}
  };
  t.after(() => { globalThis.Worker = previous; });
  const engine = createDictation();
  await engine.transcribe(new Float32Array([1]));
  const old = engine.preload();
  const oldRejected = assert.rejects(old, /closed or replaced/);
  engine.destroy();
  const status = []; engine.onStatus(value => status.push(value));
  await engine.preload(); await oldRejected;
  assert.equal(workers[1].messages.filter(x => x.type === 'preload').length, 1);
  assert.equal(status.at(-1).ready, true);
  engine.destroy();
});
