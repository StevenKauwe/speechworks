import assert from 'node:assert/strict';
import {test} from 'node:test';
import fs from 'node:fs/promises';
import {createRuntimeLoader, runtimeCapabilities, cpuThreadOption} from '../web/runtime-loader.js';
import {createWorkerOwner} from '../web/pool-owner.js';
import {createDictation} from '../web/speechworks.js';

const owner = {spawn() {}, dispose() {}};
const supported = {relaxedSimd: true, isolated: true, sharedArrayBuffer: true, sharedMemory: true};
function harness(caps = supported, failures = {}) {
  const calls = [];
  const module = name => ({
    default: async () => { calls.push(`init:${name}`); if (failures[`init:${name}`]) throw new Error('init failed'); },
    loadModel() {},
    DictationSession: class {},
    initCpuThreadPool: async count => { calls.push(`pool:${count}`); if (failures.pool) throw new Error('pool failed'); },
  });
  const load = createRuntimeLoader({config: {threadedAvailable: true},
    capabilities: () => caps,
    importModule: async path => {
      calls.push(path);
      const name = path.includes('threaded/') ? 'threaded' : path.includes('relaxed/') ? 'relaxed' : 'plain';
      if (failures[name]) throw new Error('artifact missing');
      return name === 'threaded' ? {wasm: module(name), requiresRelaxedSimd: false, setWorkerOwner() {}, terminateWorkers: async () => {
        calls.push('cleanup'); if (failures.cleanup) throw new Error('cleanup failed');
      }} : module(name);
    },
  });
  return {load: (options, context = {workerOwner: owner}) => load(options, context), calls};
}

test('relaxed SIMD uses a validating minimal instruction probe', () => {
  let probe;
  const caps = runtimeCapabilities({WebAssembly: {validate(bytes) { probe = bytes; return true; }}});
  assert.equal(caps.relaxedSimd, true);
  assert.deepEqual([...probe.slice(0, 8)], [0, 97, 115, 109, 1, 0, 0, 0]);
  assert.ok(Buffer.from(probe).includes(Buffer.from([0xfd, 0x85, 0x02])));
  assert.equal(caps.sharedMemory, false);
});

test('capability probing tolerates missing or throwing APIs', () => {
  assert.equal(runtimeCapabilities({}).relaxedSimd, false);
  const caps = runtimeCapabilities({crossOriginIsolated: true, SharedArrayBuffer,
    WebAssembly: {validate() { throw new Error('disabled'); }, Memory: class { constructor() { throw new Error('disabled'); }}}});
  assert.equal(caps.relaxedSimd, false);
  assert.equal(caps.sharedMemory, false);
});

test('nonisolated pages never even create shared WASM memory', () => {
  let attempted = false;
  runtimeCapabilities({crossOriginIsolated: false, SharedArrayBuffer,
    WebAssembly: {validate: () => false, Memory: class { constructor() { attempted = true; }}}});
  assert.equal(attempted, false);
});

test('default stays single-threaded and selects relaxed SIMD', async () => {
  const h = harness();
  assert.equal((await h.load()).variant, 'relaxed');
  assert.deepEqual(h.calls, ['./relaxed/speechworks_web.js', 'init:relaxed']);
});

test('unsupported relaxed SIMD never requests that artifact', async () => {
  const h = harness({...supported, relaxedSimd: false});
  assert.equal((await h.load()).variant, 'plain');
  assert.deepEqual(h.calls, ['./speechworks_web.js', 'init:plain']);
});

for (const absent of ['isolated', 'sharedArrayBuffer', 'sharedMemory']) {
  test(`thread request falls back without importing threads when ${absent} is absent`, async () => {
    const h = harness({...supported, [absent]: false});
    const result = await h.load({cpuThreads: 4});
    assert.equal(result.cpuThreads, 1);
    assert.match(result.fallbackReason, /requires cross-origin isolation/);
    assert.equal(h.calls.some(x => x.includes('threaded')), false);
  });
}

test('eligible opt-in initializes the pool before returning bindings', async () => {
  const h = harness();
  const result = await h.load({cpuThreads: 4});
  assert.equal(result.variant, 'threaded');
  assert.equal(result.cpuThreads, 4);
  assert.deepEqual(h.calls, ['./threaded/thread-pool.js', 'init:threaded', 'pool:4']);
});

for (const failure of ['pool', 'init:threaded']) {
  test(`${failure} failure cleans up before separate serial startup`, async () => {
    const h = harness(supported, {[failure]: true});
    const result = await h.load({cpuThreads: 4});
    assert.equal(result.variant, 'relaxed');
    assert.match(result.fallbackReason, /Threaded CPU unavailable/);
    assert.ok(h.calls.indexOf('cleanup') < h.calls.indexOf('./relaxed/speechworks_web.js'));
  });
}

test('missing optional threaded artifact falls back to serial', async () => {
  const h = harness(supported, {threaded: true});
  assert.equal((await h.load({cpuThreads: 2})).variant, 'relaxed');
  assert.equal(h.calls.includes('cleanup'), false);
});

test('cleanup failure stops rather than abandoning an orphan pool', async () => {
  const h = harness(supported, {pool: true, cleanup: true});
  await assert.rejects(h.load({cpuThreads: 2}), /cleanup failed/);
  assert.equal(h.calls.includes('./relaxed/speechworks_web.js'), false);
});

test('missing cleanup export prevents pool initialization', async () => {
  let initialized = false;
  const load = createRuntimeLoader({config: {threadedAvailable: true},capabilities: () => supported, importModule: async path => {
    const wasm = {default: async () => { initialized = true; }, loadModel() {}, DictationSession: class {}};
    return path.includes('threaded') ? {wasm} : {...wasm, default: async () => {}};
  }});
  assert.equal((await load({cpuThreads: 2}, {workerOwner: owner})).variant, 'relaxed');
  assert.equal(initialized, false);
});

for (const failure of ['relaxed', 'init:relaxed']) {
  test(`${failure} failure falls back to plain before exposing any model`, async () => {
    const h = harness(supported, {[failure]: true});
    const result = await h.load();
    assert.equal(result.variant, 'plain');
    assert.match(result.fallbackReason, /Relaxed SIMD unavailable/);
  });
}

test('plain runtime failure is retained, not repeatedly initialized', async () => {
  const h = harness({...supported, relaxedSimd: false}, {'init:plain': true});
  const pending = h.load();
  await assert.rejects(pending, /init failed/);
  assert.equal(h.load(), pending);
  await assert.rejects(h.load(), /init failed/);
  assert.equal(h.calls.length, 2);
});

test('invalid bounds do not import anything or poison a later valid request', async () => {
  const h = harness();
  for (const cpuThreads of [0, -1, 33, 1.5, NaN, Infinity, '4', null]) {
    await assert.rejects(h.load({cpuThreads}), /integer between 1 and 32/);
  }
  for (const value of [null, [], 'options']) await assert.rejects(h.load(value), /options must be an object/);
  assert.equal(h.calls.length, 0);
  assert.equal((await h.load({cpuThreads: 32})).cpuThreads, 32);
});

test('concurrent and later calls initialize once, with no memory rebinding', async () => {
  const h = harness();
  const first = h.load({cpuThreads: 2});
  assert.equal(h.load({cpuThreads: 2}), first);
  const result = await first;
  assert.equal(await h.load({cpuThreads: 2}), result);
  await assert.rejects(h.load({cpuThreads: 3}), /cannot change/);
  assert.deepEqual(h.calls, ['./threaded/thread-pool.js', 'init:threaded', 'pool:2']);
});

test('worker emits channel-ready before configuration, and awaits runtime for sessions', async t => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const events = [];
  let resolveRuntime;
  let constructions = 0;
  globalThis.__speechworksRuntimeTest = {
    cpuThreadOption, createWorkerOwner,
    loadRuntime: () => new Promise(resolve => { resolveRuntime = resolve; }),
  };
  globalThis.self = {location: {href: 'https://example.test/worker.js', origin: 'https://example.test'}, postMessage: x => events.push(x)};
  globalThis.fetch = async () => ({ok: true, json: async () => ({schema: 1, sampleRate: 16000, models: {}, defaults: {}})});
  t.after(() => { globalThis.self = previousSelf; globalThis.fetch = previousFetch; delete globalThis.__speechworksRuntimeTest; });
  const source = (await fs.readFile(new URL('../web/worker.js', import.meta.url), 'utf8'))
    .replace("import {loadRuntime, cpuThreadOption} from './runtime-loader.js';", 'const {loadRuntime, cpuThreadOption, createWorkerOwner} = globalThis.__speechworksRuntimeTest;')
    .replace("from './model-cache.js'", `from '${new URL('../web/model-cache.js',import.meta.url).href}'`)
    .replace("import {createWorkerOwner} from './pool-owner.js';", '');
  const worker = await import(`data:text/javascript;base64,${Buffer.from(source.replaceAll("new URL('./models.json', import.meta.url)", "new URL('https://example.test/models.json')")).toString('base64')}`);
  worker.startWorker();
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(events, [{type: 'ready'}]);
  const configure = self.onmessage({data: {type: 'configure', id: 1, options: {cpuThreads: 2}}});
  assert.equal(constructions, 0);
  resolveRuntime({loadModel() {}, DictationSession: class {constructor() { constructions++; }}});
  await configure;
  await self.onmessage({data: {type: 'configure', id: 3, options: {cpuThreads: null}}});
  assert.match(events.find(x => x.id === 3).message, /integer between 1 and 32/);
  await self.onmessage({data: {type: 'configure', id: 4, options: {cpuThreads: 3}}});
  assert.match(events.find(x => x.id === 4).message, /cannot change/);
  await self.onmessage({data: {type: 'start', id: 2, session: 1}});
  assert.equal(constructions, 1);
  assert.ok(events.some(x => x.type === 'configured'));
  assert.ok(events.some(x => x.type === 'started'));
});

test('client forwards cpuThreads through the existing configure protocol', async t => {
  const previous = globalThis.Worker;
  const sent = [];
  globalThis.Worker = class {
    postMessage(message) {
      sent.push(message);
      queueMicrotask(() => this.onmessage({data: {id: message.id, type: message.type === 'configure' ? 'configured' : 'loaded'}}));
    }
    terminate() {}
  };
  t.after(() => { globalThis.Worker = previous; });
  const speech = createDictation({cpuThreads: 4});
  await speech.preload();
  speech.destroy();
  assert.equal(sent.find(x => x.type === 'configure').options.cpuThreads, 4);
});

for (const requirement of [true, undefined]) {
  test(`threaded SIMD requirement ${requirement} is checked before WASM initialization`, async () => {
    let initialized = false;
    const load = createRuntimeLoader({config: {threadedAvailable: true},
      capabilities: () => ({...supported, relaxedSimd: false}),
      importModule: async path => {
        const serial = {default: async () => {}, loadModel() {}, DictationSession: class {}};
        return path.includes('threaded') ? {
          requiresRelaxedSimd: requirement,
          terminateWorkers() {},
          wasm: {...serial, default: async () => { initialized = true; }, initCpuThreadPool() {}},
        } : serial;
      },
    });
    const result = await load({cpuThreads: 2}, {workerOwner: owner});
    assert.equal(result.variant, 'plain');
    assert.equal(initialized, false);
    assert.match(result.fallbackReason, /SIMD/);
  });
}

test('successful threaded runtime preserves idempotent pool disposal', async () => {
  const h = harness();
  const runtime = await h.load({cpuThreads: 2});
  assert.equal(h.calls.includes('cleanup'), false);
  await runtime.dispose();
  await runtime.dispose();
  assert.equal(h.calls.filter(x => x === 'cleanup').length, 1);
});

for (const [hardwareConcurrency, expected] of [[undefined,1],[0,1],[-1,1],[NaN,1],[Infinity,1],['8',1],[2.5,1],[1,1],[2,1],[3,1],[4,2],[8,4],[16,4]]) {
  test(`auto CPU budget for reported hardwareConcurrency=${hardwareConcurrency}`, async () => {
    const h = harness({...supported, hardwareConcurrency});
    assert.equal((await h.load()).cpuThreads, expected);
  });
}
test('missing independent owner selects serial without requesting threaded code', async () => {
  const h = harness();
  const result = await h.load({cpuThreads: 4}, {});
  assert.equal(result.cpuThreads, 1);
  assert.match(result.fallbackReason, /independent worker owner/);
  assert.equal(h.calls.some(x => x.includes('threaded')), false);
});
test('unpackaged optional runtime is never requested', async () => {
  const paths = [];
  const load = createRuntimeLoader({config: {threadedAvailable: false},
    capabilities: () => ({...supported, hardwareConcurrency: 16}),
    importModule: async path => { paths.push(path); return {default() {}, loadModel() {}, DictationSession: class {}}; },
  });
  assert.equal((await load({}, {workerOwner: owner})).cpuThreads, 1);
  assert.deepEqual(paths, ['./relaxed/speechworks_web.js']);
});

test('legacy ready-then-configure client transcribes serially without pool control messages', async t => {
  const oldSelf = globalThis.self, oldFetch = globalThis.fetch;
  const events = [], imports = [];
  const bytes = new Uint8Array([1]);
  const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map(x => x.toString(16).padStart(2, '0')).join('');
  const wasm = {default() {}, DictationSession: class {},
    loadModel: async () => ({device: () => 'cpu', fallbackReason: () => '', transcribe: () => 'hello world', resetStream() {}})};
  const loadRuntime = createRuntimeLoader({config: {threadedAvailable: true},
    capabilities: () => ({...supported, hardwareConcurrency: 8}),
    importModule: async path => { imports.push(path); return wasm; }});
  globalThis.__speechworksLegacyTest = {loadRuntime, cpuThreadOption, createWorkerOwner};
  globalThis.self = {location: {href: 'https://example.test/worker.js', origin: 'https://example.test'}, postMessage: message => events.push(message)};
  globalThis.fetch = async url => String(url).endsWith('models.json')
    ? {ok: true, json: async () => ({schema: 1, sampleRate: 16000, defaults: {final: 'tiny'},
      models: {tiny: {url: './tiny.bin', bytes: 1, sha256: digest}}})}
    : {ok: true, arrayBuffer: async () => bytes.buffer};
  t.after(() => { globalThis.self = oldSelf; globalThis.fetch = oldFetch; delete globalThis.__speechworksLegacyTest; });
  const source = (await fs.readFile(new URL('../web/worker.js', import.meta.url), 'utf8'))
    .replace("import {loadRuntime, cpuThreadOption} from './runtime-loader.js';", 'const {loadRuntime, cpuThreadOption, createWorkerOwner} = globalThis.__speechworksLegacyTest;')
    .replace("from './model-cache.js'", `from '${new URL('../web/model-cache.js',import.meta.url).href}'`)
    .replace("import {createWorkerOwner} from './pool-owner.js';", '')
    .replaceAll("new URL('./models.json', import.meta.url)", "new URL('https://example.test/models.json')");
  const {startWorker} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
  startWorker(); await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(events, [{type: 'ready'}]);
  await self.onmessage({data: {type: 'configure', id: 1, options: {cpuThreads: 4}}});
  await self.onmessage({data: {type: 'transcribe', id: 2, samples: new Float32Array(16)}});
  for (let i = 0; i < 20 && !events.some(x => x.id === 2); i++) await new Promise(resolve => setTimeout(resolve, 1));
  assert.equal(events.find(x => x.id === 2)?.text, 'hello world');
  assert.deepEqual(imports, ['./relaxed/speechworks_web.js']);
  assert.equal(events.some(x => x.type.startsWith('speechworks:')), false);
});
