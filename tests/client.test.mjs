import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createDictation} from '../web/speechworks.js';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};
const tick = () => new Promise(resolve => setImmediate(resolve));
async function until(predicate) {
  for (let i = 0; i < 100; i++) { if (predicate()) return; await tick(); }
  assert.fail('Expected operation did not occur.');
}
async function bounded(promise) {
  let timeout;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timeout = setTimeout(() => reject(new Error('Operation failed to settle.')), 500);
    })]);
  } finally { clearTimeout(timeout); }
}

function harness(t) {
  const h = {workers: [], contexts: [], tracks: [], taps: [], messages: [],
    hold: new Set(), throw: new Set(), flush: true, captureCalls: 0};
  const globals = new Map();
  function replace(key, value) {
    globals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, {configurable: true, writable: true, value});
  }
  replace('Worker', class {
    constructor() { h.workers.push(this); }
    postMessage(message) {
      h.messages.push(message);
      if (h.throw.has(message.type)) throw new DOMException('Cannot clone message.', 'DataCloneError');
      if (h.hold.has(message.type)) return;
      queueMicrotask(() => {
        const {type, id, session} = message;
        if (type === 'configure') this.emit({type: 'configured', id});
        if (type === 'preload') this.emit({type: 'loaded', id});
        if (type === 'start') this.emit({type: 'started', id, session});
        if (type === 'transcribe') this.emit({type: 'result', id, text: 'hello'});
        if (type === 'stop') {
          this.emit({type: 'final', session, phrase: 0, text: 'hello'});
          this.emit({type: 'done', session});
        }
      });
    }
    emit(data) { if (!this.terminated) this.onmessage?.({data}); }
    terminate() { this.terminated = true; }
  });
  replace('AudioContext', class {
    constructor(options = {}) {
      this.sampleRate = options.sampleRate || 48000;
      this.destination = {};
      this.audioWorklet = {addModule: async () => {}};
      h.contexts.push(this);
    }
    createMediaStreamSource() { return {connect() {}}; }
    createAnalyser() { return {getFloatTimeDomainData(values) { values.fill(0); }}; }
    async resume() {}
    async close() { this.closed = true; }
  });
  replace('AudioWorkletNode', class {
    constructor() {
      h.taps.push(this);
      this.port = {postMessage: message => {
        if (message !== 'flush') return;
        h.flushing = true;
        if (h.flush) queueMicrotask(() => this.port.onmessage?.({data: {flushed: true}}));
      }};
    }
    connect() {}
  });
  h.stream = () => {
    const track = {stop() { this.stopped = true; }};
    h.tracks.push(track);
    return {getTracks: () => [track]};
  };
  replace('navigator', {mediaDevices: {getUserMedia() {
    h.captureCalls++;
    return h.capture?.promise || Promise.resolve(h.stream());
  }}});
  h.engine = createDictation({stopTailMs: 0});
  t.after(() => {
    h.engine.destroy();
    for (const [key, descriptor] of globals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  return h;
}

test('aborting during preload settles record without opening a microphone', async t => {
  const h = harness(t), controller = new AbortController();
  h.hold.add('preload');
  const recording = h.engine.record({signal: controller.signal});
  const rejected = assert.rejects(bounded(recording), {name: 'AbortError'});
  await until(() => h.messages.some(m => m.type === 'preload'));
  controller.abort();
  await rejected;
  assert.equal(h.captureCalls, 0);
});

test('permission granted after cancellation immediately releases the late microphone', async t => {
  const h = harness(t), controller = new AbortController();
  h.capture = deferred();
  const recording = h.engine.record({signal: controller.signal});
  const rejected = assert.rejects(bounded(recording), {name: 'AbortError'});
  await until(() => h.captureCalls === 1);
  controller.abort();
  await rejected;
  h.capture.resolve(h.stream());
  await tick();
  assert.ok(h.tracks.every(track => track.stopped));
  assert.equal(h.contexts.length, 0);
});

test('aborting while start waits releases capture and allows the next recording', async t => {
  const h = harness(t), controller = new AbortController();
  h.hold.add('start');
  const recording = h.engine.record({signal: controller.signal});
  const rejected = assert.rejects(bounded(recording), {name: 'AbortError'});
  await until(() => h.messages.some(m => m.type === 'start'));
  controller.abort();
  await rejected;
  assert.ok(h.tracks.every(track => track.stopped));
  assert.ok(h.contexts.every(context => context.closed));
  assert.ok(h.messages.some(m => m.type === 'cancel'));
  h.hold.delete('start');
  assert.equal(await bounded((await h.engine.record()).stop()), 'hello');
});

test('aborting a worklet flush rejects stop immediately and does not finalize audio', async t => {
  const h = harness(t), controller = new AbortController();
  h.flush = false;
  const recording = await h.engine.record({signal: controller.signal});
  const rejected = assert.rejects(bounded(recording.stop()), {name: 'AbortError'});
  await until(() => h.flushing);
  controller.abort();
  await rejected;
  assert.ok(h.tracks.every(track => track.stopped));
  assert.ok(!h.messages.some(m => m.type === 'stop'));
});

test('the abort signal still cancels final inference after capture has closed', async t => {
  const h = harness(t), controller = new AbortController();
  h.hold.add('stop');
  const recording = await h.engine.record({signal: controller.signal});
  const rejected = assert.rejects(bounded(recording.stop()), {name: 'AbortError'});
  await until(() => h.messages.some(m => m.type === 'stop'));
  assert.ok(h.tracks.every(track => track.stopped));
  controller.abort();
  await rejected;
});

test('a completion callback exception rejects stop and cannot strand capture', async t => {
  const h = harness(t);
  let updates = 0, errors = 0;
  const recording = await h.engine.record({
    onText() { if (++updates === 2) throw new Error('UI failed.'); },
    onError() { errors++; throw new Error('Error UI also failed.'); },
  });
  await assert.rejects(bounded(recording.stop()), /UI failed/);
  assert.equal(errors, 1);
  assert.ok(h.tracks.every(track => track.stopped));
  assert.equal(await bounded((await h.engine.record()).stop()), 'hello');
});

test('a synchronous postMessage failure rejects the request and permits retry', async t => {
  const h = harness(t);
  h.throw.add('transcribe');
  await assert.rejects(bounded(h.engine.transcribe(new Float32Array(16))), {name: 'DataCloneError'});
  h.throw.delete('transcribe');
  assert.equal(await bounded(h.engine.transcribe(new Float32Array(16))), 'hello');
});

test('audio delivery failure releases the microphone and rejects stop', async t => {
  const h = harness(t);
  const recording = await h.engine.record();
  h.throw.add('audio');
  h.taps[0].port.onmessage({data: {samples: new Float32Array(512)}});
  await assert.rejects(bounded(recording.stop()), {name: 'DataCloneError'});
  assert.ok(h.tracks.every(track => track.stopped));
  assert.ok(h.contexts.every(context => context.closed));
});

test('worker errors release capture, settle stop, and ignore stale connection errors', async t => {
  const h = harness(t);
  h.engine.onStatus(() => { throw new Error('Status UI failed.'); });
  const recording = await h.engine.record();
  const previous = h.workers[0];
  previous.onerror({preventDefault() {}, message: 'GPU worker crashed.'});
  await assert.rejects(bounded(recording.stop()), /GPU worker crashed/);
  assert.ok(previous.terminated);
  assert.ok(h.tracks.every(track => track.stopped));
  const next = await h.engine.record();
  previous.onerror({preventDefault() {}, message: 'Old worker failed again.'});
  assert.equal(await bounded(next.stop()), 'hello');
  assert.equal(h.workers.length, 2);
  assert.ok(!h.workers[1].terminated);
});

test('unreadable worker messages reject pending transcription and permit a new worker', async t => {
  const h = harness(t);
  h.hold.add('transcribe');
  const rejected = assert.rejects(bounded(h.engine.transcribe(new Float32Array(16))), /unreadable message/);
  await until(() => h.messages.some(m => m.type === 'transcribe'));
  h.workers[0].onmessageerror();
  await rejected;
  h.hold.delete('transcribe');
  assert.equal(await bounded(h.engine.transcribe(new Float32Array(16))), 'hello');
  assert.equal(h.workers.length, 2);
});

test('destroy during startup promptly rejects record and releases the worker', async t => {
  const h = harness(t);
  h.hold.add('configure');
  const recording = h.engine.record();
  const rejected = assert.rejects(bounded(recording), {name: 'AbortError'});
  await until(() => h.messages.some(m => m.type === 'configure'));
  h.engine.destroy();
  await rejected;
  assert.ok(h.workers[0].terminated);
  assert.equal(h.captureCalls, 0);
});
