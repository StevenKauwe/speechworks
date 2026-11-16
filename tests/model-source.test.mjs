import test from 'node:test';
import assert from 'node:assert/strict';
import {fetchModel} from '../web/model-source.js';

const url = new URL('https://example.test/runtime/models/hash.gguf');
const spec = {bytes: 5, parts: [{url: 'hash.gguf.part-000', bytes: 3}, {url: 'hash.gguf.part-001', bytes: 2}]};
function network(t, fn) {
  const previous = globalThis.fetch;
  globalThis.fetch = fn;
  t.after(() => { globalThis.fetch = previous; });
}

test('single-file models retain the existing request and redirect policy', async t => {
  const response = new Response('model');
  network(t, async (source, options) => {
    assert.equal(source, url);
    assert.equal(options.redirect, 'error');
    return response;
  });
  assert.equal(await fetchModel(url, {bytes: 5}), response);
});

test('parts stream in manifest order relative to the canonical model URL', async t => {
  const requests = [];
  network(t, async (source, options) => {
    requests.push(String(source));
    assert.equal(options.redirect, 'error');
    return new Response(requests.length === 1 ? 'abc' : 'de');
  });
  assert.equal(await (await fetchModel(url, spec)).text(), 'abcde');
  assert.deepEqual(requests, spec.parts.map(p => new URL(p.url, url).href));
});

test('invalid or cross-origin part manifests fail before a network request', async t => {
  network(t, async () => { assert.fail('invalid manifest made a request'); });
  for (const parts of [null, [], [null], Array(513).fill(spec.parts[0]),
    [{url: '', bytes: 5}], [{url: 'a', bytes: 0}], [{url: 'a', bytes: 1.5}],
    [{url: 'a', bytes: 4}], [{url: 'a', bytes: 6}],
    [{url: 'https://other.test/a', bytes: 5}]]) {
    await assert.rejects(fetchModel(url, {...spec, parts}));
  }
});

for (const [name, response, expected] of [
  ['missing part', () => new Response('', {status: 404}), /HTTP 404/],
  ['short part', () => new Response('ab'), /incomplete/],
  ['oversized part', () => new Response('abcd'), /exceeds/],
  ['empty body', () => new Response(null), /empty response/],
  ['network failure', () => { throw new Error('offline'); }, /offline/],
]) {
  test(`${name} aborts the model without fetching later parts`, async t => {
    let calls = 0;
    network(t, async () => { calls++; return response(); });
    await assert.rejects((await fetchModel(url, spec)).arrayBuffer(), expected);
    assert.equal(calls, 1);
  });
}

test('cancelling a model cancels the active part and prevents later requests', async t => {
  let signal, cancelled = false, requests = 0;
  network(t, async (_, options) => {
    requests++; signal = options.signal;
    return new Response(new ReadableStream({
      start(controller) { controller.enqueue(new Uint8Array([1])); },
      cancel() { cancelled = true; },
    }));
  });
  const reader = (await fetchModel(url, spec)).body.getReader();
  await reader.read();
  await reader.cancel();
  assert.equal(signal.aborted, true);
  assert.equal(cancelled, true);
  assert.equal(requests, 1);
});
