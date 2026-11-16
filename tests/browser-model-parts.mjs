#!/usr/bin/env node
// Exercise split Medium weights through the public API, including fresh-engine
// cache reuse with model requests blocked and same-size corruption rejection.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const args = Object.fromEntries(process.argv.slice(2).flatMap((v, i, a) => v.startsWith('--') ? [[v, a[i + 1]]] : []));
const root = path.resolve(args['--root']);
const fixture = await fs.readFile(args['--fixture']);
assert.equal(createHash('sha256').update(fixture).digest('hex'), '59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e');
const manifest = JSON.parse(await fs.readFile(path.join(root, 'models.json')));
const id = manifest.defaults.final, spec = manifest.models[id];
assert.equal(id, 'streaming-medium');
assert.ok(spec.parts?.length > 1);
await assert.rejects(fs.access(path.join(root, spec.url)), {code: 'ENOENT'});
const canonical = new URL(spec.url, 'http://fixture.test/models.json');
const routes = spec.parts.map(part => new URL(part.url, canonical).pathname);
for (const [index, route] of routes.entries()) {
  const stat = await fs.stat(path.join(root, route));
  assert.equal(stat.size, spec.parts[index].bytes);
  assert.ok(stat.size <= 64 * 1024 * 1024);
}
let mode = 'good';
const downloads = [], external = [], pageErrors = [];
const server = http.createServer(async (req, res) => {
  res.setHeader('Content-Security-Policy', "script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; worker-src 'self'");
  res.setHeader('Cache-Control', 'no-store');
  try {
    const route = new URL(req.url, 'http://localhost').pathname;
    if (route === '/') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>Split model acceptance</title>'); return; }
    if (route === '/fixture.wav') { res.end(fixture); return; }
    if (routes.includes(route)) {
      downloads.push({mode, route});
      if (mode === 'blocked') { res.writeHead(503); res.end(); return; }
    }
    const target = path.resolve(root, '.' + decodeURIComponent(route));
    if (!target.startsWith(root + path.sep)) { res.writeHead(403); res.end(); return; }
    const bytes = await fs.readFile(target);
    if (mode === 'corrupt' && route === routes[0]) bytes[0] ^= 1;
    res.setHeader('Content-Type', {'.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json'}[path.extname(target)] || 'application/octet-stream');
    res.end(bytes);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser;
const report = {model: id, bytes: spec.bytes, parts: spec.parts.length};
try {
  browser = await chromium.launch({headless: true, ...(process.env.CHROME_EXECUTABLE ? {executablePath: process.env.CHROME_EXECUTABLE} : {})});
  const context = await browser.newContext();
  context.on('request', r => { if (!r.url().startsWith(origin + '/')) external.push(r.url()); });
  const page = await context.newPage();
  page.on('pageerror', e => pageErrors.push(e.message));
  await page.goto(origin);
  await page.evaluate(async () => {
    const {createDictation} = await import('/speechworks.js');
    window.makeEngine = () => createDictation({backend: 'cpu', cpuThreads: 1, session: {previews: false}});
    const bytes = await (await fetch('/fixture.wav')).arrayBuffer(), view = new DataView(bytes);
    for (let at = 12; at + 8 <= bytes.byteLength;) {
      const size = view.getUint32(at + 4, true);
      if (String.fromCharCode(...new Uint8Array(bytes, at, 4)) === 'data') {
        window.pcm = new Float32Array(size / 2);
        for (let i = 0; i < pcm.length; i++) pcm[i] = view.getInt16(at + 8 + 2 * i, true) / 32768;
      }
      at += 8 + size + (size & 1);
    }
    if (!window.pcm) throw new Error('Missing speech fixture PCM');
  });
  async function recognize() {
    return page.evaluate(async () => {
      const engine = makeEngine();
      try { return {text: await engine.transcribe(pcm), cached: await engine.isCached()}; }
      finally { engine.destroy(); }
    });
  }
  report.initial = await recognize();
  assert.match(report.initial.text, /my fellow Americans/);
  assert.match(report.initial.text, /what you can do for your country/);
  assert.equal(report.initial.cached, true);
  assert.deepEqual(downloads.map(d => d.route), routes);
  mode = 'blocked';
  report.cached = await recognize();
  assert.deepEqual(report.cached, report.initial);
  assert.equal(downloads.some(d => d.mode === 'blocked'), false);
  await page.evaluate(() => caches.delete('speechworks-models-v1'));
  mode = 'corrupt';
  report.corrupt = await page.evaluate(async () => {
    const engine = makeEngine();
    try { await engine.preload(); return {accepted: true}; }
    catch (error) { return {accepted: false, error: error.message, cached: await engine.isCached()}; }
    finally { engine.destroy(); }
  });
  assert.equal(report.corrupt.accepted, false);
  assert.match(report.corrupt.error, /checksum/);
  assert.equal(report.corrupt.cached, false);
  assert.deepEqual(external, []);
  assert.deepEqual(pageErrors, []);
  report.passed = true;
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
  const output = path.resolve(args['--output'] || 'test-results/model-parts.json');
  await fs.mkdir(path.dirname(output), {recursive: true});
  await fs.writeFile(output, JSON.stringify({...report, downloads, external, pageErrors}, null, 2) + '\n');
}
