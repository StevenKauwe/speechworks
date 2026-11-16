#!/usr/bin/env node
// Exercise the public demo through its UI, with real models and microphone PCM.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const require = createRequire(import.meta.url);
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = Object.fromEntries(process.argv.slice(2).flatMap((value, i, all) => value.startsWith('--') ? [[value, all[i + 1]?.startsWith('--') === false ? all[i + 1] : true]] : []));
const root = path.resolve(args['--root'] || path.join(repo, '_site'));
const output = path.resolve(args['--output'] || path.join(repo, 'test-results'));
const backend = args['--backend'] || 'cpu';
assert.ok(['cpu', 'gpu'].includes(backend));
const results = {at: new Date().toISOString(), backend, checks: [], errors: [], externalRequests: []};
const types = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.wasm': 'application/wasm', '.json': 'application/json', '.wav': 'audio/wav'};
let server, browser;
await fs.mkdir(output, {recursive: true});
try {
  let url = args['--url'];
  if (!url) {
    // Test a repository subpath, not just /, so GitHub Pages relative URLs matter.
    server = http.createServer(async (req, res) => {
      try {
        const route = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
        if (!route.startsWith('/speechworks/')) throw new Error('Outside the demo');
        const file = path.resolve(root, route.slice('/speechworks/'.length) || 'index.html');
        if (!file.startsWith(root + path.sep)) throw new Error('Outside the demo');
        res.setHeader('Content-Type', types[path.extname(file)] || 'application/octet-stream');
        res.end(await fs.readFile(file));
      } catch { res.writeHead(404); res.end('Not found'); }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    url = `http://127.0.0.1:${server.address().port}/speechworks/`;
  }
  const origin = new URL(url).origin;
  browser = await chromium.launch({
    headless: process.env.HEADLESS !== '0',
    ...(process.env.CHROME_EXECUTABLE ? {executablePath: process.env.CHROME_EXECUTABLE} : {}),
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--disable-features=AudioServiceSandbox,AudioServiceOutOfProcess', `--use-file-for-fake-audio-capture=${path.join(root, 'sample.wav')}`],
  });
  const context = await browser.newContext({viewport: {width: 1440, height: 1120}, permissions: ['microphone', 'clipboard-read', 'clipboard-write']});
  context.on('request', request => {
    if (new URL(request.url()).origin !== origin) results.externalRequests.push(request.url());
    assert.equal(request.method(), 'GET', 'demo must not upload audio or transcripts');
  });
  const page = await context.newPage();
  page.on('pageerror', error => results.errors.push(error.message));
  await page.addInitScript(() => {
    window.__streams = [];
    window.__micCalls = 0;
    window.__violations = [];
    document.addEventListener('securitypolicyviolation', event => window.__violations.push(event.violatedDirective));
    const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async options => {
      window.__micCalls++;
      const stream = await getUserMedia(options);
      window.__streams.push(stream);
      return stream;
    };
    window.__getUserMedia = navigator.mediaDevices.getUserMedia;
  });
  const requested = [];
  page.on('request', request => requested.push(request.url()));
  const check = message => { results.checks.push(message); process.stdout.write(`${message}\n`); };
  const value = () => page.locator('#transcript').inputValue();
  const idle = () => page.waitForFunction(() => document.body.dataset.state === 'idle', null, {timeout: 180000});
  const sample = async () => {
    await page.locator('#sample').click();
    await page.waitForFunction(() => document.getElementById('status').textContent.startsWith('Sample transcribed'), null, {timeout: 180000});
    assert.match((await value()).toLowerCase(), /what you can do for your country/);
  };
  await page.goto(url);
  await page.waitForFunction(() => !document.getElementById('sample').disabled);
  assert.equal(await page.locator('#model').inputValue(), 'streaming-medium');
  assert.equal(requested.filter(url => url.endsWith('.gguf')).length, 0);
  assert.equal(await page.evaluate(() => window.__micCalls), 0);
  check('Initial page makes no model download or microphone request');

  await page.locator('#backend').selectOption(backend);
  await sample();
  results.device = await page.locator('#device').innerText();
  results.timing = await page.locator('#timing').innerText();
  results.sample = await value();
  assert.match(results.device, backend === 'gpu' ? /GPU.*WebGPU/ : /CPU/);
  if (backend === 'gpu') assert.doesNotMatch(results.device, /swiftshader|software|llvmpipe/i);
  assert.equal(await page.evaluate(() => window.__micCalls), 0);
  await page.waitForFunction(() => document.getElementById('cache-note').textContent.includes('cached'));
  await page.locator('#copy').click();
  assert.equal(await page.evaluate(() => navigator.clipboard.readText()), await value());
  await page.screenshot({path: path.join(output, 'demo-desktop.png'), fullPage: true});
  check('Streaming Medium produces real speech text, actual backend/timing, cached weights, and copyable text');

  if (!args['--smoke']) {
    await context.route('**/*.gguf', route => route.abort());
    await page.locator('#previews').uncheck(); // Recreate the engine, retaining only disk cache.
    await sample();
    await context.unroute('**/*.gguf');
    await page.locator('#previews').check();
    check('A fresh engine reuses the complete large model with network downloads blocked');
    await page.locator('#transcript').fill('An edited opening.');
    await page.locator('#record').click();
    await page.waitForFunction(() => document.body.dataset.state === 'recording', null, {timeout: 120000});
    assert.equal(await page.locator('#transcript').getAttribute('readonly'), '');
    assert.equal(await page.locator('#model').isDisabled(), true);
    await page.waitForFunction(() => !document.getElementById('live-container').hidden, null, {timeout: 30000});
    await page.waitForFunction(() => document.getElementById('duration').textContent >= '00:11', null, {timeout: 30000});
    await page.locator('#record').click();
    await idle();
    assert.equal(await page.locator('#error').isHidden(), true);
    const dictated = await value();
    assert.match(dictated, /^An edited opening\./);
    assert.match(dictated.toLowerCase(), /what you can do for your country/);
    assert.equal(dictated.split('An edited opening.').length, 2);
    assert.equal(await page.locator('#transcript').getAttribute('readonly'), null);
    assert.equal(await page.evaluate(() => window.__streams.every(stream => stream.getTracks().every(track => track.readyState === 'ended'))), true);
    check('Real microphone capture shows live previews, preserves edits, flushes final words, and releases the microphone');

    await page.locator('#record').click();
    await page.waitForFunction(() => document.body.dataset.state === 'recording');
    await page.locator('#cancel').click();
    await idle();
    assert.equal(await value(), dictated);
    assert.equal(await page.evaluate(() => window.__streams.every(stream => stream.getTracks().every(track => track.readyState === 'ended'))), true);
    check('Cancel preserves earlier text and releases microphone tracks');

    await page.locator('#clear').click();
    assert.equal(await value(), '');
    await page.locator('#model').selectOption('tiny');
    await page.locator('#previews').uncheck();
    await sample();
    check('Model switching and final-only selection use the same real engine');

    await page.evaluate(async () => {
      await caches.delete('speechworks-models-v1');
      navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Blocked in acceptance test', 'NotAllowedError'); };
    });
    await page.locator('#record').click();
    await page.waitForFunction(() => document.getElementById('error-message').textContent.includes('Microphone access was blocked'));
    await idle();
    await page.evaluate(() => { navigator.mediaDevices.getUserMedia = window.__getUserMedia; });
    check('Microphone permission errors show actionable recovery and leave controls usable');

    // A cancelled model fetch must not leave stale callbacks or strand the next attempt.
    let signalRequest;
    const requestedModel = new Promise(resolve => { signalRequest = resolve; });
    await context.route('**/*.gguf', route => { signalRequest(route); });
    const original = await value();
    await page.locator('#sample').click();
    let timeout;
    const stalled = await Promise.race([requestedModel, new Promise((_, reject) => { timeout = setTimeout(() => reject(new Error('No model fetch to cancel')), 30000); })]).finally(() => clearTimeout(timeout));
    await page.locator('#cancel').click();
    await idle();
    await stalled.abort().catch(() => {});
    await context.unroute('**/*.gguf');
    assert.equal(await value(), original);
    check('Model loading can be cancelled without changing the transcript');

    await context.route('**/*.gguf', route => route.fulfill({status: 503, body: 'Deliberate model download failure'}));
    await page.locator('#sample').click();
    await page.waitForFunction(() => !document.getElementById('error').hidden);
    await idle();
    assert.equal(await page.locator('#sample').isEnabled(), true);
    await context.unroute('**/*.gguf');
    await sample();
    assert.equal(await page.locator('#error').isHidden(), true);
    check('Failed downloads can be retried successfully with no stale error');

    // Remove only the GPU capability. CPU inference and all model loading remain real.
    await context.route('**/runtime/worker-entry.js', async route => {
      const response = await route.fetch();
      await route.fulfill({response, body: `Object.defineProperty(navigator, 'gpu', {value: undefined});\n${await response.text()}`});
    });
    await page.locator('#backend').selectOption('auto');
    await sample();
    assert.match(await page.locator('#device').innerText(), /CPU/);
    assert.equal(await page.locator('#fallback').isVisible(), true);
    await context.unroute('**/runtime/worker-entry.js');
    check('Automatic mode actually falls back to CPU and explains the fallback');
  }

  await page.setViewportSize({width: 390, height: 844});
  await page.screenshot({path: path.join(output, 'demo-mobile.png'), fullPage: true});
  for (const width of [390, 320]) {
    await page.setViewportSize({width, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `no horizontal overflow at ${width}px`);
    assert.equal(await page.locator('#record').isVisible(), true);
  }
  assert.deepEqual(await page.evaluate(() => window.__violations), []);
  assert.deepEqual(results.errors, []);
  assert.deepEqual(results.externalRequests, []);
  check('Mobile layout fits 320–390px; no CSP violations, uncaught errors, or external requests');
} catch (error) {
  results.errors.push(error.stack);
  throw error;
} finally {
  await fs.writeFile(path.join(output, 'demo-browser.json'), JSON.stringify(results, null, 2) + '\n');
  await browser?.close();
  server?.closeAllConnections();
  if (server) await new Promise(resolve => server.close(resolve));
}
