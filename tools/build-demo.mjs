#!/usr/bin/env node
// Bundle the demo and the exact same installed browser engine applications use.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2), options = {};
for (let i = 0; i < args.length; i += 2) {
  if (!['--dest', '--model-dir'].includes(args[i]) || !args[i + 1] || args[i + 1].startsWith('--')) {
    throw new Error('Usage: node tools/build-demo.mjs [--dest directory] [--model-dir cache-directory]');
  }
  options[args[i]] = args[i + 1];
}
const dest = path.resolve(options['--dest'] || path.join(root, '_site'));
const cache = path.resolve(options['--model-dir'] || path.join(root, 'models'));
if (root === dest || root.startsWith(dest + path.sep) || dest === path.join(root, 'demo') || dest.startsWith(path.join(root, 'demo') + path.sep)) {
  throw new Error('Choose an output directory outside the demo source, such as _site.');
}
await fs.mkdir(cache, {recursive: true});
const fixture = path.join(cache, 'jfk.wav');
const sampleUrl = 'https://raw.githubusercontent.com/ggml-org/whisper.cpp/b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav';
const sampleHash = '59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e';
let bytes;
try { bytes = await fs.readFile(fixture); }
catch (error) {
  if (error.code !== 'ENOENT') throw error;
  const response = await fetch(sampleUrl);
  if (!response.ok) throw new Error(`Sample download failed: ${response.status}`);
  bytes = Buffer.from(await response.arrayBuffer());
}
if (createHash('sha256').update(bytes).digest('hex') !== sampleHash) throw new Error('The speech sample does not match its pinned checksum.');
await fs.writeFile(fixture, bytes);
const build = spawnSync(path.join(root, 'tools/install-web.sh'), ['--dest', path.join(dest, 'runtime'), '--model-dir', cache], {
  cwd: root, stdio: 'inherit', env: {...process.env, SPEECHWORKS_WASM_THREADS: '0'},
});
if (build.error) throw build.error;
if (build.status !== 0) throw new Error(`Runtime build failed (${build.status ?? build.signal}).`);
for (const file of ['index.html', 'demo.css', 'demo.js', 'mark.svg']) await fs.copyFile(path.join(root, 'demo', file), path.join(dest, file));
await fs.copyFile(fixture, path.join(dest, 'sample.wav'));
await fs.writeFile(path.join(dest, '.nojekyll'), '');
process.stdout.write(`Demo ready: ${dest}\nServe this directory over HTTP on localhost, or deploy it over HTTPS.\n`);
