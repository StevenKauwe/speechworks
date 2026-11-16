#!/usr/bin/env node
// Run after install-web.sh. Download only pinned, licensed acceptance audio;
// expose the verified model cache under the names used by native tests.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const directory=path.resolve(process.argv[2] || 'models');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const manifest=JSON.parse(await fs.readFile(path.join(root,'models.lock.json'),'utf8'));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
await fs.mkdir(directory,{recursive:true});
for(const id of ['streaming-medium','streaming-small']) {
  const model=manifest.models[id];
  const source=path.join(directory,model.sha256+'.gguf');
  const name=path.basename(new URL(model.source).pathname);
  if(sha(await fs.readFile(source))!==model.sha256) throw new Error(`${id}: model cache checksum mismatch; run install-web.sh`);
  await fs.copyFile(source,path.join(directory,name));
}
const fixtures=[
  ['jfk.wav','https://raw.githubusercontent.com/ggml-org/whisper.cpp/b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav','59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'],
  // Google FLEURS en_us test id 1904, CC-BY-4.0; attribution in docs/moonshine-streaming.md.
  ['fleurs-en.wav','https://raw.githubusercontent.com/handy-computer/transcribe.cpp/adfc50210a3f026a655a3e19937d90ef2bf0728c/samples/fleurs-en.wav','7835bd6ffb54ce38a2a9bcde3905ba424faed94d50a474f21a9cbe9209b869df'],
];
for(const [name,url,hash] of fixtures) {
  const dest=path.join(directory,name);
  let bytes;
  try {bytes=await fs.readFile(dest);} catch(e) {
    if(e.code!=='ENOENT') throw e;
    const response=await fetch(url);
    if(!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    bytes=Buffer.from(await response.arrayBuffer());
  }
  if(sha(bytes)!==hash) throw new Error(`${name}: fixture checksum mismatch`);
  await fs.writeFile(dest,bytes);
}
console.log(`Verified streaming fixtures: ${directory}`);
