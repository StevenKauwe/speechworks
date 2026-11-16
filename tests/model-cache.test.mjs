import {test} from 'node:test';
import assert from 'node:assert/strict';
import {cacheModel,cachedModel,hasCachedModel,deleteCachedModel} from '../web/model-cache.js';
const url='https://example.test/model.gguf', hash='a'.repeat(64);
function memoryCache(failAt=Infinity) {
  const data=new Map(); let puts=0;
  return {data,
    async put(key,response) {if(++puts===failAt) throw new Error('quota'); data.set(key,{bytes:new Uint8Array(await response.arrayBuffer()),headers:response.headers});},
    async match(key) {const item=data.get(key); return item && new Response(item.bytes,{headers:item.headers});},
    async delete(key) {return data.delete(key);},
  };
}
test('small and previously cached models remain reusable',async()=>{
  const cache=memoryCache(),bytes=new Uint8Array([1,2,3]),spec={bytes:3,sha256:hash};
  await cacheModel(cache,url,spec,bytes);
  assert.equal(await hasCachedModel(cache,url,spec),true);
  assert.deepEqual(new Uint8Array(await(await cachedModel(cache,url,spec)).arrayBuffer()),bytes);
  await deleteCachedModel(cache,url,spec); assert.equal(cache.data.size,0);
});
test('large model survives chunked storage; missing chunks are never marked cached',async()=>{
  const cache=memoryCache(),bytes=new Uint8Array(64*1024*1024+7),spec={bytes:bytes.length,sha256:hash};
  bytes[0]=7; bytes[bytes.length-1]=9;
  await cacheModel(cache,url,spec,bytes);
  assert.equal(cache.data.size,3);
  assert.equal(await hasCachedModel(cache,url,spec),true);
  const restored=new Uint8Array(await(await cachedModel(cache,url,spec)).arrayBuffer());
  assert.deepEqual(restored,bytes);
  const part=[...cache.data.keys()].find(key=>key!==url);
  cache.data.delete(part);
  assert.equal(await hasCachedModel(cache,url,spec),false);
  assert.equal(await cachedModel(cache,url,spec),null);
  await deleteCachedModel(cache,url,spec); assert.equal(cache.data.size,0);
});
test('failed large-model write removes partial entries and completion marker',async()=>{
  const cache=memoryCache(2),bytes=new Uint8Array(64*1024*1024+1),spec={bytes:bytes.length,sha256:hash};
  await assert.rejects(cacheModel(cache,url,spec,bytes),/quota/);
  assert.equal(await hasCachedModel(cache,url,spec),false); assert.equal(cache.data.size,0);
});
