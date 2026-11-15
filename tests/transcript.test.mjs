import test from 'node:test';
import assert from 'node:assert/strict';
import {Transcript} from '../web/transcript.js';

test('previews replace rather than duplicate text, and final results resist stale previews',()=>{
  const t=new Transcript();
  assert.deepEqual(t.update({type:'live',phrase:0,text:'hello'}),{committed:'',live:'hello'});
  assert.deepEqual(t.update({type:'live',phrase:0,text:'hello world'}),{committed:'',live:'hello world'});
  assert.deepEqual(t.update({type:'final',phrase:0,text:'Hello, world.'}),{committed:'Hello, world.',live:''});
  assert.deepEqual(t.update({type:'live',phrase:0,text:'stale'}),{committed:'Hello, world.',live:''});
  assert.equal(t.finish(),'Hello, world.');
});
test('out-of-order final results stay tentative until missing phrases arrive',()=>{
  const t=new Transcript();
  assert.deepEqual(t.update({type:'final',phrase:1,text:'Second.'}),{committed:'',live:'Second.'});
  assert.throws(()=>t.finish(),/missing/);
  assert.deepEqual(t.update({type:'final',phrase:0,text:'First.'}),{committed:'First. Second.',live:''});
  assert.equal(t.finish(),'First. Second.');
});
test('successful silence is distinct from an unfinished phrase',()=>{
  const t=new Transcript();t.update({type:'final',phrase:0,text:''});assert.equal(t.finish(),'');
  t.update({type:'live',phrase:1,text:'unfinished'});assert.throws(()=>t.finish(),/unfinished/);
});
