// Moonshine Streaming's bounded-history operations. Buffers are compact f32
// rows; only GEMM inputs are padded to its 32-element K tiles.
struct Params { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32 }
@group(1) @binding(0) var<uniform> p: Params;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> other: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<f32, 128>;
var<workgroup> scores: array<f32, 2048>;

fn reduce_sum(value: f32, lid: u32) -> f32 {
  red[lid] = value;
  workgroupBarrier();
  for (var stride=64u; stride>0u; stride>>=1u) {
    if (lid<stride) { red[lid]+=red[lid+stride]; }
    workgroupBarrier();
  }
  let total=red[0];
  workgroupBarrier();
  return total;
}

// a=frames, b=frame width, c=padded width, d=eps bits, e=exp(log_k) bits.
@compute @workgroup_size(128)
fn cmvn(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  var value=0.0;
  if (lid<p.b) { value=x[wg.x*p.b+lid]; }
  let mean=reduce_sum(value,lid)/f32(p.b);
  var square=0.0;
  if (lid<p.b) { square=(value-mean)*(value-mean); }
  let inv=inverseSqrt(reduce_sum(square,lid)/f32(p.b)+bitcast<f32>(p.d));
  if (lid<p.c) {
    var result=0.0;
    if (lid<p.b) {
      let z=(value-mean)*inv*bitcast<f32>(p.e);
      result=sign(z)*log(abs(z)+sqrt(z*z+1.0));
    }
    out[wg.x*p.c+lid]=result;
  }
}

// a=old elements, b=fresh elements. Concatenate old and fresh into y.
@compute @workgroup_size(256)
fn join(@builtin(global_invocation_id) id: vec3<u32>) {
  let i=id.x;
  if (i>=p.a+p.b) { return; }
  if (i<p.a) { y[i]=x[i]; } else { y[i]=other[i-p.a]; }
}

// a=rows, b=source row width, c=output row width, d=source first row.
// Copies/slices rows and zero-pads each one for GEMM if needed.
@compute @workgroup_size(256)
fn rows(@builtin(global_invocation_id) id: vec3<u32>) {
  let i=id.x;
  if (i>=p.a*p.c) { return; }
  let col=i%p.c;
  var value=0.0;
  if (col<p.b) { value=x[(i/p.c+p.d)*p.b+col]; }
  out[i]=value;
}

// a=output rows, b=input channels, c=input absolute first row,
// d=output absolute first row. Fixed causal k=5, stride=2, left padding=4.
@compute @workgroup_size(256)
fn causal_columns(@builtin(global_invocation_id) id: vec3<u32>) {
  let width=((p.b*5u+31u)/32u)*32u;
  let i=id.x;
  if (i>=p.a*width) { return; }
  let col=i%width;
  var value=0.0;
  let at=2u*(p.d+i/width)+col%5u;
  if (col<p.b*5u && at>=4u) { value=x[(at-4u-p.c)*p.b+col/5u]; }
  out[i]=value;
}

// a=output queries, b=key rows, c=attention width, d=head dim,
// e=head count, f=first query in x, g=left window, h=right window.
@compute @workgroup_size(128)
fn local_attention(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let query=p.f+wg.x;
  let start=u32(max(i32(query)+1-i32(p.g),0));
  let end=min(query+max(p.h,1u),p.b);
  let count=end-start;
  let qbase=query*3u*p.c+wg.y*p.d;
  var mx=-3.402823e38;
  for (var k=lid; k<count; k+=128u) {
    let base=(start+k)*3u*p.c+p.c+wg.y*p.d;
    var dot=0.0;
    for (var c=0u; c<p.d; c++) { dot+=x[qbase+c]*x[base+c]; }
    let value=dot*inverseSqrt(f32(p.d));
    scores[k]=value;
    mx=max(mx,value);
  }
  red[lid]=mx;
  workgroupBarrier();
  for (var stride=64u; stride>0u; stride>>=1u) {
    if (lid<stride) { red[lid]=max(red[lid],red[lid+stride]); }
    workgroupBarrier();
  }
  let maximum=red[0];
  workgroupBarrier();
  var sum=0.0;
  for (var k=lid; k<count; k+=128u) { let value=exp(scores[k]-maximum); scores[k]=value; sum+=value; }
  let total=reduce_sum(sum,lid);
  if (lid<p.d) {
    var value=0.0;
    for (var k=0u; k<count; k++) { value+=scores[k]*x[(start+k)*3u*p.c+2u*p.c+wg.y*p.d+lid]; }
    out[wg.x*p.c+wg.y*p.d+lid]=value/total;
  }
}

// a=rows, b=width, c=absolute audio-frame position.
@compute @workgroup_size(256)
fn positions(@builtin(global_invocation_id) id: vec3<u32>) {
  let i=id.x;
  if (i<p.a*p.b) { y[i]=x[i]+other[p.c*p.b+i]; }
}
