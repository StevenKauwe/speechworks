const $ = id => document.getElementById(id);
const ui = Object.fromEntries(['record', 'record-label', 'sample', 'cancel', 'status', 'loading',
  'loading-label', 'download-progress', 'error', 'error-message', 'retry', 'transcript',
  'live-container', 'live-text', 'word-count', 'clear', 'copy', 'model', 'model-hint',
  'backend', 'previews', 'device', 'model-size', 'fallback', 'cache-note', 'sample-audio',
  'duration', 'level', 'timing'].map(id => [id, $(id)]));
let createDictation, manifest, engine, unsubscribe, active, frame;
let phase = 'idle', initialized = false;
const microphoneSupported = !!navigator.mediaDevices?.getUserMedia && !!globalThis.AudioContext;
const modelName = id => id === 'base' ? 'Moonshine Base' : id === 'tiny' ? 'Moonshine Tiny' : id;
const join = (left, right) => right ? (left ? `${left.trimEnd()} ${right}` : right) : left;
const current = operation => active === operation;

function updateWords() {
  const count = ui.transcript.value.trim().split(/\s+/u).filter(Boolean).length;
  ui['word-count'].textContent = `${count} ${count === 1 ? 'word' : 'words'}`;
  ui.copy.disabled = ui.clear.disabled = !!active || count === 0;
}

function setPhase(value, message) {
  phase = value;
  document.body.dataset.state = value;
  ui.status.textContent = message;
  const busy = value !== 'idle';
  for (const field of [ui.model, ui.backend, ui.previews]) field.disabled = busy || !initialized;
  ui.record.disabled = !initialized || !microphoneSupported || (busy && value !== 'recording');
  ui['record-label'].textContent = value === 'recording' ? 'Stop dictating' : value === 'finishing' ? 'Finishing…' : 'Start dictating';
  ui.sample.hidden = busy;
  ui.sample.disabled = busy || !initialized;
  ui.cancel.hidden = !busy;
  ui.loading.hidden = value !== 'loading';
  ui.transcript.readOnly = busy;
  updateWords();
}

function live(text = '') {
  ui['live-text'].textContent = text;
  ui['live-container'].hidden = !text;
}

function resetMeter() {
  cancelAnimationFrame(frame);
  for (const bar of ui.level.children) bar.style.height = '';
}

function meter(operation) {
  if (!current(operation) || phase !== 'recording') return;
  const seconds = Math.floor((performance.now() - operation.started) / 1000);
  ui.duration.textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  const level = operation.recording.level();
  for (const [i, bar] of [...ui.level.children].entries()) {
    bar.style.height = `${3 + level * (25 - Math.abs(i - 3) * 4)}px`;
  }
  frame = requestAnimationFrame(() => meter(operation));
}

function disposeEngine() {
  unsubscribe?.();
  unsubscribe = null;
  const previous = engine;
  engine = null;
  previous?.destroy();
}

function getEngine() {
  if (engine) return engine;
  const instance = createDictation({
    backend: ui.backend.value,
    models: {draft: ui.model.value, final: ui.model.value},
    session: {previews: ui.previews.checked},
  });
  engine = instance;
  unsubscribe = instance.onStatus(status => {
    if (instance !== engine) return;
    if (status.device) ui.device.textContent = status.device;
    ui.fallback.hidden = !status.fallback;
    ui.fallback.textContent = status.fallback ? `Fallback: ${status.fallback}` : '';
    if (phase === 'loading') {
      ui['download-progress'].value = status.progress || 0;
      ui['loading-label'].textContent = status.progress >= 1
        ? 'Model received. Preparing the speech engine…'
        : `Loading ${modelName(ui.model.value)} · ${Math.round((status.progress || 0) * 100)}%`;
    }
  });
  return instance;
}

async function refreshCacheNote(instance) {
  const cached = await instance.isCached();
  if (engine !== instance) return;
  ui['cache-note'].textContent = cached
    ? 'This model is cached in your browser. Future loads can reuse it.'
    : 'The selected model downloads on first use. Your browser can cache it for next time.';
}

function clearError() {
  ui.error.hidden = true;
  ui['error-message'].textContent = '';
  ui.retry.hidden = true;
}

function errorMessage(error) {
  if (error.name === 'NotAllowedError') return 'Microphone access was blocked. Allow this site to use your microphone in your browser settings, then try again. You can also try the sample without a microphone.';
  if (error.name === 'NotFoundError') return 'No microphone was found. Connect one and try again, or try the sample.';
  if (error.name === 'NotReadableError') return 'Your microphone could not be opened. Check that it is connected and available, then try again.';
  return `${error.message || 'Speech recognition could not start.'} Try again, or select CPU if WebGPU is unavailable.`;
}

function finish(operation, message) {
  if (!current(operation)) return;
  active = null;
  live();
  resetMeter();
  setPhase('idle', message);
}

function fail(operation, error) {
  if (!current(operation)) return;
  // Finalized phrases remain available; tentative words are never committed on error.
  finish(operation, 'Something interrupted this attempt. You can try again.');
  operation.abort.abort();
  disposeEngine();
  ui.error.hidden = false;
  ui['error-message'].textContent = errorMessage(error);
}

function cancel() {
  const operation = active;
  if (!operation) return;
  // Invalidate callbacks before aborting: an old worker must not change a new run.
  active = null;
  operation.abort.abort();
  disposeEngine();
  ui.transcript.value = operation.original;
  live();
  resetMeter();
  ui.duration.textContent = '00:00';
  setPhase('idle', 'Cancelled. Your previous text is unchanged.');
}

async function stop() {
  const operation = active;
  if (!operation?.recording || phase !== 'recording') return;
  setPhase('finishing', 'Finishing your last phrase…');
  resetMeter();
  try {
    const text = await operation.recording.stop();
    if (!current(operation)) return;
    ui.transcript.value = join(operation.original, text);
    finish(operation, text ? 'Done. Your words are ready to edit.' : 'No speech detected. Try speaking a little closer to your microphone.');
  } catch (error) { fail(operation, error); }
}

// The bundled sample is PCM16 mono at 16 kHz. Reading it directly avoids an
// AudioContext or microphone permission for the sample-only experience.
function readSample(bytes) {
  const view = new DataView(bytes);
  const four = offset => String.fromCharCode(...new Uint8Array(bytes, offset, 4));
  if (bytes.byteLength < 12 || four(0) !== 'RIFF' || four(8) !== 'WAVE') throw new Error('The sample audio is not a WAV file.');
  let format, data;
  for (let at = 12; at + 8 <= bytes.byteLength;) {
    const size = view.getUint32(at + 4, true);
    if (at + 8 + size > bytes.byteLength) throw new Error('The sample audio is incomplete.');
    const type = four(at);
    if (type === 'fmt ' && size >= 16) format = [view.getUint16(at + 8, true), view.getUint16(at + 10, true), view.getUint32(at + 12, true), view.getUint16(at + 22, true)];
    if (type === 'data') data = [at + 8, size];
    at += 8 + size + (size & 1);
  }
  if (format?.join(',') !== '1,1,16000,16' || !data || data[1] % 2) throw new Error('The sample must be 16 kHz mono PCM16 audio.');
  const pcm = new Float32Array(data[1] / 2);
  for (let i = 0; i < pcm.length; i++) pcm[i] = view.getInt16(data[0] + i * 2, true) / 32768;
  return pcm;
}

async function start(kind) {
  if (!initialized || active) return;
  clearError();
  ui['sample-audio'].pause();
  ui.timing.textContent = '';
  ui.duration.textContent = '00:00';
  live();
  const operation = {abort: new AbortController(), original: ui.transcript.value};
  active = operation;
  setPhase('loading', 'Preparing your on-device speech engine…');
  ui['download-progress'].value = 0;
  ui['loading-label'].textContent = 'Preparing model…';
  try {
    const instance = getEngine();
    await instance.preload();
    if (!current(operation)) return;
    void refreshCacheNote(instance);
    if (kind === 'sample') {
      setPhase('transcribing', 'Transcribing the sample on your device…');
      const response = await fetch(new URL('./sample.wav', import.meta.url), {signal: operation.abort.signal});
      if (!response.ok) throw new Error('The sample audio could not be loaded.');
      const samples = readSample(await response.arrayBuffer());
      if (!current(operation)) return;
      const started = performance.now();
      const text = await instance.transcribe(samples, {model: ui.model.value});
      const elapsed = performance.now() - started;
      if (!current(operation)) return;
      ui.transcript.value = join(operation.original, text);
      ui.timing.textContent = `${(samples.length / 16000).toFixed(1)} s audio · ${(elapsed / 1000).toFixed(2)} s transcription (excludes model loading)`;
      finish(operation, 'Sample transcribed. Try your own voice next.');
    } else {
      setPhase('permission', 'Allow microphone access when your browser asks.');
      const recording = await instance.record({
        signal: operation.abort.signal,
        onText: ({committed, live: preview}) => {
          if (!current(operation)) return;
          ui.transcript.value = join(operation.original, committed);
          live(preview);
          updateWords();
        },
        onError: error => fail(operation, error),
      });
      if (!current(operation)) { recording.cancel(); return; }
      operation.recording = recording;
      operation.started = performance.now();
      setPhase('recording', ui.previews.checked ? 'Listening. Speak naturally; pauses finish a phrase.' : 'Listening. Phrases appear when you pause or stop.');
      meter(operation);
    }
  } catch (error) { fail(operation, error); }
}

function configure() {
  if (active || !initialized) return;
  disposeEngine();
  clearError();
  ui.device.textContent = 'Detected on first use';
  ui.fallback.hidden = true;
  ui.timing.textContent = '';
  ui['model-size'].textContent = `${(manifest.models[ui.model.value].bytes / 1e6).toFixed(1)} MB`;
  ui['model-hint'].textContent = ui.model.value === 'tiny' ? 'English · smaller download, lighter inference.' : 'English · our default for dictation.';
  setPhase('idle', microphoneSupported ? 'Ready when you are. Choose your microphone or the sample.' : 'Microphone capture is unavailable in this browser. Try the sample instead.');
  void refreshCacheNote(getEngine());
}

ui.record.addEventListener('click', () => phase === 'recording' ? stop() : start('microphone'));
ui.sample.addEventListener('click', () => start('sample'));
ui.cancel.addEventListener('click', cancel);
ui.clear.addEventListener('click', () => { ui.transcript.value = ''; updateWords(); ui.transcript.focus(); });
ui.transcript.addEventListener('input', updateWords);
for (const control of [ui.model, ui.backend, ui.previews]) control.addEventListener('change', configure);
ui.copy.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(ui.transcript.value);
    ui.status.textContent = 'Transcript copied to your clipboard.';
  } catch {
    ui.transcript.focus(); ui.transcript.select();
    ui.status.textContent = 'Clipboard access is unavailable. Your text is selected; use your browser’s Copy command.';
  }
});
document.addEventListener('keydown', event => { if (event.key === 'Escape' && active) cancel(); });
window.addEventListener('pagehide', () => { cancel(); disposeEngine(); });
ui.retry.addEventListener('click', () => location.reload());

try {
  if (!globalThis.isSecureContext || !globalThis.WebAssembly || !globalThis.Worker) throw new Error('This demo needs a secure browser with WebAssembly and web workers. Open it over HTTPS, or localhost for development.');
  const [module, response] = await Promise.all([
    import('./runtime/speechworks.js'), fetch(new URL('./runtime/models.json', import.meta.url)),
  ]);
  if (!response.ok) throw new Error('The model catalog could not be loaded.');
  manifest = await response.json();
  if (manifest.schema !== 1 || !manifest.models?.[manifest.defaults?.final]) throw new Error('The model catalog is invalid.');
  createDictation = module.createDictation;
  ui.model.replaceChildren(...Object.keys(manifest.models).map(id => new Option(modelName(id), id)));
  ui.model.value = manifest.defaults.final;
  initialized = true;
  configure();
} catch (error) {
  setPhase('idle', 'The playground could not start.');
  ui.error.hidden = false;
  ui['error-message'].textContent = error.message;
  ui.retry.hidden = false;
}
