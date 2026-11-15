# Worker protocol

Use a module worker importing `startWorker` from the installed `worker.js`.
`startWorker({manifestUrl, backend, models, session})` installs the handler and
initializes WASM. A `ready` event means the worker can receive requests; weights
load lazily. All sample arrays are `Float32Array`. Transfer their buffers when
the caller no longer needs them. Models and manifest must be served same-origin.

| Request | Response |
|---|---|
| `{type:'configure', id, options}` before loading models | `{type:'configured', id}` |
| `{type:'preload', id}` | `{type:'loaded', id}` after selected models load |
| `{type:'transcribe', id, samples, model?, priority?}`; 16 kHz PCM | `{type:'result', id, text}` |
| `{type:'start', id, session, config?}` | `{type:'started', id, session}` |
| `{type:'audio', session, samples, sampleRate}` | `{type:'audio-ack', session, samples:count}` |
| `{type:'stop', session}` | Final events, then `{type:'done', session}` |
| `{type:'cancel', session}` | Cancels and discards pending results |

During dictation, `{type:'live'|'final', session, phrase, text}` replaces that
phrase's text. `live` is tentative; `final` is committed. Results from an old
session must be discarded. Final phrase numbers start at zero and are contiguous.
`progress` reports model download progress; `backend` reports selected device
and any automatic fallback reason. An `error` has `message` plus the relevant
request `id` or recording `session`. An error without either is a fatal worker
initialization failure. Empty successful text and failure are distinct.

Inference runs serially. Final phrases take priority over previews; at most one
current preview is decoded. Clip queue capacity is eight requests. Live capture
must enforce a bounded unacknowledged audio queue; the supplied client caps it
at five seconds. Clip length is at most ten minutes. These are explicit errors,
never silent truncation.
