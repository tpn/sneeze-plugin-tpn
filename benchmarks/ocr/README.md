# OCR provider benchmark

This benchmark compares local and API-backed document-to-Markdown providers
against the three cuPhoton pages that exposed severe DeepSeek-OCR regressions.
It is deliberately small and adversarial. Passing it qualifies a provider for
a larger stratified benchmark; it does not qualify a provider for production.

The checked-in manifest, `cuphoton-regressions.jsonl`, contains corpus-relative
stems, immutable input and reference hashes, image dimensions, and calibrated
assertions. Stems are relative to the canonical `papers` directory and do not
include a `papers/` prefix.

The hashes were verified against cuPhoton Papers commit
`60141cdde89be393cf8839f505fcc4ac4017f7d2`. Its reference OCR and Markdown
sidecars came from the historical Hiccup implementation at
`5b3d8ac46eb25395b37571732dc165be30372a74`.

## Safety boundary

All generated inputs, outputs, logs, responses, and adapter artifacts must stay
under a newly created scratch destination. The harness:

- validates each source PNG, TXT, OCR, and Markdown hash, plus PNG dimensions,
  before launching a provider;
- rejects a destination equal to or nested below the canonical input root;
- rejects an existing nonempty destination and has no force or overwrite mode;
- byte-copies PNG and TXT inputs into `inputs/` and verifies the copies again;
- includes only scratch input paths in provider requests; and
- keeps reference paths, contents, and hashes evaluator-only, never including
  them in provider request JSONL.

The provider command is trusted code, not an operating-system sandbox. In
particular, `egress: "none"` is a declaration that the adapter does not use the
network; the harness does not enforce that declaration or restrict which local
files the process can open. Review local adapters, pin their source, and use
offline runtime settings. Any adapter that may transmit an input must declare
`egress: "external"` and pass the external-authorization gate below.

Do not run `--force` against either canonical corpus. The benchmark harness has
no such option.

## Provider specification

Provider configuration is machine-specific and belongs in scratch space, not
in the repository. A provider specification has this shape:

```json
{
  "schema_version": 1,
  "provider_id": "example-local-model",
  "argv": [
    "/absolute/path/to/python",
    "/absolute/path/to/provider_adapter.py"
  ],
  "cwd": "/absolute/path/to/adapter-workdir",
  "egress": "none",
  "model": {
    "requested": "organization/model-name",
    "revision": "full-immutable-model-revision"
  },
  "env": {
    "HF_HUB_OFFLINE": "1",
    "PYTHONHASHSEED": "0",
    "TRANSFORMERS_OFFLINE": "1"
  },
  "lanes": {
    "parity": {
      "prompt": "portable literal-transcription instruction",
      "settings": {}
    },
    "best": {
      "prompt": "provider-specific best-capability instruction",
      "settings": {}
    }
  },
  "timeout_seconds": 900,
  "secret_env_names": []
}
```

`egress` must be either `none` or `external`. Put secret variable names, never
secret values, in `secret_env_names`. Other secret-looking ambient variables
are scrubbed. An optional `env` string mapping provides explicit nonsecret
overrides; secret-looking keys are rejected. Ambient inference controls such as
CUDA, PyTorch, Transformers, Paddle, and vLLM variables, plus
`PYTHONHASHSEED`, are scrubbed unless they are explicit in `env` or intentionally
allowlisted by name as a secret. This prevents an unrecorded shell setting such
as `VLLM_ATTENTION_BACKEND` from silently changing an arm. Other nonsecret,
non-inference ambient variables remain available to the provider.

The run record includes the effective reproducibility-sensitive environment,
with secret values omitted, and its SHA-256. That recorded environment is part
of `experiment_sha256`, together with the exact manifest and provider-spec file
hashes, harness source hashes, and working-tree diff hash. Each request carries
that experiment identity, and it contributes to the deterministic attempt ID.
A lane may set `include_raw_text`; it defaults to false for parity and true for
best, and parity may never enable it.

For each lane and repeat, the harness starts a fresh adapter process and
appends these arguments to `argv`:

```text
--requests-jsonl ABSOLUTE_PATH
--responses-jsonl ABSOLUTE_PATH
--artifacts-dir ABSOLUTE_PATH
```

The request file contains the complete three-case suite so an adapter may use
native batching. Fresh processes isolate lanes and repeats from mutable model,
engine, or logits-processor state. Each provider starts in a new process
session. After a normal exit, the harness cleans up lingering descendants. On
timeout it sends `SIGTERM` to the whole process group, waits two seconds, and
uses `SIGKILL` if needed, preventing a spawned inference server from leaking
into the next repeat. Standard output and standard error are captured as logs;
`responses.jsonl` is the protocol channel.

The harness rehashes the provider executable and argv-discoverable source or
configuration files ending in `.json`, `.py`, `.sh`, `.toml`, `.yaml`, or
`.yml` before and after each lane/repeat. It does not infer imported modules or
arbitrary file arguments, so adapters must record those dependencies when they
matter. If a tracked file changes, the harness aborts before the next attempt
and writes `provider-drift.json`; a drift-aborted run intentionally has no
final results or summary.

Each request object contains:

- `schema_version`, `experiment_sha256`, deterministic `attempt_id`, `case_id`,
  `lane`, and one-based `repeat`;
- the scratch PNG path, expected SHA-256, and dimensions;
- `raw_text`, which is `null` in the parity lane and a scratch TXT path plus
  hash and exact UTF-8 text in the best-capability lane;
- the exact instruction and its SHA-256; and
- the lane settings.

The parity request has this exact shape; all shown paths are absolute scratch
paths:

```json
{
  "schema_version": 1,
  "experiment_sha256": "<reproducibility identity sha256>",
  "attempt_id": "<deterministic sha256>",
  "case_id": "kessler-2015-page-14",
  "lane": "parity",
  "repeat": 1,
  "input": {
    "image": {
      "path": "/tmp/.../inputs/Kessler.../page-14.png",
      "sha256": "dc40...82e",
      "dimensions": {"w": 3400, "h": 4400}
    },
    "raw_text": null
  },
  "instruction": "<exact common prompt>",
  "instruction_sha256": "<sha256 of UTF-8 prompt>",
  "settings": {"temperature": 0.0}
}
```

In the best-capability lane, only `input.raw_text` changes shape:

```json
{
  "path": "/tmp/.../inputs/Kessler.../page-14.txt",
  "sha256": "7482...0ce",
  "text": "<exact UTF-8 text>"
}
```

Each response object must echo `attempt_id` and provide:

- `status`;
- `output.format` and exact `output.text`;
- the exact rendered `effective_request.prompt` (including provider wrappers)
  and its effective input artifacts;
- requested model identity and the exact resolved revision;
- runtime, engine, package versions, and selected attention backend;
- complete effective generation parameters;
- generated-token count, finish reason, stop reason, and EOS/max-token
  termination details;
- usage and timing information; and
- at least one raw provider response or supporting artifact beneath the
  supplied artifacts directory.

A completed response with missing provenance is a failed attempt. Adapters
must preserve the provider's raw response before normalization or evaluation.
The completed response shape is:

```json
{
  "schema_version": 1,
  "attempt_id": "<echo exact request ID>",
  "status": "completed",
  "output": {
    "format": "markdown",
    "text": "<exact output; no harness normalization>"
  },
  "effective_request": {
    "prompt": "<actual rendered prompt including wrappers>",
    "input_artifacts": [
      {
        "path": "/tmp/.../inputs/Kessler.../page-14.png",
        "sha256": "dc40...82e",
        "dimensions": {"w": 3400, "h": 4400},
        "transform": "original"
      }
    ]
  },
  "model": {
    "requested": "PaddlePaddle/PaddleOCR-VL-1.6",
    "revision": "<requested full pin>",
    "resolved_snapshot": "<actual full pin>"
  },
  "runtime": {
    "python": "3.12.13",
    "engine": "transformers",
    "engine_version": "4.57.6",
    "packages": {
      "torch": "2.11.0+cu130",
      "transformers": "4.57.6"
    },
    "attention_backend": "sdpa"
  },
  "generation": {
    "params": {"temperature": 0.0},
    "input_tokens": 1024,
    "output_tokens": 1536,
    "finish_reason": "stop",
    "stop_reason": null,
    "eos": true,
    "max_tokens": false
  },
  "usage": {
    "wall_seconds": 1.23,
    "cost_usd": null
  },
  "artifacts": [
    {
      "relative_path": "raw-response.json",
      "sha256": "<artifact bytes sha256>"
    }
  ]
}
```

The completed-response schema is strict. `model.resolved_snapshot`, Python,
engine, engine version, and finish reason are nonempty strings.
`runtime.packages` is a mapping whose keys and values are nonempty strings;
adapters should list every package that materially participates in inference.
`runtime.attention_backend` is also a nonempty string: report the selected local
backend such as `sdpa` or `FLASH_ATTN`, use `not_applicable` when the stack has
no attention backend, and use `provider_managed` when an external service does
not expose its internal backend. Do not use `null` for this field.

Input and output token counts are nonnegative integers. `generation.params` is
the complete effective parameter mapping, not merely the requested settings.
`eos` and `max_tokens` are booleans and cannot both be true. If both are false,
`stop_reason` must be a nonempty string identifying the other termination
condition. Wall time is nonnegative; `cost_usd` may be `null` when no cost is
known, otherwise it must be nonnegative. `artifacts` must be a nonempty list of
safe relative paths with lowercase SHA-256 values, and each path must resolve to
a regular file beneath the supplied artifacts directory.

An error response keeps `schema_version` and `attempt_id`, sets `status` to
`error`, and adds `error.type` and `error.message`; output and provenance may
be absent. Artifact paths are relative and must resolve beneath the supplied
artifacts directory. The harness computes output, response, and artifact
hashes itself rather than trusting provider assertions.

## Evidence and evaluator snapshots

Before launching a provider, the harness snapshots the exact manifest and
provider-spec bytes under `evidence/`. It records hashes for executable and
source/configuration files discoverable from provider `argv`, copies eligible
provider source/configuration files, snapshots the active benchmark, OCR
quality, and command modules, and saves a binary Git diff for `src`, `tests`,
and `benchmarks`. `run.json` records those artifacts, the command invocation,
repository SHA/branch/status, selected lanes, timeout, model, recorded
environment, authorization state, and experiment identity. This preserves the
actual harness and adapter evidence even when the checkout or scratch adapter
changes later.

Evaluator inputs are handled separately. During preparation, the harness reads
and rehashes the raw TXT, grounded OCR, and reference Markdown into memory.
Quality evaluation uses those verified bytes rather than rereading a canonical
corpus after a long inference run. Only the PNG and, when enabled for a lane,
raw TXT appear in provider requests. After all provider processes finish, the
verified evaluator bytes are written under `evaluator/`, preserving corpus
relative stems, with an evaluator manifest of their hashes. The summary records
a hash of that evaluator evidence.

## Running the smoke benchmark

Create the provider specification in scratch space, then run both lanes five
times:

```bash
BENCH_SCRATCH=$(mktemp -d /tmp/sneeze-tpn-ocr-benchmark.XXXXXX)
PLUGIN_ROOT=/home/trentn/src/sneeze-plugin-tpn
BENCH_MANIFEST="$PLUGIN_ROOT/benchmarks/ocr/cuphoton-regressions.jsonl"

sne ocr-benchmark \
  --manifest "$BENCH_MANIFEST" \
  --input-root /home/trentn/src/cuphoton-papers/papers \
  --provider-spec "$BENCH_SCRATCH/provider-spec.json" \
  --dest-dir "$BENCH_SCRATCH/run" \
  --lanes parity,best \
  --repeats 5
```

The destination layout is:

```text
run/
├── run.json
├── evidence/
│   ├── manifest.jsonl
│   ├── provider-spec.json
│   ├── provider/
│   └── harness/
├── evaluator/
│   ├── manifest.json
│   └── CORPUS_RELATIVE_STEMS.{txt,ocr,md}
├── inputs/
├── parity/
│   └── repeat-NNN/
│       ├── requests.jsonl
│       ├── responses.jsonl
│       ├── stdout.log
│       ├── stderr.log
│       ├── artifacts/
│       └── outputs/CASE_ID.md
├── best/
│   └── repeat-NNN/
│       └── ...
├── results.jsonl
└── summary.json
```

Malformed manifests or provider specifications, source hash mismatches, unsafe
destinations, and unapproved external egress fail before any subprocess starts.
A provider nonzero exit, timeout, or missing, duplicate, or malformed response
ID becomes a durable error row. The harness continues other attempts when
feasible, writes `results.jsonl` and `summary.json`, preserves raw evidence, and
then exits nonzero if any attempt, provenance check, or assertion failed.

## External providers

An `external` provider specification requires both `--allow-external` and a
structured authorization record in the provider specification:

```json
{
  "egress": "external",
  "secret_env_names": ["OPENAI_API_KEY"],
  "external_authorization": {
    "approved_by": "approver identity",
    "approved_at": "2026-08-03T00:00:00Z",
    "scope": "named pages, provider, model, and transmitted artifacts",
    "retention": "provider retention and training terms",
    "budget_usd": 10.0
  }
}
```

The four text fields must be nonempty, and `budget_usd` must be a nonnegative
number. The harness records this object and the external flag in `run.json` and
the snapshotted provider specification. It does not validate that
`approved_at` is a timestamp, independently verify the approver, or enforce the
declared budget; those remain operator and adapter responsibilities.

Before the first upload, stop and obtain explicit approval for the provider,
exact model, document inputs, retention/training terms, and budget. Approval to
build or test the harness is not approval to transmit a paper page. Put the API
credential's environment-variable name in `secret_env_names`; the value is
passed to the provider but omitted from recorded environment evidence. An API
adapter should report `runtime.attention_backend` as `provider_managed` unless
the service exposes a more exact value.

After approval, the command shape is the same with one additional flag:

```text
--allow-external
```

## Advance gate

The three-page smoke benchmark passes only when, for every case, lane, and
repeat:

- provider status is completed;
- all required provenance is complete; and
- every manifest assertion passes.

The resulting summary must report `all_pass: true`. With two lanes and five
repeats, this means all 30 attempts pass. Catastrophic failures are hard counts
and must not be hidden by an average score. Per-case/lane groups also report
finish-reason, EOS, and max-token counts; output, token, timing, recall, axis,
and repetition ranges; exact-output uniqueness; and known versus unknown cost.

A passing provider may advance to the stratified 60-page development and
sealed-holdout benchmark. Run local candidates first. Pause again for explicit
upload and budget approval before testing any external API provider.
