# OCR provider benchmark

This benchmark compares local and API-backed document-to-Markdown providers
against the three cuPhoton pages that exposed severe DeepSeek-OCR regressions.
It is deliberately small and adversarial. Passing it qualifies a provider for
a larger stratified benchmark; it does not qualify a provider for production.

The checked-in manifest, `cuphoton-regressions.jsonl`, contains corpus-relative
stems, immutable input and reference hashes, image dimensions, and calibrated
assertions. Stems are relative to the canonical `papers` directory and do not
include a `papers/` prefix.

An assertion may include a nonempty `lanes` list. It is evaluated only for
those lane IDs; an assertion without `lanes` applies to every selected lane.
The cuPhoton fixture scopes Markdown/raw-text recall to `parity` and `best`,
while grounded-OCR recall and Acero table-completion checks target
`grounded_ocr`. This distinction matters during structured pipeline tuning:
grounded OCR is not rejected by a Markdown-specific recall contract, and the
Markdown lane cannot satisfy the independent grounded-table gate.

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

Protect canonical corpora with the private machine-local policy file
`$XDG_CONFIG_HOME/sneeze/tpn/ocr-protected-roots` (or
`$HOME/.config/sneeze/tpn/ocr-protected-roots` when `XDG_CONFIG_HOME` is
unset). Give the file mode `0600` and list one existing canonical absolute
directory per line, such as `/absolute/path/to/canonical-corpus`. Benchmark,
tuning, provider-evidence, installation-registry, hardware-receipt, and
generated-profile destinations that overlap a configured root are rejected
before creation. Policy paths are not serialized merely because they appear in
this machine-local file; normal provenance still records paths explicitly
supplied as benchmark or tuning inputs.

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

A completed response with missing provenance is a failed attempt. Local
adapters should preserve the provider's raw response before normalization or
evaluation. The hosted adapter deliberately does not retain raw response
bodies. It records their SHA-256, allowlisted response metadata, the sanitized
request shape, delivery and usage evidence, and normalized output instead. This
keeps enough evidence to audit an attempt without retaining an extra copy of
the provider payload. The completed response shape is:

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

## Built-in local backends

The structured setup commands derive model and adapter identity from the
built-in registry. Do not hand-edit those pins into a provider specification.

This section is the lower-level provider-spec workflow for benchmark
development. It is not the complete unknown-GPU onboarding procedure. Use the
root README's **Install and register local backends** and **Tune a new GPU**
sections for the trusted-filesystem contract, both runtime installations,
25/50/max tuning, repeated confirmations, qualification review, and the final
scratch smoke.

Run these examples from a clean, activated control venv containing this
checkout. The host prerequisites are Python 3.12 with `venv`, Git, Poppler,
mamba or conda, and a supported NVIDIA GPU/driver; the root README gives the
complete bootstrap. The exact pinned DeepSeek environment installs
`vllm==0.21.1rc1.dev438+g4ff865c38` from vLLM's immutable full-build-commit
CUDA 13.0 index. The mutable nightly alias no longer lists that build. Do not
substitute a nearby build and treat it as qualified; upstream retention of the
commit-specific wheel remains an external availability dependency.

```bash
LOCAL_SCRATCH=$(mktemp -d /tmp/sneeze-tpn-ocr-local.XXXXXX)
MODEL_CACHE="$LOCAL_SCRATCH/hf-cache"
MINERU_PY=/absolute/path/to/sneeze-ocr-mineru25/bin/python

sne ocr-backend-list --format json
sne ocr-backend-create-env --backend mineru25 --dry-run
sne ocr-backend-create-env --backend mineru25
sne ocr-backend-fetch-model \
  --backend mineru25 \
  --cache-dir "$MODEL_CACHE"
sne ocr-backend-write-spec \
  --backend mineru25 \
  --output "$LOCAL_SCRATCH/provider-spec.json" \
  --python-executable "$MINERU_PY" \
  --cache-dir "$MODEL_CACHE"
sne ocr-backend-doctor \
  --provider-spec "$LOCAL_SCRATCH/provider-spec.json" \
  --load-model
```

`ocr-backend-create-env` and `ocr-backend-fetch-model` both accept `--dry-run`.
Run the doctor without `--load-model` first when only a static environment and
snapshot check is appropriate. The registry reports qualification per
capability. A new or benchmark-only capability must still pass this benchmark
and the larger advance gates before production use.

### Structured GPU-profile evidence

The memory-tier capacity is
`min(floor(total_bytes * fraction), total_bytes - reserve_bytes)`. Fractions
are 25%, 50%, and 90% for dedicated max or 75% for unified max; the dedicated
and unified reserves are 2 GiB and 16 GiB respectively. Live availability is a
separate gate.

DeepSeek tuning sweeps 1, 2, and 4 GiB of engine-overhead headroom for each
eligible tier/batch candidate. Its vLLM denominator is
`max(portable GPU total, torch CUDA total)`, and the target engine allocation
is `tier budget - headroom`. Profiles retain the complete distribution
manifest's package count and SHA-256 identity; package names are not persisted
in candidate evidence. The tuning plan also binds the
installation-record, executable, cache, snapshot, weights, adapter, and
implementation identities. A resumed run requires the exact same plan and
matching artifacts, even when all filesystem paths are unchanged.

The bundled profile
`local-mineru25-nvidia-rtx-a6000-sm86-48gb` covers the 25%, 50%, and maximum
tiers. Its selected settings were reconfirmed against the final source with
one focused discovery and five cold repeats of the three-page suite for each
of the six stage/tier configurations. All 90 confirmation outputs passed with
complete provenance and EOS termination. The selected DeepSeek batch is 3 at
every tier, with 2, 1, and 1 GiB of engine headroom at 25%, 50%, and max.
Selected MinerU batches are 2, 2, and 4 respectively. The exact trusted profile
hash is reported by `sne ocr-profile-list` and in the main README.

GB10 remains unqualified. With the final pinned source and runtime tuple, two
fresh 50% attempts failed closed during the first DeepSeek cold start when the
unchanged PSI pressure watchdog tripped. Neither attempt completed a provider
call, so the result is a platform-pressure limitation rather than OCR-quality
evidence. No GB10 profile is bundled or trusted.

## Running the smoke benchmark

Create the provider specification in scratch space, then run both lanes five
times:

```bash
BENCH_SCRATCH=$(mktemp -d /tmp/sneeze-tpn-ocr-benchmark.XXXXXX)
PLUGIN_ROOT=$(git rev-parse --show-toplevel)
INPUT_ROOT=/absolute/read-only/path/to/papers
BENCH_MANIFEST="$PLUGIN_ROOT/benchmarks/ocr/cuphoton-regressions.jsonl"

sne ocr-benchmark \
  --manifest "$BENCH_MANIFEST" \
  --input-root "$INPUT_ROOT" \
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

Hosted OpenAI and Anthropic execution is benchmark-only. It is not a production
`ocr-pdf-pages` Markdown backend. Listing and planning are offline operations:
they do not read credentials, transmit source bytes, or call a provider.

Start by inspecting the dated profiles and their model, pricing, request, and
retention policies:

```bash
sne ocr-external-list
sne ocr-external-list --format json
```

Create one exact, expiring upload plan in scratch space. The command validates
the manifest-bound PNG hashes and dimensions, then writes a secret-free
`provider-spec.json`, `upload-plan.json`, and byte-free
`request-previews.jsonl`:

```bash
HOSTED_SCRATCH=$(mktemp -d /tmp/sneeze-tpn-ocr-hosted.XXXXXX)
PLUGIN_ROOT=$(git rev-parse --show-toplevel)
INPUT_ROOT=/absolute/read-only/path/to/papers
BENCH_MANIFEST="$PLUGIN_ROOT/benchmarks/ocr/cuphoton-regressions.jsonl"

sne ocr-external-plan \
  --profile openai-responses-gpt-5.6-sol \
  --manifest "$BENCH_MANIFEST" \
  --input-root "$INPUT_ROOT" \
  --dest-dir "$HOSTED_SCRATCH/plan" \
  --lane parity \
  --prompt-mode parity \
  --repeats 1 \
  --max-api-attempts 1 \
  --max-output-tokens 8192 \
  --budget-usd 100.000000 \
  --valid-hours 1
```

`--max-input-tokens` is a post-response scope check because these image APIs do
not enforce it before billing. The plan therefore prices every permitted API
attempt against the model's full documented context window at the highest
applicable input rate. Before each network call, the adapter writes a durable
reservation keyed by the exact plan, case, lane, repeat, and API-attempt slot.
Reservations remain counted after delivery. An unknown delivery state blocks
the ledger and all remaining calls. The planning command refuses a budget below
the plan-wide pessimistic maximum.

Review the receipt, plan, and request previews. A plan is not permission to
upload. Obtain approval for that exact provider profile, model, manifest,
cases, prompt, lane, transmitted artifacts, repeats, API-attempt cap, budget,
and retention policy. Record it separately:

```bash
sne ocr-external-approve \
  --plan "$HOSTED_SCRATCH/plan/upload-plan.json" \
  --output "$HOSTED_SCRATCH/plan/upload-approval.json" \
  --approved-by operator@example.com \
  --accept-retention \
  --valid-hours 1
```

The approval is bound to the plan digest and expires no later than the plan.
Changing the provider specification, executor, model profile, manifest, input
hashes, prompt, lane, raw-text scope, repeats, or budget invalidates the
binding. `--allow-external` alone is never sufficient.

Provision the credential named by the profile, `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`, through the process environment only. Never place a secret
value in an argument, provider specification, plan, approval, request preview,
or log. The harness passes only the named credential to the adapter, omits its
value from recorded environment evidence, and treats any detected credential
leak in responses or artifacts as a failed run.

The generated provider specification starts its pinned Python executable with
`-I -S -B`. The benchmark supplies only allowlisted locale values, a default
`PATH`, fixed Python safety controls, and the named credential. Before reading
that credential, the adapter verifies isolated no-site startup and attests the
Python executable, adapter, planning helper, provider specification, plan, and
approval. A changed executable or helper requires a new plan and approval.

After the exact plan and separate approval exist, run the benchmark. Lanes and
repeats default to the values in the plan:

```bash
sne ocr-benchmark \
  --manifest "$BENCH_MANIFEST" \
  --input-root "$INPUT_ROOT" \
  --provider-spec "$HOSTED_SCRATCH/plan/provider-spec.json" \
  --dest-dir "$HOSTED_SCRATCH/run" \
  --allow-external \
  --external-plan "$HOSTED_SCRATCH/plan/upload-plan.json" \
  --external-approval "$HOSTED_SCRATCH/plan/upload-approval.json"
```

The default parity plan uploads the PNG only. External raw text is opt-in. If a
best-capability experiment genuinely requires it, create and approve a new
plan with `--lane best --prompt-mode best --include-raw-text`. The raw-text
hashes and transmission scope then become part of the plan. The parity lane
cannot include raw text.

The dated policy profiles require explicit retention acknowledgement:

- OpenAI uses `store: false`, but `store: false` is not Zero Data Retention.
  The profile records the documented default abuse-monitoring retention of up
  to 30 days. GPT-5.6 implicit prompt caching is disabled with explicit mode
  and no cache breakpoints, so this benchmark performs no prompt-cache reads
  or writes.
- Anthropic Claude Fable 5 has mandatory 30-day retention for API inputs and
  outputs. Zero Data Retention is unavailable for this profile.

The Anthropic request places the image before the text, following its current
vision guidance. Fable 5 still performs provider-side downscaling when a page
exceeds 2,576 pixels on an edge or 4,784 visual tokens. The canonical
three-page PNGs exceed those limits, so compare that arm as provider-native
high-resolution OCR, not as original-pixel inference. OpenAI uses
`detail: "original"` for the same source PNG bytes.

Use `anthropic-messages-claude-fable-5` in `--profile` to plan the Anthropic
arm. Re-run `ocr-external-list --format json` before planning so any dated
policy change is visible. All plans, approvals, copied inputs, normalized
outputs, ledgers, logs, and artifacts must remain in scratch space. Never point
a hosted run destination at either canonical corpus.

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
