# PR 11850 review evidence

## Scope

- Source PR: `unslothai/unsloth#11850`
- Base: `f9bffe265889379126785d129700345a11f38b60`
- Original head: `9b3056c8200c25d66bc83f699eb76e36ea18a617`
- GitHub merge ref tested: `9cfb8bb323f4e553d6706c70a0fe392424d49e26`
- Mirror PR: `mahiatlinux/unsloth-staging-review#51`
- Host: Ubuntu 26.04, Linux 7.0.0-31-generic, x86_64
- Accelerator: NVIDIA GeForce RTX 5060 Ti 16 GB, driver 595.91.07

## Review conclusion

No reachable correctness defect was confirmed in the source head. The change makes the
training resolver reuse the same audio, transcript, and speaker detection that the dataset
check already uses, while preserving exact-name and explicit-mapping precedence.

## Executed checks

| Check | Ref/runtime | Result |
|---|---|---|
| New source regression test | head, Python 3.13 | 8 passed |
| Negative control: head test against base trainer | base, Python 3.13 | 6 failed and 2 passed for the expected column-resolution mismatch |
| New source regression test | cached GitHub merge ref, Python 3.13 | 8 passed |
| Independent resolver boundary tests | head, Python 3.13 | 5 passed |
| Independent resolver boundary tests | cached GitHub merge ref, Python 3.13 | 5 passed |
| Independent resolver boundary tests | head, Python 3.11 | 5 passed |
| Audio regression group | head, Python 3.13 | 107 passed |
| Ruff format and check | repaired-head worktree | passed with a clean worktree |

The independent boundaries cover case-insensitive names, value-detected audio, exact-name
precedence over detector output, empty and leading-null datasets, detector call count, and
explicit mapping precedence.

The 107-test group was:

```text
studio/backend/tests/test_audio_detected_columns.py
studio/backend/tests/test_audio_dataset_decode.py
studio/backend/tests/test_audio_eval_dataset.py
studio/backend/tests/test_whisper_audio_vlm_eval_dataset.py
```

## Repository CI observed on the source head

GitHub showed the source head green for the Python 3.11 floor job, all three Python 3.13
backend shards, CPU Studio and rest suites, lint, wheel, Linux startup, native Windows API/UI
and GGUF jobs, macOS startup and UI/API/inference, and the default/latest dependency matrix.
The Kaggle GPU workflow was skipped, so the local live GPU run is recorded separately.

## Live Studio UI/GPU A/B

Two clean, isolated source installs ran the same generated 24 kHz WAV JSONL with columns
`audio` and `normalized_text`, no manual mapping, `unsloth/csm-1b`, LoRA, and one training
step. The run used torch 2.11.0+cu130, Transformers 5.10.2, CUDA toolkit 13.0, Triton 3.6.0,
the RTX 5060 Ti, and task-local Chromium 153 via Playwright 1.63.0 at 1500x1000.

| Fact | Base | Head |
|---|---|---|
| Terminal phase | error | completed |
| Missing-text error | true | false |
| Step | 0 | 1 / 1 |
| Loss | none | 4.982524394989014 |
| Grad norm in UI | none | 3.124 |
| Resolver log | `No text column found` | `audio_col='audio', text_col='normalized_text'` |

The manually inspected composite is `ui/combined/pr11850_before_after.png`; `ui/meta.json`
contains the structured facts, and the per-side Studio logs contain the complete GPU worker
trace. The image visibly shows the base Current Run card in Error with the exact missing-column
message and the head card Completed at 100 percent with one step and its metrics.
