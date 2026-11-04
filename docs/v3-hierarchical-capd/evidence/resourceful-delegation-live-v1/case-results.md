# Per-case results

- `access-denied`: repeat 1 fail (inference_cancelled); repeat 2 fail.
- `broken-dependency`: repeat 1 fail (live_inference_budget_exhausted); repeat 2 pass.
- `cancel-before-dispatch`: repeat 1 pass; repeat 2 pass.
- `changed-form`: repeat 1 fail (inference_cancelled); repeat 2 pass.
- `empty-result`: repeat 1 pass; repeat 2 fail.
- `essential-input`: repeat 1 fail (inference_cancelled); repeat 2 fail (inference_cancelled).
- `lookup-failure`: repeat 1 fail (inference_cancelled); repeat 2 fail (inference_cancelled).
- `missing-dependency`: repeat 1 fail (inference_cancelled); repeat 2 pass.
- `owner-notes`: repeat 1 pass; repeat 2 pass.
- `ready-runtime`: repeat 1 fail; repeat 2 fail (openai_codex_invalid_json).
- `recurring-check`: repeat 1 fail; repeat 2 pass.
- `restart-after-inspection`: repeat 1 fail (live_inference_budget_exhausted); repeat 2 pass.
- `schedule-proposal`: repeat 1 pass; repeat 2 pass.
- `unfamiliar-form`: repeat 1 fail (openai_codex_empty_response); repeat 2 fail (openai_codex_empty_response).
