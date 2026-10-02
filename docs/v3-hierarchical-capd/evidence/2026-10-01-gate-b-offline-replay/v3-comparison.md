# V2 / V3 replay comparison

Generated: 2026-10-01T15:53:03.790479+00:00

| Engine | Passed | Failed | Inference calls | Tool calls | Estimated tokens | Elapsed ms |
|---|---:|---:|---:|---:|---:|---:|
| tactical_v2 | 5 | 5 | 28 | 15 | 60228 | 225 |
| hierarchical_v3 | 6 | 4 | 62 | 25 | 172119 | 41 |

## Cases

### solar-project-completion

- tactical_v2: FAIL — required_capability_missing:exact_text_mutation
- hierarchical_v3: FAIL — The task stopped after repeated phases made no acceptance progress.

### lg-temperature-native-client

- tactical_v2: PASS — no policy violations
- hierarchical_v3: PASS — no policy violations

### medical-treatment-recall

- tactical_v2: FAIL — required_capability_missing:project_artifact_query
- hierarchical_v3: FAIL — The task stopped after repeated phases made no acceptance progress.

### prescription-image-ocr

- tactical_v2: PASS — no policy violations
- hierarchical_v3: PASS — no policy violations

### markdown-prescription-no-ocr

- tactical_v2: PASS — no policy violations
- hierarchical_v3: PASS — no policy violations

### tool-failure-local-fallback

- tactical_v2: PASS — no policy violations
- hierarchical_v3: PASS — no policy violations

### conflicting-authoritative-records

- tactical_v2: FAIL — expected_outer_route_mismatch
- hierarchical_v3: PASS — no policy violations

### steering-during-phase

- tactical_v2: FAIL — expected_outer_route_mismatch
- hierarchical_v3: FAIL — expected_outer_route_mismatch

### restart-mid-phase

- tactical_v2: FAIL — required_capability_missing:exact_text_mutation, required_capability_missing:project_record_search
- hierarchical_v3: FAIL — The task stopped after repeated phases made no acceptance progress.

### telegram-reminder-cross-channel-continuity

- tactical_v2: PASS — no policy violations
- hierarchical_v3: PASS — no policy violations
