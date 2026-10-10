# V2 / V3 replay comparison

Generated: 2026-09-20T06:21:30.053139+00:00

| Engine | Passed | Failed | Inference calls | Tool calls | Estimated tokens | Elapsed ms |
|---|---:|---:|---:|---:|---:|---:|
| tactical_v2 | 9 | 0 | 54 | 23 | 101127 | 339 |
| hierarchical_v3 | 9 | 0 | 48 | 15 | 30724 | 57 |

## Cases

### solar-project-completion

- tactical_v2: PASS — no policy violations
- hierarchical_v3: PASS — no policy violations

### lg-temperature-native-client

- tactical_v2: PASS — no policy violations
- hierarchical_v3: PASS — no policy violations

### medical-treatment-recall

- tactical_v2: PASS — no policy violations
- hierarchical_v3: PASS — no policy violations

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

- tactical_v2: PASS — no policy violations
- hierarchical_v3: PASS — no policy violations

### steering-during-phase

- tactical_v2: PASS — no policy violations
- hierarchical_v3: PASS — no policy violations

### restart-mid-phase

- tactical_v2: PASS — no policy violations
- hierarchical_v3: PASS — no policy violations
