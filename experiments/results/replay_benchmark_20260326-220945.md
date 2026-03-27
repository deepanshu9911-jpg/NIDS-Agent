# Replay Benchmark

- Timestamp: `20260326-220945`
- Graph source: `C:\Users\deepanshu\Documents\nids-agent\data\processed\graphs.pt`
- Model checkpoint: `C:\Users\deepanshu\Documents\nids-agent\experiments\checkpoints\best_gnn.pt`
- Windows evaluated: `1000`

| Metric | Baseline | Adaptive |
| --- | ---: | ---: |
| Window accuracy | 90.90% | 91.40% |
| Window F1 | 0.9327 | 0.9362 |
| Attack precision | 0.8740 | 0.8801 |
| Attack recall | 1.0000 | 1.0000 |
| Avg confidence | 0.9856 | 0.9858 |
| Drift events | 2 | 1 |
| Hypotheses tried | 0 | 1 |
| Hypotheses accepted | 0 | 1 |
| Final model version | 0 | 1 |
| Runtime (s) | 5.26 | 7.43 |

## Notes

- Baseline uses the same checkpoint for the entire replay and only records drift triggers.
- Adaptive replay uses the self-evolving agent and can update the model version during the run.
