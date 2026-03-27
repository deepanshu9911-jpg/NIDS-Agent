# Replay Benchmark

- Timestamp: `20260326-223017`
- Graph source: `C:\Users\deepanshu\Documents\nids-agent\data\processed\replay_cache.pt`
- Model checkpoint: `C:\Users\deepanshu\Documents\nids-agent\experiments\checkpoints\best_gnn.pt`
- Windows evaluated: `500`

| Metric | Baseline | Adaptive |
| --- | ---: | ---: |
| Window accuracy | 90.00% | 89.80% |
| Window F1 | 0.8913 | 0.8894 |
| Attack precision | 0.8402 | 0.8367 |
| Attack recall | 0.9491 | 0.9491 |
| Avg confidence | 0.9620 | 0.9671 |
| Drift events | 3 | 3 |
| Hypotheses tried | 0 | 3 |
| Hypotheses accepted | 0 | 3 |
| Final model version | 0 | 3 |
| Runtime (s) | 2.29 | 8.44 |

## Notes

- Baseline uses the same checkpoint for the entire replay and only records drift triggers.
- Adaptive replay uses the self-evolving agent and can update the model version during the run.
