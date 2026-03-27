# Results And Discussion

## Experimental Setup

To evaluate the proposed self-evolving NIDS, a replay benchmark was run on `1,000` CICIDS2017 graph windows generated from the real processed dataset at [graphs.pt](c:/Users/deepanshu/Documents/nids-agent/data/processed/graphs.pt). The same baseline checkpoint at [best_gnn.pt](c:/Users/deepanshu/Documents/nids-agent/experiments/checkpoints/best_gnn.pt) was evaluated in two modes:

1. **Baseline replay**
   The trained GNN checkpoint was used without any online adaptation.

2. **Adaptive replay**
   The self-evolving agent monitored the replay stream, detected drift, generated an adaptation hypothesis, evaluated the candidate update, and accepted the new model only when online validation conditions were satisfied.

The benchmark artifact for this run is stored at [replay_benchmark_20260326-220945.json](c:/Users/deepanshu/Documents/nids-agent/experiments/results/replay_benchmark_20260326-220945.json).

## Quantitative Results

| Metric | Baseline | Adaptive |
| --- | ---: | ---: |
| Window accuracy | 90.90% | 91.40% |
| Window F1 | 0.9327 | 0.9362 |
| Attack precision | 0.8740 | 0.8801 |
| Attack recall | 1.0000 | 1.0000 |
| Average confidence | 0.9856 | 0.9858 |
| Drift events observed | 2 | 1 |
| Hypotheses tried | 0 | 1 |
| Hypotheses accepted | 0 | 1 |
| Final model version | v0 | v1 |
| Runtime | 5.26 s | 7.43 s |

## Key Observations

- The adaptive system produced a **higher window-level accuracy** than the static baseline (`91.40%` vs `90.90%`).
- The adaptive system also achieved a **slightly higher F1 score** (`0.9362` vs `0.9327`), indicating a small but measurable improvement in detection quality.
- Attack recall remained `1.0000` in both settings, while adaptive replay slightly improved precision (`0.8801` vs `0.8740`), meaning the adapted model reduced false positives without missing attack windows in this replay slice.
- The adaptive pipeline successfully completed the full self-evolution loop:
  - drift was detected at **window 136**
  - the trigger came from the **SHIFT** detector
  - trigger reason: `Recent drift score rose from 0.018 to 0.032`
  - the candidate update was evaluated and **accepted**
  - the model version changed from **v0** to **v1** at **window 137**

## Interpretation

These results show that the project is no longer only a visualization demo. The system now demonstrates the intended closed-loop behavior:

- ingest real graph windows
- detect distribution change
- attempt online model adaptation
- accept or reject the hypothesis based on validation logic
- update and expose the new model version in the live system

This is an important milestone because it aligns the implementation more closely with the project goal of a **self-evolving intrusion detection agent** rather than a static classifier with a dashboard.

## Limitations

- The measured gains are currently **modest**, so more extensive replay experiments are needed across larger slices and different traffic segments.
- The accepted update in this benchmark was driven by online loss/calibration improvement rather than a large F1 jump, which is reasonable for a strong baseline but should be explained clearly in the final report.
- The current adaptive loop is best described as a **working online adaptation system inspired by the MAML design**, while the full offline meta-learning story can still be strengthened further.

## Conclusion

The experiment demonstrates that adaptive replay is functioning correctly and can outperform the static baseline on the evaluated stream segment. Most importantly, the system now produces a real and traceable **model evolution event (`v0 -> v1`)**, which is a strong result for the project demonstration and final presentation.

## Viva / PPT Summary

- Baseline vs adaptive replay was evaluated on `1,000` real CICIDS2017 graph windows.
- Adaptive replay improved accuracy from `90.90%` to `91.40%`.
- Adaptive replay improved F1 from `0.9327` to `0.9362`.
- The system detected drift at window `136` and accepted an adaptation.
- The model evolved from `v0` to `v1` during live replay.
- This validates the project’s self-evolving detection claim in prototype form.
