# CS229 Machine Learning — Study Notes & Implementations

Self-study journey following [Stanford CS229](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) by Andrew Ng — 20 lectures across 14 topics, from Linear Regression to Reinforcement Learning.

Each lecture is distilled into structured notes and clean NumPy implementations. Selected topics include a Kaggle project to validate theory on real-world data.

---

## Topics

```
01 Linear Regression                        08 Approximation, Estimation Error & ERM
02 Locally Weighted & Logistic Regression   09 Decision Trees & Ensemble Methods
03 Perceptron & GLMs                        10 Boosting
04 GDA & Naive Bayes                        11 Neural Network
05 Support Vector Machine                   12 EM Algorithm & Factor Analysis
06 Kernel                                   13 PCA & ICA
07 Data, Model & Cross-Validation           14 Reinforcement Learning
```

---

## Lecture Structure

Each lecture folder follows this layout:

```
XX Topic Name/
├── note.md          # Derivations, geometric intuitions, probabilistic interpretations,
│                    # connections to related work, and research-level insights
├── *.py             # From-scratch NumPy implementations following lecture formulations
└── project/         # (Selected lectures) Kaggle competition applying the lecture's methods
    ├── src/
    └── project-note.md   # Experiment log: results, key learnings, theory↔practice gaps
```

> `project/` is only present for lectures where a suitable real-world dataset exists to meaningfully stress-test the concepts.

---

## Philosophy

- **Notes** — not transcripts. Captures what's worth retaining: proofs, intuitions, non-obvious insights, and links to broader ML theory.
- **Code** — not wrappers. Core algorithms implemented from scratch in NumPy; no scikit-learn for anything covered in lecture.
- **Projects** — not toy examples. Kaggle datasets chosen to surface practical constraints (noise, missing values, high dimensionality) that textbook problems hide.

---

## Requirements

```
numpy                  # all implementations
pandas, scikit-learn   # project/ folders only
```
