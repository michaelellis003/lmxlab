# Test-Time Compute Below 1B: What 10 Experiments Taught Us About Sampling, Grokking, and the Limits of Prediction

*A research summary from the lmxlab project. All experiments run on a single Apple M3 Pro (36GB) using MLX. All sampling at temperature=0.8.*

---

## The Question

Test-time compute (TTC) scaling -- using extra compute at inference through sampling strategies like best-of-N -- has been studied primarily at scales above 1B parameters. Does it work at 10-30M? And what can it tell us about how models learn?

We ran 10 pre-registered experiments (HYP-007 through HYP-016) over 5 days on a single task (modular arithmetic, mod 97) to find out. The results are specific to this task and scale, but several findings generalize as hypotheses worth testing more broadly.

## Key Findings

**1. TTC works 50-100x below the literature's minimum scale.** At 10M parameters, pass@64 is 11.9x pass@1 on modular addition (mod 97, n=3 seeds). Prior work studied TTC at 1.5B+ parameters. We do not know whether 12-15x amplification is higher or lower than at larger scales, as prior work reports different metrics. (HYP-007)

**2. TTC amplification is architecture-independent within a task.** Across 4 architecture families (n=3 seeds each), amplification factors range 13.4x-14.8x (max/min ratio = 1.10x). However, amplification is strongly task-dependent: 12.5x for addition but only 3.8x for multiplication (HYP-012). The amplification factor reflects base accuracy, not architecture. (HYP-008, HYP-012)

**3. During grokking, pass@64 saturates long before greedy accuracy.** In one well-characterized trajectory (LLaMA, seed 42; 2/3 seeds did not grok within 50K steps), pass@64 reached 98.9% at step 4K while greedy accuracy was only 14.1%. Greedy accuracy did not reach 99% until step 43K. This suggests grokking may involve a shift from stochastic to deterministic expression of learned capabilities, though we lack mechanistic confirmation and this has only been observed on a single task. (HYP-009)

**4. Val loss inversely predicts pass@k across architectures.** LLaMA has the *worst* val_loss but the *best* pass@k at every k (n=3 seeds per architecture). The mechanism: LLaMA concentrates more probability on the correct answer token while being proportionally worse at predicting prompt tokens, which dominate average loss. This is the most mechanistically well-supported finding in the series. (HYP-008, HYP-011)

**5. P(correct) at the answer token predicts TTC amplification with r = -0.98.** A single forward pass tells you how much sampling will help (n=6 condition means across 2 tasks x 3 seeds). This relationship is partly tautological -- pass@1 approximates P(correct), so amplification = pass@64/pass@1 is mechanically related to P(correct). Still practically useful. (HYP-013)

**6. SSM hybrids grok faster than pure attention on modular arithmetic.** Bamba groks in 20K steps vs LLaMA's 44K (n=1 seed per architecture in HYP-014). Given that grokking onset varies 12x across seeds (Finding #8), these single-seed comparisons should be treated as suggestive, not definitive. (HYP-014)

**7. MoE architectures with 8% more parameters grok more reliably.** MoE-Jamba (7.6M params) grokked 9/10 seeds; noMoE-Jamba (7.0M params) grokked 1/3 seeds. This could reflect extra capacity (590K more parameters) rather than MoE routing specifically. A parameter-matched control is needed to distinguish these. (HYP-015, HYP-016)

**8. Grokking onset is massively seed-dependent and unpredictable.** Across 10 seeds of identical MoE-Jamba, grokking onset ranges from 4K to 48K steps (12x range). No early metric at step 2K -- not pass@64 (rho=0.11, p=0.76), not val_loss (rho=-0.06, p=0.87), not val_acc (rho=-0.01, p=0.99) -- predicts which seeds will grok. This is the most statistically robust finding (n=10). (HYP-015, HYP-016)

---

## The Experiments

### Phase 1: Does TTC Work at Small Scale? (HYP-007, HYP-008)

We trained 10M-parameter models on modular addition (mod 97) -- a task with a verifiable correct answer, enabling exact pass@k measurement. For each test prompt (e.g., "42 + 55 ="), we sampled 64 completions at temperature 0.8 and checked if any produced the correct answer.

**Result:** pass@64/pass@1 amplification of ~12-15x across all architectures tested (n=3 seeds per architecture). The scaling curve shows no sign of saturation at k=64.

| Architecture | pass@1 | pass@64 | Amplification | Seeds |
|-------------|--------|---------|---------------|-------|
| LLaMA | 0.56% | 8.29% | 14.8x | 3 |
| Falcon-H1 | 0.43% | 6.27% | 14.6x | 3 |
| Jamba | 0.44% | 6.01% | 13.7x | 3 |
| Bamba | 0.47% | 6.27% | 13.4x | 3 |

These models are tiny (10M params) and partially trained, yet sampling helps enormously. All results use temperature=0.8; amplification factors may be sensitive to this choice, which was not ablated.

### Phase 2: TTC Meets Grokking (HYP-009, HYP-010)

We trained LLaMA on the same task with weight decay (wd=0.1, which induces grokking) and tracked pass@k throughout training. Of 3 seeds tested, 1 grokked within 50K steps (seed 42). The trajectory for that seed:

```
Step 2K:  pass@1 = 0.7%   pass@64 = 14.1%   (20x amplification)
Step 4K:  pass@1 = 14.1%  pass@64 = 98.9%   (7x)
Step 8K:  pass@1 = 24.3%  pass@64 = 99.7%   (4x)
Step 43K: pass@1 = 99.0%  pass@64 = 100.0%  (1x -- grokked)
```

**The observation:** pass@64 saturated near 100% at step 4K, 39,000 steps before greedy accuracy caught up. One interpretation is that the generalization circuit forms gradually but the model can only express it deterministically later -- grokking as "sudden decodability" rather than "sudden generalization." However, this is a single trajectory on a single task, and we have no mechanistic evidence (no circuit analysis, no weight-space inspection) to confirm this interpretation. It should be treated as a hypothesis, not a conclusion.

At 30M parameters (HYP-010), TTC amplification was similar (11.9x) but absolute performance was *worse*. The 30M model had worse val_loss (3.10 vs 2.75) and worse pass@1 (0.43% vs 0.56%) than the 10M model despite 3x more parameters -- the classic overparameterized-undertrained pattern.

### Phase 3: Why Does Val Loss Lie? (HYP-011, HYP-012, HYP-013)

An anomaly from HYP-008 demanded explanation: LLaMA had the worst val_loss across all architectures yet the best pass@k. How?

**Per-token loss decomposition** (HYP-011) revealed the answer. Val_loss averages over all token positions, but pass@k only cares about the answer token. LLaMA was 29-31% better at predicting the answer token but only 7% better at prompt tokens. Since prompts have ~5 tokens and the answer has ~1, the answer-token advantage gets diluted 5:1 in the average.

Moreover, LLaMA's answer-token distribution was simultaneously more entropic (diverse) *and* more correct -- the ideal combination for best-of-N sampling. Higher entropy means each sample explores different answers, while higher P(correct) means more of those samples hit the right one.

**Cross-task testing** (HYP-012) showed amplification is strongly task-dependent: 12.5x for addition but only 3.8x for multiplication (n=3 seeds per task). The difference traces to base accuracy: multiplication has 3.6x higher pass@1 despite worse val_loss -- another instance of the val-loss-inversely-predicts-accuracy pattern.

This led to a clean result: **P(correct) at the answer token predicts TTC amplification with Pearson r = -0.98** (HYP-013, n=6 condition means). One forward pass tells you how much sampling will help. The relationship is partly tautological (pass@1 ~ P(correct), so amplification ~ 1/P(correct)), but it eliminates the need for expensive pass@k sweeps.

### Phase 4: Grokking Across Architectures (HYP-014, HYP-015, HYP-016)

We asked: does grokking depend on architecture?

**Single-seed comparison** (HYP-014, n=1 per architecture):

| Architecture | Grok Step | p@64 at Step 2K | Note |
|-------------|-----------|-----------------|------|
| Bamba | 20K | 99.6% | n=1 seed |
| Falcon-H1 | 26K | 78.5% | n=1 seed |
| Jamba | 36K | 96.3% | n=1 seed |
| LLaMA | 44K | 46.2% | n=1 seed |

The early TTC signal (pass@64 at step 2K) had a perfect rank correlation with grokking order. This seemed like a powerful predictive finding -- until HYP-016 tested it properly.

**The MoE surprise** (HYP-015, n=3 seeds per condition): Jamba showed "un-grokking" in HYP-014. We hypothesized MoE routing caused this. The multi-seed ablation showed the opposite: MoE-Jamba grokked 3/3 seeds; noMoE-Jamba grokked 1/3 seeds. The un-grokking was seed-specific (1/3 MoE seeds), not architecture-caused.

**The prediction failure** (HYP-016, n=10 seeds): We ran 10 seeds of MoE-Jamba to test whether pass@64 at step 2K predicts *which seeds* grok. It doesn't. At all. Spearman rho = 0.111 (p = 0.76). The seed with the lowest early pass@64 (0.445) grokked at 12K (among the fastest). The seed with the highest (0.981) never grokked within 50K steps.

**What this means:** The cross-architecture TTC-grokking correlation was real but confounded. Architectural inductive bias drives *both* early TTC and grokking speed -- it's not that TTC causally predicts grokking. Within a single architecture, grokking onset is essentially random, varying 12x across seeds with no predictable signal from any aggregate metric we tested.

---

## What We Learned About Research Methodology

**1. Single-seed grokking experiments are unreliable.** Grokking onset varies 12x across seeds. Any single-seed result can be misleading. We made this mistake in HYP-014 (attributing instability to MoE based on one seed) and corrected it in HYP-015 (multi-seed ablation).

**2. Confounders hide in correlations.** The perfect cross-architecture rank correlation (rho = 1.0, n=4) between early TTC and grokking order was entirely confounded. Only the within-architecture test (HYP-016, rho = 0.11, n=10) revealed this. Always test the causal claim, not just the correlation.

**3. Val loss is a bad proxy for task accuracy.** The val-loss-inversely-predicts-pass@k pattern appeared in three separate experiments (HYP-008, HYP-011, HYP-012). When your evaluation metric averages over positions and only some positions matter for the task, the average can mislead.

**4. Negative results have value.** HYP-016's finding that grokking is unpredictable from early signals is as informative as the positive findings. It tells us grokking onset depends on weight initialization details invisible to aggregate metrics.

**5. Pre-registration prevents narrative bias.** Several experiments produced surprising results (MoE helps grokking, val_loss inversely predicts accuracy, prediction fails within-architecture). Without pre-registered hypotheses and falsification criteria, it would be tempting to cherry-pick the narrative.

---

## Limitations

**Task specificity.** All experiments use modular arithmetic (mod 97), a narrow algorithmic task with a single correct answer per prompt. Results may not generalize to language modeling, reasoning, or other tasks. The val-loss-vs-pass@k inversion depends on the prompt/answer token ratio, which differs across tasks.

**Scale.** All models are 10-30M parameters. TTC dynamics may differ at 100M+ where models have higher base accuracy and different loss landscapes.

**Temperature.** All sampling uses temperature=0.8. The 12-15x amplification factors may be sensitive to this choice, which was not ablated. Optimal temperature likely depends on model calibration.

**Sample sizes.** Most experiments use 3 seeds per condition. Only HYP-016 (10 seeds) has adequate statistical power. Claims from n=1 comparisons (HYP-014 architecture grokking speeds) should be treated as preliminary.

**Hardware.** All experiments on a single Apple M3 Pro GPU via MLX. TTC cost-effectiveness may differ on multi-GPU cloud infrastructure where k samples can run in parallel.

**MoE confound.** The MoE grokking advantage (Finding #7) is confounded by an 8% parameter count difference (7.6M vs 7.0M). The effect could be capacity, not routing.

---

## Consolidated Beliefs

After 10 experiments, here's where our confidence stands:

| Claim | Confidence | Evidence | Caveat |
|-------|-----------|----------|--------|
| TTC works at 10-30M params | 0.95 | HYP-007, 008, 010 (n=3 seeds x 4 archs) | Single task |
| TTC amplification is architecture-independent within a task | 0.60 | HYP-008; contradicted cross-task by HYP-012 | Task-dependent |
| Val loss is a poor proxy for task accuracy | 0.95 | HYP-008, 011, 012 (replicated 3x) | Best-supported finding |
| P(correct) predicts TTC amplification | 0.85 | HYP-013 (r=-0.98, n=6) | Partly tautological |
| pass@64 leads greedy accuracy during grokking | 0.85 | HYP-009 | 1 seed, 1 task |
| SSM hybrids grok faster than pure attention | 0.70 | HYP-014 (n=1 per arch) | Single-seed |
| MoE capacity helps grokking | 0.70 | HYP-015, 016 (n=13 total) | Parameter confound |
| Grokking onset is seed-dependent and unpredictable | 0.95 | HYP-015, 016 (n=10 seeds) | Most robust finding |

---

## Open Questions

1. **What determines whether a seed groks?** If aggregate metrics can't predict it, weight-space analysis (eigenvalue structure, circuit formation) might. Clauw et al. (ICML 2024) found that synergy peaks in early training correlate with eventual grokking in a 2-layer FC network on the same task (mod 97, 5 seeds). Testing whether synergy differentiates grokking from non-grokking seeds in transformer/hybrid architectures is the natural next step.

2. **Does the pass@64-leads-greedy-accuracy pattern generalize beyond modular arithmetic?** The clean separation may be specific to tasks with peaked answer distributions. Testing on tasks with more diffuse correct-answer distributions (e.g., language modeling perplexity) would clarify scope.

3. **Why does MoE help grokking?** Is it the extra capacity (more parameters for the generalization circuit) or the routing diversity (multiple pathways to the solution)? A parameter-matched dense model (scaling d_ff to match MoE parameter count) would distinguish these.

4. **How does 12-15x amplification compare to larger scales?** Prior TTC work (Snell et al., Brown et al.) reports different metrics, making direct comparison difficult. Running the same pass@k protocol at 100M+ parameters would clarify whether amplification is scale-dependent.

5. **How sensitive is amplification to temperature?** All our results use T=0.8. A temperature sweep would establish whether the 12-15x finding is robust or an artifact of this specific choice.

---

## Reproducibility

All experiments are fully reproducible:
- **Code:** [lmxlab](https://github.com/michaelellis003/lmxlab)
- **Recipes:** `recipes/hyp007_*.py` through `recipes/hyp016_*.py`
- **Results:** `experiments/hyp0*_results.json`
- **Methodology:** Pre-registered hypotheses in `memory/hypotheses.md`
- **Hardware:** Apple M3 Pro, 36GB, single GPU via MLX

Each experiment uses the `/hypothesis` -> run -> `/interpret` workflow with Bayesian belief updates and competing hypotheses. Total compute: approximately 24 GPU-hours across all 10 experiments.

---

*Built with lmxlab on Apple Silicon. No cloud GPUs were harmed in the making of these findings.*
