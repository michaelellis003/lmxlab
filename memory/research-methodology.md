# Research Methodology Lessons

Hard-won lessons about formulating hypotheses, choosing research
questions, and avoiding common traps. Distilled from our HYP-001
series (4 rounds, diminishing returns) and a meta-research review
of expert advice (Hamming, Schulman, Greydanus, Olah, Nielsen).

---

## The "Nobody Has Done This" Trap

"Nobody has studied X" can mean opportunity OR a warning sign.
**Absence of research is NOT evidence of opportunity.** You need
additional positive signals:

- Experts explicitly calling for work in the area
- Related findings suggesting importance
- Theoretical reasons to expect meaningful results
- The gap is newly created (new technique/hardware/paradigm)

**Red flags that "nobody has done this" means "don't bother":**

1. The answer is trivially predictable from existing theory
2. The question is ill-posed or confounded at your scale
3. Well-known results already cover the space (e.g., Kaplan 2020:
   "architecture details don't matter compared to scale")
4. Your own prior experiments already show no signal

**Process:** Before committing to a hypothesis, explicitly ask
"Why hasn't anyone done this?" and write down the answer. Record
it in the hypothesis entry. Only proceed if the answer is (d)
"genuinely fills a gap" — not (a) obvious, (b) ill-posed, or
(c) already done.

---

## The Fruit Fly Model (Greydanus 2020)

Small-scale research is valuable when it follows the "fruit fly"
pattern: like how fruit fly genetics guided the Human Genome
Project.

**Small scale as ADVANTAGE (do this):**

- Test fundamental principles / mechanisms
- Develop new methods that later scale (dropout, Adam, GANs,
  batch norm all started at small scale)
- Study training dynamics (loss landscapes, gradient
  distributions, feature formation)
- Rapid iteration for understanding
- Mechanistic / interpretability research
- Educational value (nanoGPT's lasting impact was pedagogical)

**Small scale as LIMITATION (don't do this):**

- Predict large-scale performance from small experiments
- Architecture horse-races ("which is best at 3M params?")
- Scaling law characterization below ~100M params (noise
  dominates, results unreliable beyond ~2 OOM extrapolation)
- Benchmarking papers (trivially predictable, not research)

---

## Expert Advice on Research Taste

**Hamming (You and Your Research):** Work on IMPORTANT problems.
The test isn't "is there a gap?" but "does anyone care about the
answer?" Most scientists fail because they work on safe,
incremental problems instead of important ones.

**Schulman (Opinionated Guide to ML Research):** Work on
DIFFERENT problems from the community. His focus on policy
gradients (when everyone else did Q-learning) led to TRPO, GAE,
PPO. Keep a daily research notebook.

**Nielsen (Principles of Effective Research):** Internalize a
strong vision. Work on what you find genuinely interesting, not
what gap analysis tells you to do.

**Olah (Research Taste Exercises):** Testing whether an idea is
good is expensive (months). Develop taste by: having mentors
rate your ideas, studying history of science, critically examining
community taste.

**Greydanus (Scaling Down):** Small-scale research is
complementary to large-scale. The question is always "does this
scale up?" Focus on ideas, not benchmark numbers.

---

## Hypothesis Quality Checklist

Before pre-registering a hypothesis, pass these gates:

### Gate 1: Is the question important?
- Would the answer change how people build or train models?
- Would a reviewer say "so what?" to any possible outcome?
- Is this a FUNDAMENTAL principle or just a parameter sweep?

### Gate 2: Is small scale appropriate?
- Does small scale give you an ADVANTAGE (speed, iteration,
  mechanistic access)?
- Or is it a LIMITATION (noise > signal, won't transfer)?
- Greydanus test: is this "fruit fly genetics" or "toy car
  racing"?

### Gate 3: Has someone already answered this?
- Search hard. Check arXiv, Google Scholar, Semantic Scholar.
- Check if the answer is implied by existing theory (e.g.,
  "architecture doesn't matter" from Kaplan 2020).
- Check if your OWN prior experiments already answer it.

### Gate 4: Is the answer trivially predictable?
- Could a knowledgeable reviewer predict the outcome without
  running the experiment?
- "Bandwidth-efficient architectures are faster on
  bandwidth-limited hardware" — obvious, not research.
- "μP works for any architecture with width as a parameter" —
  trivially yes from the theory.

### Gate 5: Is the methodology sound at your scale?
- Can you get statistical significance with affordable seeds?
- Is your dataset large enough to avoid overfitting artifacts?
- Are you in a regime where your metrics are meaningful?
  (3M params on Shakespeare is NOT a regime where architecture
  comparisons are meaningful — we proved this in 4 rounds.)

### Gate 6: What's the worst-case outcome?
- If EVERY hypothesis is falsified, did you learn something
  useful? (HYP-001c: yes — learned about overfitting.)
- If the null is true, is that still publishable/interesting?

---

## Calibration Lessons from HYP-001 Series

### What went wrong (4 rounds, diminishing returns):

1. **Wrong question for the scale.** "Which architecture is
   best?" is a question that only matters at scale (>100M).
   At 3M params, the answer is always "they're all about the
   same" (Kaplan 2020, Narang 2021).

2. **Poorly calibrated priors.** B-006 started at 0.70 when
   literature-informed prior should have been ~0.40. Over-
   confident priors led to surprise when results were negative.

3. **Wrong primary metric.** Three rounds used training loss
   before discovering it masked overfitting (train-val gap
   ~0.85). DEC-008 (val loss as primary) came too late.

4. **Methodology bugs compounded.** No val split (3 rounds),
   parameter mismatch (d_ff for SwiGLU), fixed LR across
   architectures. Each bug required another round to fix.

5. **Sunk cost fallacy.** After HYP-001 was inconclusive, the
   instinct was "run it again better" (HYP-001b, 001c, 001d)
   instead of "ask a different question."

### What went right:

1. **Infrastructure was valuable.** FLOP counting, MLflow, val
   splits, μP — all built during HYP-001 series, all reusable.

2. **One genuinely novel finding emerged.** The dropout ×
   normalization interaction (ANOM-009/010/011) is not in the
   literature.

3. **Methodology improved.** DEC-004 through DEC-008 were all
   learned the hard way but are now permanent improvements.

### Rules going forward:

- **Max 2 rounds on the same question.** If 2 rounds don't
  produce clear signal, the question is wrong for the scale.
- **Val loss from day 1.** Never pre-register with train loss.
- **Lit review before AND after.** Would have caught the dropout
  and overfitting literature earlier.
- **Question the question.** Before round 2, ask: "Am I refining
  methodology or am I in a sunk cost loop?"

---

## Sources

- Hamming, "You and Your Research" (1986 talk)
- Schulman, "An Opinionated Guide to ML Research" (blog)
- Nielsen, "Principles of Effective Research" (blog)
- Olah, "Research Taste Exercises" (colah.github.io)
- Greydanus, "Scaling Down Deep Learning" (2020 blog)
- Narang et al. 2021, EMNLP (LIT-001)
- Kaplan et al. 2020 (LIT-005)
- "Position: Embracing Negative Results in ML" (arXiv 2024)
- "Scaling Laws Are Unreliable for Downstream Tasks" (2025)
