# Cross-Reference Validation Registry

Each implementation is compared against established references.
Status: `validated` | `fixed` | `flagged` | `pending`.

## Validated Implementations

### GQA (Grouped-Query Attention)
- **File:** `src/lmxlab/core/attention.py:GQA`
- **References:** Ainslie et al. (2023), HuggingFace transformers
- **Validation:** MLX `mx.fast.scaled_dot_product_attention`
  handles KV head broadcasting natively. No explicit repeat
  needed. Verified via behavioral tests.
- **Tests:** `tests/test_cross_reference.py::test_gqa_kv_broadcast`
- **Status:** validated

### SwiGLU / GatedFFN
- **File:** `src/lmxlab/core/ffn.py:GatedFFN`
- **References:** Shazeer (2020), LLaMA (Touvron 2023),
  HuggingFace LlamaForCausalLM
- **Formula:** `down(SiLU(gate(x)) * up(x))`
- **Verification:** Matches HF, nanoGPT, MLX-examples exactly.
- **Tests:** `tests/test_cross_reference.py::test_swiglu_formula`
- **Status:** validated

### FLOP Estimation
- **File:** `src/lmxlab/experiments/flops.py`
- **References:** Megatron-LM (Narayanan 2021), Chinchilla
  (Hoffmann 2022), PaLM (Chowdhery 2022)
- **Method:** Per-token amortized counting (total FLOPs / seq).
  Attention: `2 * h * seq * hd` per token = `2 * h * seq^2 * hd`
  total, divided by `seq`. This is correct.
- **6ND test:** Within 6% of `6 * N * D` approximation.
- **Tests:** `tests/test_flops.py::test_6nd_approximation`
- **Status:** validated

### Causal Mask
- **File:** `src/lmxlab/models/base.py:_create_causal_mask`
- **References:** Vaswani (2017), PyTorch, HuggingFace
- **Note:** Uses `-1e9` instead of `-inf`. Pragmatic choice
  for numerical stability. Docstring updated to match.
- **Tests:** `tests/test_cross_reference.py::test_causal_mask`
- **Status:** validated

### Attention Scaling (SP and muP)
- **File:** `src/lmxlab/core/attention.py` (MHA, GQA, SWGQA)
- **References:** Vaswani (2017) for SP, Yang et al. (2022)
  for muP, Microsoft mup library
- **SP:** `1/sqrt(d_head)` — matches PyTorch default
- **muP:** `1/d_head` — matches Microsoft mup, Cerebras
- **Tests:** `tests/test_mup.py::TestMupAttentionScaleReference`
- **Status:** validated

### muP Weight Initialization
- **File:** `src/lmxlab/models/base.py:_apply_mup_init`
- **References:** Yang et al. (2022) Table 8, Microsoft mup,
  Cerebras modelzoo
- **Method:** Hidden weight matrices scaled by `1/sqrt(m)`.
  Embedding weights unchanged. Output head scaled by `1/sqrt(m)`.
- **Tests:** `tests/test_mup.py::TestMupWeightInit`
- **Status:** validated (added after cross-reference found gap)

### muP Logit Scaling
- **File:** `src/lmxlab/models/base.py:LanguageModel.__call__`
- **References:** Yang et al. (2022), Microsoft mup
- **Method:** `logits / width_mult` when muP enabled.
- **Tests:** `tests/test_mup.py::TestMupLogitScaling`
- **Status:** validated

### muP Optimizer (Per-Layer LR)
- **File:** `src/lmxlab/training/optimizers.py:create_mup_optimizer`
- **References:** Yang et al. (2022), Microsoft mup, Cerebras
- **Method:** Embed LR = base_lr, Hidden LR = base_lr / width_mult.
  Uses MLX MultiOptimizer with embed/hidden filter groups.
- **Tests:** `tests/test_mup.py::TestMupOptimizer`
- **Status:** validated

### Knowledge Distillation
- **File:** `src/lmxlab/training/distillation.py`
- **References:** Hinton et al. (2015), HuggingFace transformers,
  PyTorch KLDivLoss documentation
- **KL Direction:** KL(teacher || student). Correct per Hinton 2015.
  Formula: `sum(teacher_probs * (teacher_log_probs - student_log_probs))`.
- **T^2 Scaling:** Applied only to KL term: `alpha * KL * T^2 +
  (1-alpha) * CE`. Correct per Hinton 2015. The T^2 compensates
  for gradient magnitude reduction from temperature scaling
  (gradients scale as 1/T^2, so multiply by T^2 to maintain scale).
- **Alpha Convention:** Alpha weights soft targets (KL), 1-alpha
  weights hard targets (CE). Matches Hinton 2015 and HF convention.
- **Tests:** `tests/test_cross_reference.py::TestDistillationCrossReference`
- **Status:** validated

### LoRA (Low-Rank Adaptation)
- **File:** `src/lmxlab/core/lora.py`
- **References:** Hu et al. (2021), HuggingFace PEFT layer.py
- **Init A:** Code uses `normal * sqrt(2/fan_in)` = **Kaiming normal**.
  Comment says "Kaiming uniform" (cosmetic inaccuracy). Both are
  valid (HF PEFT uses kaiming_uniform by default, original LoRA
  paper uses Gaussian). Implementation is sound.
- **Init B:** Zero initialization. Correct per Hu et al. (2021).
  Ensures BA = 0 at init for stable start from pretrained weights.
- **Scaling:** `alpha / rank`. Correct per Hu et al. (2021) and HF PEFT.
- **Merge Formula:** `W + (A @ B)^T * scaling`. Correct. Forward is
  `x @ W^T + x @ A @ B * s`, so merged is `W + (AB)^T * s`.
- **Tests:** `tests/test_cross_reference.py::TestLoRACrossReference`
- **Status:** validated (comment fixed: "Kaiming uniform" -> "normal")

### Sampling (Top-p and Repetition Penalty)
- **File:** `src/lmxlab/models/generate.py`
- **References:** Holtzman et al. (2019), Keskar et al. (2019),
  HuggingFace transformers logits_process.py
- **Top-p Threshold:** Mask is `cumulative - sorted_probs > top_p`.
  Keeps tokens until cumulative exceeds p, **including** the token
  that crosses the boundary. Correct per Holtzman 2019, HF
  transformers, and thomwolf gist.
- **Top-p Renormalization:** After masking, probabilities are
  renormalized and sampled via categorical. Correct.
- **Top-p Edge Case:** Adding 1e-10 before log in categorical
  prevents issues if sorted_probs has exact zeros after masking.
- **Repetition Penalty:** Positive logits **divided**, negative
  logits **multiplied** by penalty. Correct per Keskar et al. (2019)
  CTRL paper and HF issue #2302. Rationale: dividing negative
  logits would increase probability (wrong direction).
- **Tests:** `tests/test_cross_reference.py::TestSamplingCrossReference`
- **Status:** validated

### MLA (Multi-Head Latent Attention)
- **File:** `src/lmxlab/core/mla.py`
- **References:** DeepSeek-V2 (arXiv:2405.04434), DeepSeek-V3
- **KV Compression:** Down-project x -> (c_kv, k_pe), cache the
  compressed latent. Up-project c_kv -> K_nope, V at attention time.
  Total cache = kv_lora_rank + rope_dim per token (vs
  2*n_heads*head_dim for MHA). Correct.
- **Decoupled RoPE:** RoPE applied only to q_pe and k_pe dimensions.
  Shared single-head k_pe (MQA-style broadcast). Correct per DSV2
  equations 17-21.
- **Q/K Dimension Ordering:** Our code uses `[pe, nope]` for both Q
  and K. DeepSeek-V2 paper uses `[nope, pe]`. This is a **convention
  difference**, not a bug — both Q and K use the same ordering, so
  the dot product aligns correctly.
- **Tests:** `tests/test_cross_reference.py::TestMLACrossReference`
- **Status:** validated (convention difference documented)

### DPO (Direct Preference Optimization)
- **File:** `src/lmxlab/training/dpo.py`
- **References:** Rafailov et al. (2023) arXiv:2305.18290,
  HuggingFace TRL DPOTrainer, eric-mitchell/DPO
- **Loss Formula:** `logaddexp(0, -(chosen_rewards - rejected_rewards))`
  = `softplus(-(r_w - r_l))` = `-logsigmoid(r_w - r_l)`.
  Matches paper equation 7. Numerically stable via logaddexp.
- **Log Probs:** Sum over positions (not mean). Correct per paper
  and all reference implementations.
- **Beta Default:** 0.1, matches TRL, TorchTune, literature.
- **Sign Convention:** Correct — chosen rewards > rejected reduces loss.
- **Tests:** `tests/test_cross_reference.py::TestDPOCrossReference`
- **Status:** validated (label smoothing is optional enhancement)

## Fixed Implementations

### Position Encoding Application
- **File:** `src/lmxlab/core/block.py`, `src/lmxlab/models/base.py`
- **Bug:** Position encoding modules (RoPE, sinusoidal, ALiBi)
  were created per-block but never called in the forward pass.
  Models trained without any position information.
- **Fix:** RoPE passed to attention modules for Q/K rotation.
  Sinusoidal applied at model level after embedding.
- **References:** Vaswani (2017) for sinusoidal, Su et al. (2021)
  for RoPE, HuggingFace transformers, MLX-examples
- **Tests:** `tests/test_cross_reference.py::test_rope_applied`,
  `tests/test_cross_reference.py::test_sinusoidal_applied`
- **Status:** fixed

### Causal Mask Docstring
- **File:** `src/lmxlab/models/base.py:_create_causal_mask`
- **Bug:** Docstring said "-inf" but code uses -1e9.
- **Fix:** Updated docstring to say "-1e9".
- **Status:** fixed

### GatedDeltaNet Output Gate
- **File:** `src/lmxlab/core/deltanet.py`
- **References:** Yang et al. "Gated Delta Networks" ICLR 2025,
  flash-linear-attention reference implementation
- **Bug:** Output gate used `mx.sigmoid` instead of `nn.silu`.
  The paper and reference implementations use SiLU (swish) for
  the output gate, not sigmoid.
- **Fix:** Changed `mx.sigmoid(self.out_gate_proj(x))` to
  `nn.silu(self.out_gate_proj(x))`.
- **Validated parts:** Delta rule formula, gate initialization
  (bias=-3), L2 normalization of Q/K, constant-size state matrix.
- **Tests:** `tests/test_cross_reference.py::TestGatedDeltaNetCrossReference`
- **Status:** fixed

### MoE Top-K Routing
- **File:** `src/lmxlab/core/moe.py` (MoEFFN, SharedExpertMoEFFN)
- **References:** Jiang et al. "Mixtral" (2024), DeepSeek-V3,
  Shazeer et al. "Outrageously Large Neural Networks" (2017)
- **Bug:** Softmax was applied over all experts, then re-gathered
  for top-k. Should softmax over top-k logits only (Mixtral
  convention). This made gating weights not sum to 1.
- **Fix:** Select top-k indices first, gather those logits, then
  softmax. For SharedExpertMoEFFN, biased logits for selection but
  un-biased logits for weight computation (DSV3 convention).
- **Tests:** `tests/test_cross_reference.py::TestMoECrossReference`
- **Status:** fixed

### LoRA Init Comment
- **File:** `src/lmxlab/core/lora.py` line 83
- **Bug:** Comment said "Kaiming uniform init" but code uses
  `normal * sqrt(2/fan_in)` = Kaiming **normal**.
- **Fix:** Updated comment to say "Kaiming normal init".
- **Status:** fixed

### ALiBi Wiring and Wrapper
- **File:** `src/lmxlab/core/position.py`, `src/lmxlab/core/block.py`
- **References:** Press et al. "Train Short, Test Long" ICLR 2022,
  HuggingFace BLOOM `build_alibi_tensor`
- **Bugs (2):**
  1. ALiBi wrapper called `nn.ALiBi` with mask as first arg
     (attention_scores), not as the `mask` kwarg. Wrong API usage.
  2. ALiBi was never applied in the block's forward pass. The
     position module was created but never called.
- **Fix:** Rewrote `ALiBi.__call__` to create a dummy scores
  tensor and pass mask/offset correctly. Added `self._alibi` in
  `ConfigurableBlock.__init__` and apply it to the mask before
  attention in both pre/post-norm paths.
- **Validated:** Slopes follow geometric sequence `2^(-8h/H)`,
  bias increases with distance, output differs vs no-position.
- **Tests:** `tests/test_cross_reference.py::TestALiBiCrossReference`
- **Status:** fixed

### Mamba-2 SSD
- **File:** `src/lmxlab/core/mamba2.py`
- **References:** Dao & Gu (2024) arXiv:2405.21060,
  state-spaces/mamba mamba2.py, HF transformers modeling_mamba2.py,
  nvidia/Nemotron-H modeling_nemotron_h.py
- **Validation:** Cross-referenced split order (z, xBC, dt), conv1d
  applied to xBC only (not gate z), discretization formula
  (dA=exp(A*dt), dB=dt*B), state update (dA*S + x outer dB),
  output (S@C + D*x), dt_bias via softplus, A_log init as
  log(arange(1..n_heads)), gate-before-norm ordering
  (RMSNorm(SiLU(z)*y) with norm_before_gate=False).
- **Fixed during xref:** dt_bias missing, norm/gate order reversed,
  A_log initialization wrong (was zeros-1, now log(arange)).
- **Known simplification:** n_groups=1 (reference supports multi-group
  B,C sharing). Recurrent form only (no chunkwise SSD dual).
- **Tests:** `tests/test_cross_reference.py::TestMamba2CrossReference`
- **Status:** validated

### Squared ReLU FFN (Primer)
- **File:** `src/lmxlab/core/ffn.py` (ReluSquaredFFN)
- **References:** So et al. (2021) "Primer" NeurIPS,
  nvidia/Nemotron-H NemotronHMLP with mlp_hidden_act="relu2"
- **Validation:** relu2(x) = max(0,x)^2 confirmed. Non-gated
  2-layer FFN: down(relu2(up(x))). Distinct from GatedReluSquaredFFN
  which is a SwiGLU-style 3-layer variant.
- **Fixed during xref:** Nemotron-H config was using gated_relu2
  (3-layer) instead of relu2 (2-layer) for all FFNs.
- **Tests:** `tests/test_cross_reference.py::TestReluSquaredCrossReference`
- **Status:** validated

### LatentMoE (Nemotron-H)
- **File:** `src/lmxlab/core/moe.py` (LatentMoEFFN)
- **References:** nvidia/Nemotron-H-8B modeling_nemotron_h.py,
  LatentMoE paper arXiv:2601.18089
- **Validation:** Router operates on full hidden dim (not latent).
  Expert FFNs are non-gated relu2 (2 projections) in latent space.
  Shared expert is non-gated relu2 at full dimension. Sigmoid
  routing with normalization (not softmax).
- **Fixed during xref:** Router was routing from latent dim (should
  be full hidden dim). Expert FFNs were gated 3-layer (should be
  non-gated 2-layer). Router was using softmax (should be sigmoid
  + normalize).
- **Known simplification:** No grouped expert selection, no
  score_correction_bias, no routed_scaling_factor.
- **Tests:** `tests/test_cross_reference.py::TestLatentMoECrossReference`
- **Status:** validated

### Multi-Token Prediction (MTP)
- **File:** `src/lmxlab/training/mtp.py`
- **References:** DeepSeek-V3 (arXiv:2412.19437), Meta
  (arXiv:2404.19737)
- **Validation:** MTPHead architecture (concat norm(h)+norm(e)
  -> project -> block), sequential chaining (h=h_mtp), target
  alignment (k=1,2 verified), loss formula (main + lambda *
  mean), shared lm_head, RMSNorm for both norms. All match
  DeepSeek-V3 exactly.
- **Tests:** `tests/test_cross_reference.py::TestMTPCrossReference`
- **Status:** validated

### GatedReluSquaredFFN
- **File:** `src/lmxlab/core/ffn.py`
- **References:** So et al. (2021) "Primer" NeurIPS
- **Formula:** `down(relu(gate(x))^2 * up(x))` — SwiGLU-style
  gated variant with squared ReLU instead of SiLU.
- **Tests:** `tests/test_cross_reference.py::TestGatedReluSquaredCrossReference`
- **Status:** validated

### Nemotron-H Config Factory
- **File:** `src/lmxlab/models/nemotron.py`
- **References:** nvidia/Nemotron-H-8B-Base-8K config.json
- **Validation:** 8B config matches HF: 52 layers (24M+24-+4*),
  vocab=131072, d_ff=21504, no MoE. Pattern verified character
  by character. Attention (*) layers have ffn='none' — FFN is
  in separate dense (-) layers.
- **Tests:** `tests/test_cross_reference.py::TestNemotronConfigCrossReference`
- **Status:** validated (fixed pattern, vocab, d_ff, ffn)

### Nemotron-H Weight Conversion
- **File:** `src/lmxlab/models/convert.py`
- **References:** nvidia/Nemotron-H-8B-Base-8K
  model.safetensors.index.json
- **Validation:** Verified all weight name mappings against
  actual HF safetensors index. Fixed: embedding (embeddings
  not embed_tokens), lm_head (not output_head), dense MLP
  (mixer.* not mlp.*), attention layers (no FFN weights).
  Config extraction uses flat HF fields (mamba_num_heads,
  ssm_state_size, expand, n_groups, conv_kernel).
- **Tests:** `tests/test_cross_reference.py::TestWeightConversionCrossReference`
- **Status:** validated (3 name fixes + config extraction)

### Dropout Wiring
- **File:** `src/lmxlab/core/block.py`, `src/lmxlab/models/base.py`
- **References:** GPT-2 (Radford 2019), nanoGPT (Karpathy)
- **Validation:** Residual dropout placement correct (after
  sublayer output, before residual add) in both pre-norm and
  post-norm paths. Embedding dropout correct (after lookup).
  Single instance reuse is valid.
- **Known difference:** No attention weights dropout (inside
  attention after softmax). GPT-2/nanoGPT include this.
  lmxlab only has residual dropout. Flagged, not a bug.
- **Tests:** `tests/test_cross_reference.py::TestDropoutWiringCrossReference`
- **Status:** flagged (missing attn weights dropout)

### ModularArithmeticDataset
- **File:** `src/lmxlab/data/modular_arithmetic.py`
- **References:** Ground truth (exhaustive arithmetic verification)
- **Validation:** All (a+b) mod p answers verified exhaustively for
  both train and test splits. Deterministic hash-based split produces
  disjoint, complete coverage. GPT-2 BPE single-token assumption
  verified for all numbers 0-96.
- **Tests:** `tests/test_modular_arithmetic.py::TestModularArithmeticDataset`,
  `tests/test_modular_arithmetic.py::TestTokenFormatCrossReference`
- **Status:** validated

### pass@k Evaluation (Fast Path)
- **File:** `src/lmxlab/eval/metrics.py:pass_at_k`,
  `recipes/hyp007_test_time_compute.py:evaluate_pass_at_k_modular`
- **References:** Chen et al. (2021) arXiv:2107.03374 (Codex paper),
  HuggingFace evaluate pass@k
- **Validation:** Unbiased estimator formula verified against
  hand-computed values from Chen et al. Fast-path single forward
  pass sampling shown equivalent to generate(max_tokens=1) by
  comparing logit distributions.
- **Tests:** `tests/test_cross_reference.py::TestPassAtKCrossReference`,
  `tests/test_modular_arithmetic.py::TestPassAtKEvaluation`
- **Status:** validated

## Pending

(None — all implementations cross-referenced.)

## Process

When implementing or modifying a method:
1. Identify 2+ reference implementations (HF, nanoGPT, MLX-examples,
   original paper)
2. Compare formulas, shapes, and behavior
3. Add a cross-reference test in `tests/test_cross_reference.py`
4. Update this file with the validation result
