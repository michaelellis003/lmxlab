# Models

Language model base class and architecture config factories.

## LanguageModel

::: lmxlab.models.base.LanguageModel
    options:
      members: true

## Generation

::: lmxlab.models.generate

## Config Factories

Each factory returns a `ModelConfig` that builds the corresponding
architecture when passed to `LanguageModel`.

### GPT

::: lmxlab.models.gpt.gpt_config

::: lmxlab.models.gpt.gpt_tiny

### LLaMA

::: lmxlab.models.llama.llama_config

::: lmxlab.models.llama.llama_tiny

### Gemma

::: lmxlab.models.gemma.gemma_config

::: lmxlab.models.gemma.gemma_tiny

### Gemma 3 (Sliding Window)

::: lmxlab.models.gemma3.gemma3_config

::: lmxlab.models.gemma3.gemma3_tiny

### DeepSeek V2 (MLA)

::: lmxlab.models.deepseek.deepseek_config

::: lmxlab.models.deepseek.deepseek_tiny

### DeepSeek V3 (MLA + MoE)

::: lmxlab.models.deepseek.deepseek_v3_config

::: lmxlab.models.deepseek.deepseek_v3_tiny

### Mixtral (MoE)

::: lmxlab.models.mixtral.mixtral_config

::: lmxlab.models.mixtral.mixtral_tiny

### Llama 4 (iRoPE + Chunked Attention + MoE)

::: lmxlab.models.llama4.llama4_scout_config

::: lmxlab.models.llama4.llama4_scout_tiny

### Nemotron (Mamba-2 + LatentMoE + Attention)

::: lmxlab.models.nemotron.nemotron3_config

::: lmxlab.models.nemotron.nemotron3_tiny

### Falcon H1 (Mamba-2 Hybrid)

::: lmxlab.models.falcon.falcon_h1_config

::: lmxlab.models.falcon.falcon_h1_tiny

### Jamba (Mamba-2 + MoE)

::: lmxlab.models.jamba.jamba_config

::: lmxlab.models.jamba.jamba_tiny

### Qwen 3.5 (DeltaNet Hybrid)

::: lmxlab.models.qwen35.qwen35_config

::: lmxlab.models.qwen35.qwen35_tiny

## Weight Conversion

::: lmxlab.models.convert.load_from_hf

::: lmxlab.models.convert.config_from_hf

::: lmxlab.models.convert.convert_weights
