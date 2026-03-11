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

::: lmxlab.models.gpt.gpt_small

::: lmxlab.models.gpt.gpt_medium

### LLaMA

::: lmxlab.models.llama.llama_config

::: lmxlab.models.llama.llama_tiny

::: lmxlab.models.llama.llama_7b

::: lmxlab.models.llama.llama_13b

### Gemma

::: lmxlab.models.gemma.gemma_config

::: lmxlab.models.gemma.gemma_tiny

### Gemma 3

::: lmxlab.models.gemma3.gemma3_config

::: lmxlab.models.gemma3.gemma3_tiny

### Qwen

::: lmxlab.models.qwen.qwen_config

::: lmxlab.models.qwen.qwen_tiny

### Mixtral

::: lmxlab.models.mixtral.mixtral_config

::: lmxlab.models.mixtral.mixtral_tiny

### Qwen 3.5 (Hybrid DeltaNet)

::: lmxlab.models.qwen35.qwen35_config

::: lmxlab.models.qwen35.qwen35_tiny

### DeepSeek

::: lmxlab.models.deepseek.deepseek_config

::: lmxlab.models.deepseek.deepseek_tiny

## Weight Conversion

::: lmxlab.models.convert.load_from_hf

::: lmxlab.models.convert.config_from_hf

::: lmxlab.models.convert.convert_weights
