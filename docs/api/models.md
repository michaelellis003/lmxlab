# Models

Language model base class and architecture config factories.

## LanguageModel

::: lmt_metal.models.base.LanguageModel
    options:
      members: true

## Generation

::: lmt_metal.models.generate

## Config Factories

Each factory returns a `ModelConfig` that builds the corresponding
architecture when passed to `LanguageModel`.

### GPT

::: lmt_metal.models.gpt.gpt_config

::: lmt_metal.models.gpt.gpt_tiny

### LLaMA

::: lmt_metal.models.llama.llama_config

::: lmt_metal.models.llama.llama_tiny

### Gemma

::: lmt_metal.models.gemma.gemma_config

::: lmt_metal.models.gemma.gemma_tiny

### Gemma 3

::: lmt_metal.models.gemma3.gemma3_config

::: lmt_metal.models.gemma3.gemma3_tiny

### Qwen

::: lmt_metal.models.qwen.qwen_config

::: lmt_metal.models.qwen.qwen_tiny

### Mixtral

::: lmt_metal.models.mixtral.mixtral_config

::: lmt_metal.models.mixtral.mixtral_tiny

### DeepSeek

::: lmt_metal.models.deepseek.deepseek_config

::: lmt_metal.models.deepseek.deepseek_tiny
