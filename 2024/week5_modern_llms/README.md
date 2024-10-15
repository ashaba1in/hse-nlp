# Cовременные LLM

На этой неделе мы смотрим на то, чем архитектурно современные трансформерные модели отличаются от оригинального Трансформера 2017-го года, а также на особенности их обучения.



## Статьи по теме

#### Attention
1. Flash Attention: [статья](https://arxiv.org/abs/2205.14135), [hf блогпост](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention).
1. KV-кеширование: [hf блогпост](https://huggingface.co/blog/kv-cache-quantization).
1. Multi-Query Attention: [статья](https://arxiv.org/abs/1911.02150).
1. Grouped-Query Attention: [статья](https://arxiv.org/abs/2305.13245).

#### Positional encodings
1. Relative Position Encodings: [статья](https://arxiv.org/abs/1803.02155), [блогпост](https://jaketae.github.io/study/relative-positional-encoding/).
1. Rotary Position Embeddings (RoPE): [статья](https://arxiv.org/abs/2104.09864), [блогпост](https://afterhoursresearch.hashnode.dev/rope-rotary-positional-embedding), [реализация с комментариями](https://nn.labml.ai/transformers/rope/index.html).
1. Attention with Linear Biases (ALiBi): [статья](https://arxiv.org/abs/2108.12409).

#### Feed Forward Network
1. Gated Linear Unit (GLU): [статья](https://arxiv.org/abs/2002.05202).
1. Mixture of Experts (MoE): [статья](https://arxiv.org/abs/1701.06538), [hf блогпост](https://huggingface.co/blog/moe).

#### Normalization
1. Pre-normalization: [статья](https://arxiv.org/pdf/2002.04745)
1. Root Mean Square (RMS) Norm: [статья](https://arxiv.org/abs/1910.07467)
