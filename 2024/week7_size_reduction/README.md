# Cовременные LLM

На этой неделе изучаем способы уменьшения размеров нейронных сетей: прунинг, квантизацию и дистилляцию знаний, и выясняем, когда нужно применять одни методы, а когда – другие.


## Статьи по теме

#### Прунинг
1. Оригинальная статья 1989 (Y. LeCun): [статья](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)
1. SparseGPT (2023) – эффективный итеративный one-time прунинг для LLM [статья](https://arxiv.org/abs/2301.00774)
1. Wanda (2023) – прунинг для LLM на основе весов и активаций: [статья](https://arxiv.org/abs/2306.11695)

#### Квантизация
1. Блогпосты для погружения: [первый](https://huggingface.co/docs/optimum/concept_guides/quantization), [второй](https://github.com/google/gemmlowp/blob/master/doc/quantization.md), [третий](https://www.tensorops.ai/post/what-are-quantized-llms)
1. LLM.int8() – популярная динамическая квантизация: [статья](https://arxiv.org/abs/2208.07339)
1. GGUF – формат хранения модели в сильно сжатом виде, поддерживается библиотекой [GGML](https://github.com/ggerganov/ggml): [документация/блогпост](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
1. AQLM – квантизация до 1-2 бит: [статья](https://arxiv.org/abs/2401.06118)

#### Дистилляция знаний
1. Оригинальная статья 2015 (G. Hinton): [статья](https://arxiv.org/abs/1503.02531)
1. DistilBERT: [статья](https://arxiv.org/abs/1910.01108)
1. Обзор большого числа методов: [статья](https://arxiv.org/abs/2006.05525)
