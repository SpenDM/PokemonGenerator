# PokemonGenerator
AI generation of new pokedex entries.

Specify your desired pokemon type, and the model will output a corresponding pokedex entry! 

*TODO*: examples

# Background
We use a distilled version of the OpenAI GPT-2 model. [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) is a transformer-based language model which can generate synthetic text samples in response to being primed with an arbirary input. Network distillation is performed in a process similar to [DistilBERT](https://arxiv.org/abs/1910.01108). DistilGP2 is two times faster and 33% smaller than GPT-2, with perplexity dropping from 21.1 to 16.3 in the distilled version.

# Quick Start

*TODO*

1. Fine-tune DistilGPT-2 on 


# Prerequisites 
The repository uses the [Transformers](https://github.com/SpenDM/PokemonGenerator.git) repository. It provides state-of-the-art general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, T5, etc.) for Natural Language Understanding (NLU) and Natural Language Generation (NLG).

# References
http://jalammar.github.io/illustrated-gpt2/