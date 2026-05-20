---
dataset_info:
  features:
  - name: preference
    dtype: string
  - name: question
    dtype: string
  - name: explanation
    dtype: string
  - name: persona
    dtype: string
  - name: topic
    dtype: string
  - name: preference_type
    dtype: string
  - name: conversation
    struct:
    - name: '0'
      struct:
      - name: assistant
        dtype: string
      - name: user
        dtype: string
    - name: '1'
      struct:
      - name: assistant
        dtype: string
      - name: user
        dtype: string
    - name: '2'
      struct:
      - name: assistant
        dtype: string
      - name: user
        dtype: string
    - name: '3'
      struct:
      - name: assistant
        dtype: string
      - name: user
        dtype: string
    - name: '4'
      struct:
      - name: assistant
        dtype: string
      - name: user
        dtype: string
    - name: '5'
      struct:
      - name: assistant
        dtype: string
      - name: user
        dtype: string
    - name: '6'
      struct:
      - name: assistant
        dtype: string
      - name: user
        dtype: string
  splits:
  - name: train
    num_bytes: 4952836
    num_examples: 1000
  download_size: 2645930
  dataset_size: 4952836
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---


# PrefEval Benchmark: Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs


Welcome to the PrefEval dataset repository! 

| [Website](https://prefeval.github.io/) | [Paper](https://arxiv.org/abs/2502.09597) | [GitHub Repository](https://github.com/amazon-science/PrefEval) |

## Dataset Overview

We introduce PrefEval, a benchmark for evaluating LLMs' ability to infer, memorize and adhere to user preferences in a long-context conversational setting. The benchmark consists of three distinct preference forms, each requiring different levels of preference understanding and reasoning.

Due to their different structures and column names, we've organized them into three separate huggingface dataset repos:

- [Explicit Preferences](https://huggingface.co/datasets/siyanzhao/prefeval_explicit)
- [Implicit Choice-Based Preferences](https://huggingface.co/datasets/siyanzhao/prefeval_implicit_choice)
- [Implicit Persona-Driven Preferences](https://huggingface.co/datasets/siyanzhao/prefeval_implicit_persona)

Each preference form has 1000 data points.
## Preference Forms
1. Explicit Preference.
```
{
    "preference": [string] The user's stated preference that the LLM should follow.
    "question": [string] The user's query related to the preference, where a generic response to this question is highly likely to violate the preference.
    "explanation": [string] A 1-sentence explanation of why answering this question in a preference-following way is challenging.
}

```
2. Implicit Preference - Choice-based Conversation
```
{
    "preference": [string] The user's explicit preference that the LLM should follow.
    "question": [string] The user's query related to the preference, where a generic response to this question is highly likely to violate the preference.
    "explanation": [string] A 1-sentence explanation of why answering this question in a preference-following way is challenging.
    "implicit_query": [string] A secondary query that offers further insight into the user’s preference, where the assistant provides multiple options.
    "options": [list] A set of options that the assistant presents in response to the user's implicit query, some of which align with and others that violate the user’s implied preference.
    "conversation": {
        "query": [string] Implicit_Query,
        "assistant_options": [string] The assistant's presenting multiple options, some aligned and some misaligned with the user's preference,
        "user_selection": [string] The user's choice or rejection of certain options.
        "assistant_acknowledgment": [string] The assistant's recognition of the user’s choice.
    },
    "aligned_op": [string] The option that aligns with the user’s preference.
}
```
3. Implicit Preference - Persona-driven Conversation

```
{
    "preference": [string] The user's explicit preference that the LLM should follow.
    "question": [string] The user's query related to the preference, where a generic response to this question is highly likely to violate the preference.
    "explanation": [string] A 1-sentence explanation of why answering this question in a preference-following way is challenging.
    "persona": [string] The assigned persona guiding the conversation, e.g., "a retired postal worker enjoying his golden years.",
    "conversation": {
        "turn1": { "user": [string], "assistant": [string] },
        "turn2": { "user": [string], "assistant": [string] },
        ...,
        "turnN": { "user": [string], "assistant": [string] }
    },
}
```

## Practical Guide to Using PrefEval
Please refer to our code repo for benchmarking code.

While our benchmark enables comprehensive evaluation across multiple dimensions (including various baselines, conversation turns, preference forms and topics, and tasks), benchmarking on complete setups is computationally intensive. For practical use, we provide guidance based on available resources:

- **If an evaluator model like Claude 3 Sonnet is available**: Use the generation task for comprehensive evaluation.
- **If using a local LLM as evaluator**: Choose a subset of the benchmark or opt for our classification task, which doesn't require LLM-based evaluators but still has strong correlation with generation task performance (see paper Sec 3.4).

For initial testing, we recommend:
1. Start with a subset of topics and conversation lengths using explicit preference forms
2. Our GitHub repository includes a leaderboard comparing various LLMs on the "travel restaurant" topic at both 10 and 300 turns, assessing both short-turn and long-turn preference-following capabilities
3. With additional computational resources, use generation task evaluators for detailed error-type analysis, and test implicit preference forms to evaluate more advanced preference-following capabilities

## Citation

```
@misc{zhao2025llmsrecognizepreferencesevaluating,
      title={Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs}, 
      author={Siyan Zhao and Mingyi Hong and Yang Liu and Devamanyu Hazarika and Kaixiang Lin},
      year={2025},
      eprint={2502.09597},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.09597}, 
}
```
