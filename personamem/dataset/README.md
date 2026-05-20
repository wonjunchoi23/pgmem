---
license: mit
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- personalization
- preference
- chatbot
- memory
- long-context
- alignment
- persona
- benchmark
pretty_name: PersonaMem
size_categories:
- 100M<n<1B
configs:
- config_name: benchmark
  data_files:
  - split: 32k
    path: questions_32k.csv
  - split: 128k
    path: questions_128k.csv
  - split: 1M
    path: questions_1M.csv
---

**🚨 We invite everyone to checkout our PersonaMem-v2 on [🤗HuggingFace](https://huggingface.co/datasets/bowen-upenn/ImplicitPersona), focusing on realistic and implicit user preferences in long conversations!**
 
This is the official Huggingface repository of the paper **Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale** and the **PersonaMem benchmark**.

We present PersonaMem, a new **LLM personalization benchmark** to assess how well language models can infer evolving user profiles and generate personalized responses across task scenarios. PersonaMem emphasizes persona-oriented, multi-session interactions between users and chatbots, facilitated by a synthetic dialog generation pipeline that simulates realistic and evolving conversational contexts. **For instructions on using the data and running inference, please refer to our Github repository at https://github.com/bowen-upenn/PersonaMem/.**

Different users have different personas. Personalization in LLMs involves adapting model responses to individual users based on their traits, preferences, and interaction history. By analyzing previous interactions, LLMs learn to deliver more relevant and tailored responses to different users, rather than merely providing generic correct answers. As a result, personalization enhances the model’s effectiveness in various tasks such as writing assistance, recommendations, or consultations, and thereby user experience and engagement.

<img src="https://cdn-uploads.huggingface.co/production/uploads/6686e3c8978a88805740efec/YxWGFOLyvxeo2FeQkMaOK.png" width="800"/>

As shown in the overview, each benchmark sample is a user persona with static (e.g., demographic info.) and dynamic attributes (e.g., evolving preferences). Users engage with a chatbot in multi-session interactions across a variety of topics such as food recommendation, travel planning, and therapy consultation. As the user’s preferences evolve over time, the benchmark offers annotated questions assessing whether models can track and incorporate the changes into their responses.

# 📊 Benchmark Data
**The dataset is available with three versions based on context token length**

  - 32k tokens
    - questions_32k.csv
    - shared_contexts_32k.jsonl
  - 128k tokens
    - questions_128k.csv
    - shared_contexts_128k.jsonl
  - 1M tokens
    - questions_1M.csv
    - shared_contexts_1M.jsonl

**File Format**

Each questions_[SIZE].csv file contains the following columns:

- persona_id: Unique ID for each user persona
- question_id: Unique ID for each question
- question_type: We provide 7 fine-grained in-situ question types defined in the figure below.
- topic: Topic of the conversation session
- context_length_in_tokens: Total tokens in the context
- context_length_in_letters: Total English letters in the context
- distance_to_ref_in_blocks: Blocks from question to most recent preference mention
- distance_to_ref_in_tokens: Tokens from question to most recent preference mention
- num_irrelevant_tokens: Tokens from irrelevant interactions
- distance_to_ref_proportion_in_context: Proportional position of latest preference in context
- user_question_or_message
- correct_answer
- all_options: list of all answer choices presented for this question
- shared_context_id: Key to retrieve full context from shared_contexts_[SIZE].jsonl
- end_index_in_shared_context: Use to slice the loaded context as context[:int(end_index_in_shared_context)]

Each shared_contexts_[SIZE].jsonl file is a JSONL-formatted list of API dicts of user–model interaction sequences.

**Types of Skills Evaluated**

To evaluate LLMs' ability to (1) memorize the user profile, (2) track how the user profile evolve over time, and (3) generate personalized responses accordingly in new scenarios, we design the following 7 types of in-situ user queries in the PersonaMem benchmark.

- **Recall user-shared facts:** We evaluate whether a personalized chatbot can recall static events, activities, or interests the user has shared in previous interactions, and incorporate the information in its responses.
- **Suggest new ideas:** We evaluate whether a chatbot can suggest new items or activities that have not been mentioned in the interaction history, when users explicitly request so, e.g., suggest new restaurants I haven't ordered from before.
- **Acknowledge latest user preferences:** We evaluate whether a chatbot can recognize the latest preference expressed by the user in the interaction history.
- **Track full preference evolution:** We evaluate whether a chatbot can keep track of how users' preferences shift by time.
- **Revisit reasons behind preference updates:** We evaluate whether a chatbot can recall the reason(s) or event(s) leading to the preference change from a user.  
- **Provide preference-aligned recommendations:** We test whether a chatbot can proactively offer new recommendations that aligns with the user's current preferences.
- **Generalize to new scenarios:** We evaluate whether a chatbot can transfer what it learns about the user from other task scenarios to a new task.  

Examples:
<img src="https://cdn-uploads.huggingface.co/production/uploads/6686e3c8978a88805740efec/SBpaVjjN67B92HC6VMrAj.png" width="800"/>



**Citation**

If you find our work inspires you, please consider citing it. Thank you!

    @article{jiang2025know,
      title={Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale},
      author={Jiang, Bowen and Hao, Zhuoqun and Cho, Young-Min and Li, Bryan and Yuan, Yuan and Chen, Sihao and Ungar, Lyle and Taylor, Camillo J and Roth, Dan},
      journal={arXiv preprint arXiv:2504.14225},
      year={2025}
    }