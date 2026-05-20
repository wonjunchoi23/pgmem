# MemoryBank Batch — LoComo

`memorybank/`의 배치 실험 variant.

동일한 실험 결과를 내면서, 여러 sample의 동일 단계 LLM call을 묶어 한 번에 실행하도록 구현되었다.

---

## memorybank와의 차이

### 배치화 단위와 구조

MemoryBank의 Phase 1 LLM call은 **turn 단위가 아닌 session 단위**로 발생한다.
- turn 저장 (`add_memory`): embedding only, LLM 없음
- session 종료 후: session event 요약 + session personality 요약 (2 LLM calls)
- 모든 session 종료 후: global event 합성 + global personality 합성 (2 LLM calls)

따라서 배치 축은 **session_idx** 기준이다:

```
for session_idx = 0, 1, ..., max_sessions:
  active samples의 해당 session 턴 embedding 저장 (LLM 없음)
  ↓ batch LLM
  session event 요약 (active samples 동시)
  ↓ batch LLM
  session personality 요약 (active samples 동시)

Phase 1 End:
  ↓ batch LLM
  global event 합성 (전체 samples 동시)
  ↓ batch LLM
  global personality 합성 (전체 samples 동시)
  forgetting 적용 (LLM 없음)

Phase 2:
  QA 프롬프트 전체 수집 → temperature 기준 그룹 → QA_BATCH_SIZE chunk 단위 batch LLM
```

### 파일별 변경 내용

| 파일 | 변경 내용 |
|------|-----------|
| `config_0.py` | `BATCH_SIZE`, `QA_BATCH_SIZE` 추가 |
| `retriever.py` | `__init__`이 model name 또는 이미 로드된 `SentenceTransformer` 인스턴스 수용 |
| `memory_bank.py` | `build_session_prompts`, `apply_session_event_result`, `apply_session_personality_result`, `build_global_prompts`, `apply_global_event_result`, `apply_global_personality_result`, `accumulate_token_counts` 추가 |
| `agent.py` | `build_qa_prompt`, `accumulate_summary_tokens` 추가; `__init__`에 `embedding_model` 파라미터 추가; `answer_qa`에 `choice_order_seed` 파라미터 추가 |
| `run_experiment.py` | `BatchedMemoryBankRunner`로 전면 재작성 |
| `load_dataset.py` | 변경 없음 (복사) |
| `merge_results.py` | 변경 없음 (복사) |

### 새로운 API (memory_bank.py)

기존 `summarize_session()`, `synthesize_global()`, `_call_llm_for_summary()`는 그대로 유지된다 (sequential fallback 호환).

배치 runner가 사용하는 추가 API:

```python
# 프롬프트 빌더 (LLM call 없음)
build_session_prompts(session_id, date_str, dialogue_text, speaker_a, speaker_b)
    → (event_prompt, personality_prompt)

build_global_prompts()
    → (event_prompt, personality_prompt) | None

# 결과 적용 (LLM call 없음)
apply_session_event_result(session_id, date_str, event_text)
apply_session_personality_result(session_id, personality_text)
apply_global_event_result(event_text)
apply_global_personality_result(personality_text)

# 수동 token 누적 (batch call은 자동 카운팅 우회)
accumulate_token_counts(input_tokens, output_tokens, api_calls)
```

### Checkpoint 포맷 변경

| | sequential (`memorybank`) | batch (`memorybank_batch`) |
|--|--|--|
| 포맷 | `{"last_completed_sample_index": 17}` | `{"completed_sample_ids": ["id1", ...]}` |
| 이유 | 순차 처리 → index로 충분 | batch 내 일부만 성공할 수 있음 |

구 포맷도 로드 시 자동 변환 지원.

### Category 5 deterministic ordering

sequential 코드는 `random.shuffle()`을 사용하지만, batch 실행에서는 실행 순서 변화가 RNG 소비 순서에 영향을 줄 수 있다.

`build_qa_prompt()`와 `answer_qa()` 모두 `choice_order_seed` 파라미터를 받으며, seed가 주어지면 `hashlib.md5` 기반 결정론적 ordering을 사용한다:

```python
choice_seed = f"{sample_id}::{qa_idx}::{question}"
prompt, temperature = agent.build_qa_prompt(..., choice_order_seed=choice_seed)
```

---

## 사용법

```bash
python run_experiment.py \
  --start-sample 0 \
  --end-sample 9 \
  --batch-size 4 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel 1 \
  --gpu-memory 0.5 \
  --config config_0
```

결과 병합:

```bash
python merge_results.py config_0_outputs_Llama-3.1-8B-Instruct
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--batch-size` | `BATCH_SIZE` (config) | 한 번에 병렬 처리할 sample 수 |
| `--start-sample` | - | 시작 sample 인덱스 (포함) |
| `--end-sample` | - | 끝 sample 인덱스 (포함) |
| `--model` | config 기본값 | vLLM 모델 경로 |
| `--tensor-parallel` | config 기본값 | tensor parallel 크기 |
| `--gpu-memory` | config 기본값 | GPU 메모리 사용률 (0~1) |

`config_0.py`에서 조정 가능한 batch 관련 설정:

```python
BATCH_SIZE = 4      # sample 단위 병렬 수 (session summary batch 크기와 동일)
QA_BATCH_SIZE = 64  # QA 한 번의 batch LLM call에 넣는 프롬프트 수
```

---

## 출력 구조

```
memorybank_batch/
└── config_0_outputs_{model_name}/
    └── sample_{start}_{end}/
        ├── results_{model}_sample_{start}_{end}.json
        ├── checkpoint_{model}_sample_{start}_{end}.json
        ├── logs/
        ├── retrieval_logs/
        │   └── sample_{id}_retrieval_log.jsonl
        ├── memory_snapshots/
        │   └── sample_{id}/
        │       ├── entries.json
        │       ├── summaries.json
        │       ├── metadata.json
        │       ├── embeddings.npy
        │       └── corpus.pkl
        └── prompt_log/
            └── sample_{id}/
                ├── call_1_session_event/calls.jsonl
                ├── call_2_session_personality/calls.jsonl
                ├── call_3_global_event/calls.jsonl
                ├── call_4_global_personality/calls.jsonl
                └── call_5_qa/calls.jsonl
```

### `retrieval_logs/sample_{id}_retrieval_log.jsonl`

One JSON object per line, one entry per QA question (Phase 2 only).

| Field | Description |
|---|---|
| `phase` | `"qa"` for all entries |
| `query` | The QA question text used as retrieval query |
| `memory_type` | Parallel labels for `num_retrieved` (`["dialogue_memory", "session_summary"]`) |
| `retrieved_items` | Top-k memories: content preview, cosine score, `dia_id`, `memory_type` |
| `num_retrieved` | Per-type counts of retrieved dialogue memories and retrieved session summaries |
| `module_specific.total_memories` | Total memory entries in store at retrieval time |
| `module_specific.event_summary_length` | Character count of global event summary |
| `module_specific.user_portrait_length` | Character count of global personality portrait |
| `module_specific.prompt_context_type` | Full QA-context labels including always-appended global summary / portrait |
| `module_specific.num_prompt_context` | Parallel counts for the final QA context composition |
