# Theanine Batch — Timeline-based Memory Management

`theanine_batch`는 `exp_locomo/theanine`의 **sample-batched** 변형이다.  
핵심 목표는 sample 내부의 memory semantics는 그대로 유지하면서, 서로 독립적인
LLM 호출만 묶어서 한 번에 실행하는 것이다.

즉, 아래 원칙을 따른다.

- sample 내부의 session 순서, finalize 순서, memory mutation 순서는 유지
- sample 간에 같은 단계의 LLM 호출만 batch로 실행
- retrieval, logging, result aggregation은 sample-local 그대로 유지

---

## 무엇이 batch 되었는가

원본 `theanine`에서 LLM 병목은 4곳이었다.

1. Phase 1 date-boundary summarization
2. Phase 1 relation extraction
3. Phase 2 timeline refinement
4. Phase 2 QA generation

`theanine_batch`는 이 4단계를 모두 batch 대상으로 바꿨다.

반대로 아래는 그대로 sample별로 수행한다.

- session 순회
- turn별 retrieval
- memory graph mutation 적용
- retrieval log 저장
- results/checkpoint 저장

---

## 파일 구성

| File | Role |
|---|---|
| `run_experiment.py` | sample-batched runner |
| `theanine_module.py` | batch-friendly facade |
| `memory_graph.py` | summarize/link step 분리형 memory graph |
| `timeline.py` | path retrieval + refine prompt builder |
| `generator.py` | QA prompt builder + sequential fallback |
| `config_0.py` | batch 크기 포함 실험 설정 |
| `load_dataset.py` | 원본 `theanine/load_dataset.py` wrapper |
| `merge_results.py` | 원본 `theanine/merge_results.py` wrapper |
| `batch_difference.md` | 배치화 변경점 및 포팅 가이드 |

---

## Batch 실행 흐름

### Phase 1

Theanine은 A-MEM처럼 turn lockstep으로 memory를 쓰지 않는다.  
대신 **date boundary에서 finalize**가 발생하므로, runner도 그 구조를 따른다.

```
for each session round across active samples:
    1. 이번 round 시작 전에 date change가 난 sample들의 finalize job 수집
    2. summarize prompt들을 batch 실행
    3. summarize 결과로 new node 생성 + embedding
    4. relation prompt들을 batch 실행
    5. relation 결과 적용 후 node register
    6. 현재 round의 session turn들을 sample별로 순차 처리
       - retrieval
       - retrieval log
       - accumulated_turns 갱신
```

중요한 점은 다음과 같다.

- sample마다 `accumulated_turns`, `accumulated_sessions`, `prev_date`, `finalize_idx`를 따로 유지한다.
- 한 sample 안에서는 `summarize -> relation apply -> register` 순서를 절대 깨지 않는다.
- 같은 round에서 여러 sample의 finalize job이 생기면, **LLM 호출만** 묶어서 처리한다.

### Phase 2

QA는 아래 두 단계로 쪼개서 batch 처리한다.

```
retrieve_for_response(question)           # sample-local
    -> use_timeline
path_text 생성                           # sample-local
refine prompt batch
    -> refined_texts
QA prompt batch
    -> answer
```

즉, retrieval은 그대로 두고 그 이후의 refine / answer만 batch한다.

---

## 메모리 구성 방식

원본과 동일하게 memory는 per-turn으로 쓰지 않고, 같은 날짜의 session들을 모아
한 번에 finalize한다.

```
date change detected
    -> summarize completed dialogue batch
    -> create MemoryNode list
    -> embed new nodes
    -> find top-j past associative nodes
    -> extract relations
    -> register finalized nodes
```

node key format도 원본과 동일하다.

- `f{finalize_idx}-m{idx}`

예:

- `f0-m1`
- `f2-m3`

---

## Batch-safe 변경점

### 1. Step-wise memory graph API

`memory_graph.py`는 batch orchestration이 가능하도록 아래 단계를 분리했다.

- `build_summarize_prompt(...)`
- `apply_summarize_result(...)`
- `embed_nodes(...)`
- `build_relation_jobs(...)`
- `apply_relation_results(...)`
- `register_new_nodes(...)`

원래의 `finalize_conv(...)`도 남겨두었지만, batch runner는 위 step API를 사용한다.

### 2. Shared embedding model

batch runner는 `SentenceTransformer`를 한 번만 만들고, 각 sample의
`TheanineModule`에 주입한다.  
그래서 sample batch가 커져도 embedding model을 중복 로드하지 않는다.

### 3. Deterministic category 5 choice order

category 5는 기존에는 choice order가 random이었다.  
batch 환경에서는 실행 순서가 달라질 수 있으므로, `theanine_batch`에서는
`sample_id + qa_idx + question` 기반으로 choice order를 결정한다.

### 4. Completed-sample-id checkpoint

checkpoint는 더 이상 `last_completed_sample_index`에만 의존하지 않는다.

새 포맷:

```json
{
  "completed_sample_ids": ["0", "1", "3"]
}
```

기존 index checkpoint도 읽을 수 있게 backward compatibility를 유지했다.

---

## 새 batch 설정

`config_0.py`에 아래 값이 추가됐다.

```python
BATCH_SIZE = 4
FINALIZE_BATCH_SIZE = 8
RELATION_BATCH_SIZE = 64
REFINE_BATCH_SIZE = 64
QA_BATCH_SIZE = 64
```

의미는 다음과 같다.

- `BATCH_SIZE`: 동시에 처리할 sample 수
- `FINALIZE_BATCH_SIZE`: summarize job chunk 크기
- `RELATION_BATCH_SIZE`: relation extraction chunk 크기
- `REFINE_BATCH_SIZE`: timeline refinement chunk 크기
- `QA_BATCH_SIZE`: QA generation chunk 크기

---

## 실행 예시

```bash
python run_experiment.py \
  --start-sample 0 \
  --end-sample 9 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel 1 \
  --gpu-memory 0.5 \
  --batch-size 4 \
  --config config_0
```

---

## 출력물

원본과 같은 결과 스키마를 유지한다.

- `results_*.json`
- `checkpoint_*.json`
- `retrieval_logs/sample_*_retrieval_log.jsonl`
- `memory_snapshots/sample_*/`
- `prompt_log/sample_*/call_*/*.jsonl`

따라서 기존 후처리 스크립트와 merge 흐름을 거의 그대로 재사용할 수 있다.

---

## 검증 상태

현재 확인한 항목:

- `python -m py_compile ...` 통과
- `python run_experiment.py --help` 실행 확인

아직 하지 않은 항목:

- 실제 GPU inference smoke test
- 소규모 sample 범위에 대한 end-to-end 결과 검증
