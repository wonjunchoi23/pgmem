# Current `amem/` Layout: What Changed and Why

이 문서는 기존 순차 실행 구현을 현재의 batched main implementation으로 옮기면서
어떤 부분을 어떻게 수정했는지 기록한다. 현재 구조 기준으로는:

- `amem/`가 메인 batched 구현
- `amem/amem_sequential/`는 이전 sequential 구현 보관본

다른 메모리 모듈에 동일한 전환을 적용할 때 참고용으로 작성했다.

---

## 핵심 아이디어

여러 세션은 서로 독립적(메모리 상태 공유 없음)이므로 동시에 진행할 수 있다.
단, 세션 내부의 turn 순서는 유지해야 하므로 (turn N의 메모리가 turn N+1에 영향), 세션 전체를 병렬화하는 것이 아니라 **동일 turn_idx 시점의 LLM call을 여러 세션에서 모아서 배치**로 처리한다.

```
기존 sequential: session_0 turn_0 → LLM → turn_1 → LLM → ...
                                                     session_1 turn_0 → LLM → ...

현재 amem/:      [session_0 turn_0 prompt]
                 [session_1 turn_0 prompt]  →  vllm.generate([p0, p1, ...])  →  결과 분배
                 [session_N turn_0 prompt]
```

---

## 변경 파일 목록

| 파일 | 변경 규모 | 내용 요약 |
|---|---|---|
| `llm_module/llm_client.py` | 추가 | `generate_batch_raw()` |
| `memory_layer.py` | 중간 | `add_note()` 단계 분리, 공유 임베딩 모델, `accumulate_usage()` |
| `agent.py` | 소 | `embedding_model` 파라미터, `accumulate_memory_tokens()` |
| `run_experiment.py` | 대 | `BatchedAMEMRunner`, 세트 기반 checkpoint |
| `config_0.py` | 소 | `BATCH_SIZE`, `QA_BATCH_SIZE` 추가 |

---

## 1. `llm_module/llm_client.py` — `generate_batch_raw()` 추가

### 변경 내용

`vLLMClient`와 `UnifiedLLMClient`에 `generate_batch_raw()` 메서드 추가.

```python
def generate_batch_raw(
    self,
    prompts: List[str],          # 여러 prompt content (chat template 미적용 상태)
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    guided_json=None,            # 있으면 GuidedDecodingParams 적용
    return_usage: bool = False,  # True면 (texts, usages) 반환
) -> Union[List[str], Tuple[List[str], List[dict]]]
```

내부적으로 각 prompt에 chat template을 적용한 뒤, `self.llm.generate(all_prompts, sampling_params)` 를 **한 번** 호출한다. 기존 `generate()`는 `[single_prompt]`를 넘기던 것을 복수로 바꾼 것.

### 포인트

- 기존 `generate()`는 그대로 유지 → **non-breaking 추가**
- 배치 내 모든 prompt는 동일한 `SamplingParams`를 공유하므로, 호출자는 **같은 call type(analyze끼리, evolve끼리, QA끼리)** 로만 묶어야 함
- `return_usage=True`로 항상 호출해서 per-item 토큰 수를 받아 각 세션의 토큰 카운터에 귀속시킴

---

## 2. `memory_layer.py` — `add_note()` 단계 분리

### 배경

기존 `add_note()`는 내부에서 LLM call을 2번 연속으로 실행:
1. `_analyze_content()` → LLM call (plain text)
2. `_process_memory()` → LLM call (JSON, evolution)

배치 처리를 위해 이 두 call 사이에 "다른 세션의 같은 단계 prompt를 모아서 배치로 날리는" 동기화 지점이 필요하다. 따라서 `add_note()`를 **4개의 독립적인 단계**로 분리했다.

### 새 메서드 구조

```python
# Step 1: prompt 문자열 생성 (LLM call 없음)
prompt_str = memory_system.build_analyze_prompt(content)

# Step 2: analyze LLM 결과를 받아 MemoryNote 생성, evolution 필요 여부 판단 (LLM call 없음)
note, evolve_prompt, evolve_ctx = memory_system.apply_analyze_result(content, analysis_text, time)

# evolution이 필요 없으면 바로 저장
if evolve_prompt is None:
    memory_system.store_note(note)

# evolution이 필요하면 외부에서 LLM call 후 결과 전달
else:
    # (외부에서 배치 LLM call → response_json)
    memory_system.apply_evolve_result(note, response_json, evolve_ctx)
    # apply_evolve_result 내부에서 store_note() 호출
```

`add_note()`는 이 4단계를 순서대로 호출하는 wrapper로 유지 → **기존 순차 실행 경로와 완전히 호환**.

### `evolve_ctx`의 역할

`apply_analyze_result()` 시점의 메모리 상태 스냅샷 `(indices, all_notes, note_ids)`.
배치 처리 중 다른 세션의 note가 저장되더라도 각 세션의 evolution은 자신의 스냅샷 기준으로 수행되어 순차 실행과 동일한 결과를 낸다.

### 공유 임베딩 모델

```python
# 기존
class SimpleEmbeddingRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)  # 매 인스턴스마다 로드

# 변경
class SimpleEmbeddingRetriever:
    def __init__(self, model_name_or_instance):
        if isinstance(model_name_or_instance, str):
            self.model = SentenceTransformer(model_name_or_instance)
        else:
            self.model = model_name_or_instance  # 외부에서 주입된 공유 인스턴스
```

`AgenticMemorySystem`도 `embedding_model=None` 파라미터를 받아 전달.
corpus/embeddings는 여전히 세션별로 독립 관리되므로 공유해도 무방.

### `accumulate_usage()`

배치 LLM call은 `LLMWrapper`를 거치지 않으므로 토큰 카운터가 자동으로 업데이트되지 않는다. 배치 runner에서 per-item usage를 받아 수동으로 귀속시키기 위해 추가:

```python
# LLMWrapper
def accumulate_usage(self, input_tokens, output_tokens, api_calls=1): ...

# AgenticMemorySystem
def accumulate_token_counts(self, input_tokens, output_tokens, api_calls=1): ...
```

### `_EVOLUTION_GUIDED_JSON` export

`_EVOLUTION_JSON_SCHEMA`(response_format 구조)에서 실제 schema dict를 추출해 모듈 레벨 상수로 export. `generate_batch_raw(guided_json=...)`에 직접 전달하기 위해.

```python
_EVOLUTION_GUIDED_JSON = _EVOLUTION_JSON_SCHEMA["json_schema"]["schema"]
```

다른 모듈에 적용할 때: evolution이나 structured output call에서 사용하는 JSON schema가 있으면 동일하게 추출해서 export하면 됨.

---

## 3. `agent.py` — 공유 임베딩 + 토큰 귀속

```python
# 변경 전
def __init__(self, llm_client, model_path=""):

# 변경 후
def __init__(self, llm_client, model_path="", embedding_model=None):
    # embedding_model을 AgenticMemorySystem에 전달
```

```python
# 추가
def accumulate_memory_tokens(self, input_tokens, output_tokens, api_calls=1):
    self.memory_system.accumulate_token_counts(input_tokens, output_tokens, api_calls)
```

---

## 4. `run_experiment.py` — `BatchedAMEMRunner`

### Checkpoint 포맷 변경

순차 실행은 "마지막 완료 세션 인덱스" 하나만 저장하면 됐지만, 배치 내에서 일부 세션이 실패해 빠질 수 있으므로 **완료된 세션 ID 집합**으로 변경.

```python
# 기존
{"last_completed_session_index": 59}

# 변경
{"completed_session_ids": [0, 1, 2, ..., 59]}
```

구 포맷도 읽을 수 있도록 load 시 자동 변환 처리.
세션 완료 즉시 저장 (배치 전체 완료를 기다리지 않음).

### `BatchedAMEMRunner` 구조

```
run_batch(sessions, ...)
  ├── 세션별 agent 생성 (shared_embedding_model 주입)
  ├── _run_phase1_batched()
  │     └── for turn_idx:
  │           _process_turn_batch(is_user=True)   # user turn 배치
  │           _process_turn_batch(is_user=False)  # assistant turn 배치
  ├── _run_phase2_batched()                        # QA 전체 배치
  └── snapshot 저장, memory clear, 결과 집계
```

### `_process_turn_batch()` 흐름

```
1. 각 세션 retrieve (embedding, 순차)
2. [user only] response prompt 토큰 카운트, retrieval log 기록
3. analyze prompt 수집
4. [BATCH] generate_batch_raw(analyze_prompts, plain text)
5. 결과 로그 + 토큰 귀속 (accumulate_memory_tokens)
6. apply_analyze_result → evolve / no-evolve 분류
7. no-evolve notes 즉시 저장
8. [BATCH] generate_batch_raw(evolve_prompts, JSON)
9. 결과 로그 + 토큰 귀속 + apply_evolve_result
```

### `_batch_generate_with_retry()` 에러 처리

| 에러 유형 | 처리 방식 |
|---|---|
| JSON parse error (vLLM guided decoding 실패) | 해당 item만 `generate()`로 sequential retry |
| vLLM 내부 오류, OOM | 예외 propagate → main loop에서 프로세스 중단 |

### Phase 2 QA 배치

Phase 1 이후 메모리가 frozen된 상태에서, 모든 세션의 모든 QA 프롬프트를 모아 `QA_BATCH_SIZE` 단위로 chunking해서 처리.

### 공유 임베딩 모델 생성 위치

`main()`에서 LLM 로드 전에 SentenceTransformer를 한 번만 생성하고 `BatchedAMEMRunner`에 주입.
각 `run_batch()` 호출 시 이 인스턴스를 모든 agent에 공유.

---

## 다른 모듈에 적용할 때 체크리스트

1. **memory store 단계에 LLM call이 몇 개인가?**
   각 LLM call을 기준으로 단계를 분리. `build_prompt()` / `apply_result()` 패턴으로 쪼개면 됨.

2. **LLM call 간 의존성이 있는가?**
   예) evolution이 analyze 결과에 의존 → 두 call을 하나의 배치 단위로 묶을 수 없음. 두 단계를 별도 배치로 처리.

3. **JSON schema가 있으면 guided JSON dict를 export**
   `response_format` 구조에서 실제 schema dict를 꺼내 상수로 export → `generate_batch_raw(guided_json=...)` 에 전달.

4. **임베딩 모델이 있으면 공유 인스턴스 주입 인터페이스 추가**
   `__init__(model_name_or_instance)` 패턴.

5. **토큰 카운터 수동 귀속 인터페이스 추가**
   배치 call이 `LLMWrapper`를 bypass하므로 `accumulate_usage()` 류의 메서드 필요.

6. **Checkpoint를 세션 ID 집합으로 변경**
   배치 내 부분 실패 가능성 때문에 last_index 방식으로는 정확한 resume 불가.
