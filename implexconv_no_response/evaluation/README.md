# evaluation

실험 결과를 평가하는 스크립트 모음.

## evaluation_basic.py

개인화 대화 실험 결과에 대한 QA 평가 및 토큰/메모리 통계 산출 스크립트.

### 입력 구조

```
evaluation/{root}/{subset}_{session_num}/{model}_{session_num}/
    ├── results_{LLM}_{subset_full}_session_{s}_{e}.json
    └── retrieval_logs/
        ├── session_0_retrieval_log.jsonl
        └── ...
```

### 출력 구조

```
evaluation/{root_without_results}_eval/{subset}_{session_num}/
    ├── qa_opp_{session_num}_score.csv      # opp subset
    ├── qa_sup_{session_num}_score.csv      # sup subset
    ├── token_memory_stats.csv
    └── qa_log_{session_num}/              # opp only
        └── qa_top_bottom_{model}_{llm}.json
```

### 실행

```bash
python evaluation_basic.py \
    --root qwen3_1.7b_results \
    --subset opp \          # opp (opposed) 또는 sup (supportive)
    --session_num 500 \
    --k 5 \                 # top/bottom k QA 저장 개수 (default: 5)
    --emb-model sentence-transformers/all-MiniLM-L6-v2
```

### 평가 지표

| subset | 지표 | 설명 |
|--------|------|------|
| `opp` | `avg_emb_sim` | generated/ground-truth answer 간 코사인 유사도 평균 |
| `sup` | `accuracy` | yes/no 정답 분류 정확도 |
| 공통 | `total/valid_qa`, `total/valid_turn` | QA 개수 및 턴 수 |
| 공통 | `token_memory_stats.csv` | 평균 토큰 사용량, API 호출 수, 메모리 검색 수 |

### 기타

- **증분 처리**: 이미 평가된 모델은 CSV에서 확인 후 스킵.
- **유효 QA**: `generated_answer`와 `ground_truth_answer`가 모두 비어 있지 않고 `"N/A"`가 아닌 경우. `sup`의 경우 GT가 `yes`/`no`로 정규화되어야 함.
- **턴 수**: `retrieval_logs/` 내 JSONL 파일의 라인 수로 산출.

---

## evaluation_llm_judge.py

LLM을 judge로 활용해 personalization 품질을 3개 축으로 평가하는 스크립트. `opp` subset 전용.

### 평가 축

| 축 | 점수 | 설명 |
|----|------|------|
| Axis 1 — Implicit Reasoning Correctness | 0–4 | 숨겨진 persona factor를 얼마나 정확히 파악하고 반영했는지 |
| Axis 2 — Task Answer Quality | 0–4 | 질문에 직접적이고 유용하게 답했는지 |
| Axis 3 — Personalization Quality | 0–2 | 해당 사용자의 상황에 맞게 구체적으로 개인화했는지 |
| **합산** | **0–10** | 세 축의 단순 합산 |

### 입력 구조

```
evaluation/{root}/opp_{n}/{model}_{n}/
    └── results_*.json
```

### 출력 구조

```
evaluation/{eval_base}_judge_{judge_short}/opp_{n}/
    ├── judge_scores_{model}_{llm}.json   # 세션별 QA 점수 상세
    └── judge_summary.csv                 # 모델별 집계 점수
```

- `eval_base`: `{root}`에서 `_results` 제거 (e.g. `qwen3_1.7b_results` → `qwen3_1.7b`)
- `judge_short`: judge 모델 경로의 마지막 컴포넌트 (e.g. `Llama-3.1-8B-Instruct`)

### 실행

```bash
python evaluation_llm_judge.py \
    --root qwen3_1.7b_results \
    --judge-model meta-llama/Llama-3.1-8B-Instruct \
    --session-num 200 \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --max-model-len 8192 \
    --batch-size 32
```

### judge_scores JSON 스키마

```json
[
  {
    "session_id": 1,
    "qa_scores": [
      {
        "question": "...",
        "generated_answer": "...",
        "ground_truth_answer": "...",
        "axis1_implicit_reasoning": 3,
        "axis2_task_quality": 4,
        "axis3_personalization": 2,
        "total": 9,
        "valid": true
      }
    ]
  }
]
```

### judge_summary.csv 컬럼

| 컬럼 | 설명 |
|------|------|
| `model` | 평가 대상 모델명 |
| `llm` | 추론에 사용한 LLM |
| `subset` | 항상 `opp` |
| `num_valid_qa` | 유효하게 평가된 QA 수 |
| `num_failed_qa` | 파싱/평가 실패한 QA 수 |
| `avg_implicit_reasoning` | Axis 1 평균 (0–4) |
| `avg_task_quality` | Axis 2 평균 (0–4) |
| `avg_personalization` | Axis 3 평균 (0–2) |
| `avg_total` | 합산 평균 (0–10) |

### 기타

- **Judge LLM**: vLLM으로 로컬 추론, `GuidedDecodingParams`로 JSON 스키마 강제 → 파싱 실패 최소화. 실패 시 개별 재시도 (최대 3회).
- **Reference conv**: 데이터셋의 `retrieved_conv_ids` (GT 기준) 사용. 비어있을 경우 임베딩 유사도 top-3으로 fallback.
- **증분 처리**: `judge_scores_{model}_{llm}.json` 파일과 `judge_summary.csv` 행이 모두 존재하면 스킵.
- **Temperature**: 0 (결정론적).
