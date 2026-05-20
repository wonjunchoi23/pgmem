# Evaluation

`evaluation_basic.py`는 각 LLM 결과 폴더를 읽어 LOCOMO 평가용 CSV 파일을 생성하는 스크립트입니다.
`evaluation_llm_judge.py`는 vLLM judge 모델로 각 QA 답변을 0-2점 척도로 평가합니다.

## 입력 구조

평가할 결과는 `evaluation/{llm}_results/` 아래에 모델별 하위 폴더로 있어야 합니다.

```text
evaluation/
  evaluation_basic.py
  {llm}_results/
    {model}/
      results_*.json
      retrieval_logs/
        *.jsonl
```

예시:

```text
evaluation/qwen3_1.7b_results/amem/results_Qwen3-1.7B_sample_0_9.json
evaluation/llama3.1_8b_results/theanine/results_Llama-3.1-8B-Instruct_sample_0_9.json
```

데이터셋은 상위 디렉토리의 `dataset/locomo10.json`을 사용합니다.

## 실행 방법

프로젝트 루트에서 실행합니다.

```bash
python evaluation/evaluation_basic.py --llm qwen3_1.7b
```

다른 평가 디렉토리를 지정하려면 `--eval_dir`를 사용합니다.

```bash
python evaluation/evaluation_basic.py --llm qwen3_1.7b --eval_dir /path/to/evaluation
```

LLM judge 평가는 다음처럼 실행합니다. `--root_dir`는 `evaluation/` 아래 결과 폴더 이름만 넣으면 됩니다.

```bash
python evaluation/evaluation_llm_judge.py \
  --root_dir qwen3_1.7b_results \
  --judge_model meta-llama/Llama-3.1-8B-Instruct \
  --tensor_parallel 1 \
  --gpu-memory 0.9 \
  --max-model-len 32768 \
  --batch-size 16
```

## 출력

`dataset_statistic.csv`는 `evaluation/` 바로 아래에 1회 생성되며, 이미 있으면 건너뜁니다.
모델별 평가 결과는 `evaluation/{llm}_basic_eval/`에 저장됩니다.

```text
evaluation/
  dataset_statistic.csv
  {llm}_basic_eval/
    answer_stats.csv
    sub_stats.csv
```

각 파일의 내용은 다음과 같습니다.

| 파일 | 내용 |
| --- | --- |
| `dataset_statistic.csv` | 샘플별 세션 수, 턴 수, 화자별 턴 수, 카테고리별 QA 개수 |
| `answer_stats.csv` | 모델별 전체 및 카테고리별 METEOR 점수, ROUGE-L 점수, 임베딩 cosine similarity |
| `sub_stats.csv` | 모델별 토큰 사용량, LLM 호출 수, retrieval 통계 |

LLM judge 결과는 `evaluation/{llm}_judge_{judge_model}/`에 저장됩니다.

```text
evaluation/qwen3_1.7b_judge_llama3.1_8b/
  amem/
    judge_results_Qwen3-1.7B_sample_0_9.json
  lbllm/
    judge_results_Qwen3-1.7B_sample_0_9.json
  judge_summary.csv
```

모델별 JSON에는 각 QA의 `question`, `generated_answer`, `ground_truth_answer`, `evidence`, `ref_conv`, `judge` 결과가 저장됩니다.
`judge_summary.csv`에는 모델별 전체 평균 점수와 카테고리별 평균 점수가 저장됩니다.

LLM judge 스크립트는 출력 JSON이 이미 있으면 해당 `results_*.json` 파일을 건너뜁니다.

## 참고

- 평가 카테고리는 `1`부터 `5`까지 사용합니다.
- 임베딩 similarity 계산에는 `sentence-transformers`의 `all-MiniLM-L6-v2`를 사용하며, 로컬에 모델이 없으면 similarity 컬럼은 `NaN`으로 남깁니다.
- ROUGE-L 계산에는 `fmeasure`를 사용합니다.
- METEOR 계산은 로컬 토크나이저와 WordNet 비활성 fallback으로 수행합니다.
- `sub_stats.csv`의 `avg_llm_calls`는 `token_statistics.total_llm_calls`를 우선 사용하고, 구스키마에서는 `num_total_api_calls`로 fallback 합니다.
- `sub_stats.csv`의 `avg_qa_input` / `avg_qa_output`는 각 sample의 `qa_results[*].qa_tokens` 총합 평균을 사용하며, 필요 시 `call_*_qa` 또는 구스키마 `total_qa_input` / `total_qa_output`로 fallback 합니다.
- LLM judge에서 카테고리 `1-4`는 `2=정답`, `1=부분 정답/애매함`, `0=오답`으로 평가합니다.
- LLM judge에서 카테고리 `5` adversarial question은 답변 불가를 올바르게 인식하면 `2`, 그렇지 않으면 `0`으로 평가합니다.
