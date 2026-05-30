# PrefEval Experiment

## Dataset
- `dataset/implicit_persona.json` — 1,000 samples, each with a multi-turn
  `conversation` (5–6 turns), a `preference` (ground-truth label), a `question`
  (test query), plus `persona`, `topic`, `explanation`.
- In the raw data each turn stores `assistant` then `user`; feed modules in
  natural order `user → assistant`.
- `question`, `preference`, `explanation`, `persona` are **never** passed to the
  memory module or answering model — only used for QA/judging and copied to output.

## Core idea: chained sessions
- Take `samples[0..K]` and treat them as `K+1` consecutive sessions of the
  **same** user (even though personas/topics differ). This injects interference
  and tests whether the right preference can still be recalled per question.
- One chain per run; deterministic (no shuffling/seed). `K` set via `--end-session K`.

## Procedure
1. **Incremental memory build:** `m_i = update_memory(m_{i-1}, session_i)`,
   reusing a single in-memory module instance (snapshot to disk per checkpoint,
   never rebuild).
2. **Inline QA per checkpoint:** after reaching `m_i`, evaluate questions
   `q_0..q_i` against it before moving on. Each `q_j` is re-answered at every
   later checkpoint to measure degradation as unrelated sessions pile up.
   Total QA calls = `(K+1)(K+2)/2`. Don't cache answers.
3. Answering model sees only `m_i` (retrieved/dumped memory) + `q_j` text.

## Conventions
- **Virtual time:** reuse ImplexConv's scheme. `session_id=0`, `conv_id`=chain
  position, `turn_id`=local utterance index (user→`2k`, assistant→`2k+1`),
  `global_turn_id`=cumulative. `CONV_IDS_PER_DAY=2`, `TIME_PER_TURN_MINUTES=10`.
- **Persona:** not given to module/model; modules infer from conversation.
- **Constants:** `MAX_TOKENS=1500`, QA answer cap **200 words**;
  `RETRIEVE_K`/`EVOLUTION_THRESHOLD` keep module defaults.

## Resume
- Idempotent: scan `memory_snapshots/m_*/` for largest existing `k`.
  - `K ≤ k_existing` → no-op. `K > k_existing` → load `m_{k_existing}`,
    continue ingestion + inline QA, append new rows.
- Only valid if config is unchanged; otherwise use a new output dir.

## Output (`config_<n>_outputs_<model>/`)
- `results.jsonl` — one row per `(k, question_session j≤k)`: question,
  model_answer, retrieved_memories, topic/persona/preference/explanation, tokens.
- `retrieval_log.jsonl` — one row per QA retrieval (with `k`).
- `memory_snapshots/m_0..m_K/` — module-specific snapshot per checkpoint.
- `stats.json` (cumulative tokens/stats), `meta.json`, `prompt_log/`, `logs/`.

## CLI
- Add `--end-session K`. Keep `--model`, `--tensor-parallel`, `--gpu-memory`,
  `--max-model-len`, `--config`.
- Drop `--start-session`/range `--end-session`, `--subset`, session `--batch-size`.

## Batching
- Batch the `k+1` independent QA generations at each checkpoint (and judge calls
  later). Memory build is strictly sequential within and across sessions.
