# ---------- Group 1: Response competence (Q + GT + gen) ----------

SYSTEM_PROMPT_RC = (
    "You are an expert evaluator assessing the standalone quality of an AI-generated response. "
    "Your job is to evaluate one specific aspect of response competence at a time. "
    "Do not consider personalization or user history — focus only on the requested aspect."
)

USER_PROMPT_TEMPLATE_RC1 = """\
## Task Description
A dialogue system has generated an answer to a user's question. Your job: judge **whether the generated answer directly addresses what the question is asking**.

---

## Evaluation Target
**Question:** {query}
**Reference Answer:** {gt_answer}
**Generated Answer:** {generated_answer}

---

## Scoring Rubric — Question Addressing (0 / 1)
- **1 — Addressed**: The generated answer directly responds to what the question asks. It engages with the actual request rather than deflecting, changing topic, or restating the question.
- **0 — Not Addressed**: The answer is off-topic, fails to engage with the question (e.g., only asks a clarifying question back, or restates the question without answering), or provides a response so general it could apply to any question.

---

Output ONLY a JSON object:
{{"reasoning": "<why this score — cite specific phrases from the generated answer>", "rc_question_addressing": <int 0 or 1>}}"""

JUDGE_JSON_SCHEMA_RC1 = {
    "type": "object",
    "properties": {
        "reasoning":              {"type": "string"},
        "rc_question_addressing": {"type": "integer", "minimum": 0, "maximum": 1},
    },
    "required": ["reasoning", "rc_question_addressing"],
    "additionalProperties": False,
}


# ---------- Group 2: Persona adaptation (Q + GT + reason + rel_conv + gen) ----------

SYSTEM_PROMPT_PA = (
    "You are an expert evaluator for a personalized conversational AI system. "
    "Your task is to assess one specific aspect of persona adaptation at a time. "
    "Focus only on the requested aspect — do not evaluate general answer quality or unrelated persona dimensions."
)

USER_PROMPT_TEMPLATE_PA1 = """\
## Task Description
A memory-augmented dialogue system has generated an answer for a user with a hidden **implicit persona factor** — a state, preference, constraint, or circumstance that should shape the response.
The supporting evidence is provided only to clarify the persona factor.
Your job: judge **whether the implicit persona factor is reflected in the generated answer**.

---

## Implicit Persona Factor
{reason}

## Supporting Evidence (relevant past conversations)
{reference_conv_block}

---

## Evaluation Target
**Question:** {query}
**Generated Answer:** {generated_answer}

---
## Scoring Rubric — Persona Recognition (0 / 1)
- **1 — Recognized**: The generated answer either acknowledges the implicit persona factor or makes a clear response choice because of it.
- **0 — Not Recognized**: The answer shows no evidence that the persona factor motivated any content choice. It reads as a generic response to the question alone.

---

Output ONLY a JSON object:
{{"reasoning": "<explain whether the generated answer clearly reflects the implicit persona factor; cite specific phrases>", "pa_persona_recognition": <int 0 or 1>}}"""


JUDGE_JSON_SCHEMA_PA1 = {
    "type": "object",
    "properties": {
        "reasoning":              {"type": "string"},
        "pa_persona_recognition": {"type": "integer", "minimum": 0, "maximum": 1},
    },
    "required": ["reasoning", "pa_persona_recognition"],
    "additionalProperties": False,
}


USER_PROMPT_TEMPLATE_PA2 = """\
## Task Description
A memory-augmented dialogue system has generated an answer for a user who has hidden persona context (state, preference, constraint, or circumstance not visible to you).
Your job: judge whether the answer reads as **tailored to a specific individual** or as a **generic answer that could be given to anyone** asking the same question.

You are intentionally NOT given the user's persona or past context. Judge purely from how the answer is written — does it treat the asker as a specific person with their own situation, or as an interchangeable querant?

---

## Evaluation Target
**Question:** {query}
**Generated Answer:** {generated_answer}

---

## Scoring Rubric — Generic Distinctness (0 / 1)
- **1 — Distinct**: The answer reads as personalized to a specific user. It references or builds on user-specific details (situation, preferences, constraints, prior choices, context), adapts its recommendation/framing to the individual, or makes choices that only make sense for *this* asker rather than the general public.
- **0 — Generic**: The answer reads as a stock response that could be addressed to almost any user asking the same question. It treats the asker as anonymous — no reference to their specifics, no individualized framing, no user-conditional reasoning.

Important:
- Do NOT reward concreteness alone — a detailed, specific answer can still be fully generic (e.g., a textbook how-to).
- Do NOT penalize an answer for lacking detail if the personalization shows up in framing or choice.
- You cannot verify whether the personalized claims are correct — only judge whether the answer reads as tailored to a specific individual.

---

Output ONLY a JSON object:
{{"reasoning": "<why this score — cite specific phrases that signal (or fail to signal) tailoring to a specific user>", "pa_generic_distinctness": <int 0 or 1>}}"""

JUDGE_JSON_SCHEMA_PA2 = {
    "type": "object",
    "properties": {
        "reasoning":               {"type": "string"},
        "pa_generic_distinctness": {"type": "integer", "minimum": 0, "maximum": 1},
    },
    "required": ["reasoning", "pa_generic_distinctness"],
    "additionalProperties": False,
}


USER_PROMPT_TEMPLATE_PA3 = """\
## Task Description
A memory-augmented dialogue system has generated an answer for a user with a hidden **implicit persona factor**.
Your job: judge **whether the persona is integrated into the answer's substance** (recommendations, alternatives, caveats, framing, stance) — beyond a mere surface-level word borrow.

---

## Implicit Persona Factor
{reason}

## Supporting Evidence (relevant past conversations)
{reference_conv_block}

## Reference Answer (one acceptable persona-aware response)
{gt_answer}

---

## Evaluation Target
**Question:** {query}
**Generated Answer:** {generated_answer}

---

## Scoring Rubric — Substantive Integration (0 / 1)
- **1 — Integrated**: The implicit persona factor shapes at least one substantive content choice in the answer, such as what is recommended, what alternative is offered, what caveat is raised, how the issue is framed, or what stance is taken.
- **0 — Surface-only or absent**: The factor is ignored or only mentioned at the wording/tone level. For example, the answer may say "given your situation" or repeat persona-related words, but still give the same recommendation, alternatives, caveats, framing, or stance it would give to any user.

---

Output ONLY a JSON object:
{{"reasoning": "<why this score — distinguish substantive integration from surface mention by citing specific phrases>", "pa_substantive_integration": <int 0 or 1>}}"""

JUDGE_JSON_SCHEMA_PA3 = {
    "type": "object",
    "properties": {
        "reasoning":                  {"type": "string"},
        "pa_substantive_integration": {"type": "integer", "minimum": 0, "maximum": 1},
    },
    "required": ["reasoning", "pa_substantive_integration"],
    "additionalProperties": False,
}


# ---------- Group 3: Groundedness (Q + history + gen) ----------

SYSTEM_PROMPT_GR = (
    "You are an expert evaluator assessing the groundedness of a personalized AI response against the user's prior utterances. "
    "Your job is to detect whether the answer fabricates or contradicts persona information. "
    "Focus only on persona facts — do not evaluate general answer quality."
)

USER_PROMPT_TEMPLATE_GR1 = """\
## Task Description
A memory-augmented dialogue system has generated an answer for a user. Your job: judge **whether the answer invents persona information that is not present in the user's prior utterances**.

---

## User's Prior Utterances
{conv_history}

---

## Evaluation Target
**Question:** {query}
**Generated Answer:** {generated_answer}

---

## Scoring Rubric — No Persona Hallucination (0 / 1)
- **1 — No hallucination**: All persona-related claims in the answer (the user's preferences, traits, circumstances, history) are supported by the user's prior utterances, OR the answer makes no persona-specific claims at all.
- **0 — Hallucinated persona**: The answer attributes preferences, traits, situations, or history to the user that are not supported by the user's prior utterances. Inventing persona facts counts here.

Note: General world facts unrelated to the user are not the focus — only fabricated **persona** information matters here.

---

Output ONLY a JSON object:
{{"reasoning": "<why this score — cite any unsupported persona claims, or note that all persona claims are grounded>", "gr_no_hallucination": <int 0 or 1>}}"""

JUDGE_JSON_SCHEMA_GR1 = {
    "type": "object",
    "properties": {
        "reasoning":           {"type": "string"},
        "gr_no_hallucination": {"type": "integer", "minimum": 0, "maximum": 1},
    },
    "required": ["reasoning", "gr_no_hallucination"],
    "additionalProperties": False,
}


USER_PROMPT_TEMPLATE_GR2 = """\
## Task Description
A memory-augmented dialogue system has generated an answer for a user. Your job: judge **whether the answer contradicts what is known about the user from the user's prior utterances**.

---

## User's Prior Utterances
{conv_history}

---

## Evaluation Target
**Question:** {query}
**Generated Answer:** {generated_answer}

---

## Scoring Rubric — No Persona Contradiction (0 / 1)
- **1 — No contradiction**: The answer is consistent with the user's established preferences, traits, constraints, and circumstances as revealed in their prior utterances.
- **0 — Contradicts persona**: The answer makes assumptions or claims that conflict with what the user's prior utterances establish about them (e.g., recommending something the user has explicitly rejected, or assuming a circumstance opposite to what the user has stated).

---

Output ONLY a JSON object:
{{"reasoning": "<why this score — cite the specific contradiction, or note consistency with the user's prior utterances>", "gr_no_contradiction": <int 0 or 1>}}"""

JUDGE_JSON_SCHEMA_GR2 = {
    "type": "object",
    "properties": {
        "reasoning":          {"type": "string"},
        "gr_no_contradiction": {"type": "integer", "minimum": 0, "maximum": 1},
    },
    "required": ["reasoning", "gr_no_contradiction"],
    "additionalProperties": False,
}
