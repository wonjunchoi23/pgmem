# ImplexConv Preprocessed Dataset

## Overview

This directory contains the preprocessed ImplexConv dataset, which has been structured and cleaned for easier access and analysis. The original dataset has been restructured to improve usability while maintaining all conversation and QA pair information.

## Dataset Files

### Raw Data
- **ImplexConv_opposed.json** (507 MB): Original opposed conversations dataset with 1,550 sessions
- **ImplexConv_supportive.json** (180 MB): Original supportive conversations dataset with 814 sessions

### Processed Data
- **ImplexConv_opposed_processed.json** (700 MB): Preprocessed opposed conversations dataset
- **ImplexConv_supportive_processed.json** (262 MB): Preprocessed supportive conversations dataset

## Preprocessing Steps

### 1. Data Filtering
- **Removed sessions with zero QA pairs** to ensure data quality
  - Opposed: 1,550 → 1,433 sessions (117 removed)
  - Supportive: 814 → 814 sessions (0 removed)

### 2. Data Sorting
- **Sessions sorted by QA count** (descending order)
- Sessions with more QA pairs appear first in the processed data

### 3. Data Restructuring
Each processed session follows this structure:

```json
{
  "metadata": {
    "session_id": int,
    "total_conversations": int,
    "total_turns": int
  },
  "conversations": [
    {
      "session_id": int,
      "conv_id": int,
      "turn_id": int,
      "global_turn_id": int,
      "speaker": str,
      "utterance": str
    }
  ],
  "qa": [...]
}
```

#### Field Descriptions
- **metadata**
  - `session_id`: Unique session identifier
  - `total_conversations`: Number of conversation threads in the session
  - `total_turns`: Total number of conversational turns (user/assistant exchanges)

- **conversations** (array of turns)
  - `session_id`: Session identifier (matches parent session)
  - `conv_id`: Conversation thread identifier
  - `turn_id`: Turn index within the conversation thread (0-indexed)
  - `global_turn_id`: Sequential turn index across all conversations in the session (0-indexed)
  - `speaker`: Either "user" (for user utterances) or "assistant" (for assistant utterances)
  - `utterance`: The actual text of the turn

- **qa**: Question-Answer pairs related to the session (original structure preserved)

## Dataset Statistics

### ImplexConv Opposed (Processed)
- **Sessions**: 1,433
- **Total Conversations**: 158,281
- **Total Turns**: 2,090,322
- **Total QA Pairs**: 3,633
- **Average QA per Session**: 2.54
- **QA Range**: 1-5 pairs per session

### ImplexConv Supportive (Processed)
- **Sessions**: 814
- **Total Conversations**: 68,322
- **Total Turns**: 790,433
- **Total QA Pairs**: 4,884
- **Average QA per Session**: 6.00
- **QA Range**: 6 pairs per session (uniform)

## Data Characteristics

### Conversation Parsing
The conversation text is parsed from the raw format using the following rules:
- Lines are split by newline (`\n`)
- Each line is expected to follow the format: `[speaker]: [utterance]`
- Speaker identification:
  - If the speaker part contains the word "speaker" (case-insensitive), it's labeled as `"user"`
  - Otherwise, it's labeled as `"assistant"`
- Empty lines are ignored
- Lines without a colon are skipped

### Session Types

**Opposed Sessions**: Conversations where participants express opposing viewpoints or disagreements (1,433 sessions after filtering)

**Supportive Sessions**: Conversations characterized by supportive and collaborative dialogue (814 sessions)

## Speaker Labels

The restructured data uses standardized speaker labels:
- **"user"**: Utterances from the user or speaker
- **"assistant"**: Utterances from the assistant or AI system

These labels make it easy to filter and analyze conversations based on speaker identity.

## Usage

### Loading the Data
```python
import json

# Load opposed conversations
with open('ImplexConv_opposed_processed.json', 'r') as f:
    opposed_data = json.load(f)

# Load supportive conversations
with open('ImplexConv_supportive_processed.json', 'r') as f:
    supportive_data = json.load(f)

# Access a session
session = opposed_data[0]
print(f"Session {session['metadata']['session_id']}")
print(f"Conversations: {session['metadata']['total_conversations']}")
print(f"Turns: {session['metadata']['total_turns']}")
print(f"QA pairs: {len(session['qa'])}")

# Iterate through conversations
for turn in session['conversations'][:5]:
    print(f"{turn['speaker']}: {turn['utterance']}")
```

### Filtering by Speaker
```python
# Get all user utterances in a session
user_turns = [turn for turn in session['conversations'] if turn['speaker'] == 'user']

# Get all assistant utterances in a session
assistant_turns = [turn for turn in session['conversations'] if turn['speaker'] == 'assistant']
```

### Analyzing Conversations
```python
# Get a specific conversation thread
conversation_id = 0
conv = [turn for turn in session['conversations'] if turn['conv_id'] == conversation_id]

# Sort by turn order within the conversation
conv_sorted = sorted(conv, key=lambda x: x['turn_id'])

for turn in conv_sorted:
    print(f"[Turn {turn['turn_id']}] {turn['speaker']}: {turn['utterance']}")
```

## File Sizes
- ImplexConv_opposed_processed.json: ~700 MB
- ImplexConv_supportive_processed.json: ~262 MB
- **Total**: ~962 MB

## Preprocessing Configuration

The preprocessing was performed with the following settings:
- Minimum QA requirement: 1 pair per session
- Speaker identification: Case-insensitive matching for "speaker" keyword
- Session ID offset: 0 for both datasets
- Conversation ID sorting: Numeric ascending order
- Output format: JSON with indent=2 for readability

## Data Quality Notes

- The `supportive` dataset has uniform QA distribution (all sessions have exactly 6 QA pairs), while the `opposed` dataset has variable QA counts (1-5 pairs)
- Total turns significantly exceeds QA pairs, indicating rich conversational context around each QA pair
- The restructured format separates conversation turns from QA pairs for flexible analysis
- All sessions in the processed data contain at least 1 QA pair
- Speaker labels are normalized to lowercase ("user" and "assistant") for consistency

## Related Files

- **implexconv_preprocess.ipynb**: Jupyter notebook containing the preprocessing pipeline, data analysis, and visualization
- **ImplexConv_opposed.json**: Original opposed conversations (raw format)
- **ImplexConv_supportive.json**: Original supportive conversations (raw format)

## Version History

- **Current Version**: Uses lowercase speaker labels ("user", "assistant")
- Previous versions may have used capitalized labels ("Speaker", "Assistant")

## Contact & Questions

For questions about the preprocessing pipeline, refer to `implexconv_preprocess.ipynb` which contains detailed code and explanations of each preprocessing step.
