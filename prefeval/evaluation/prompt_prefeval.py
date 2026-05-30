# ---------- 1) Preference Violation (Figure 15) ----------

USER_PROMPT_TEMPLATE_VIOLATION = """\
You will analyze a conversation between a user and an assistant, focusing on whether the assistant's response violates the user's stated preference.

Evaluate the response based on these criteria:

Answer "Yes" if:

1.  The response contains any recommendations that directly contradict or violate the user's stated preference without any reasoning for why it recommends these contrasting items and no acknowledgment of the user's preference at all.

Answer "No" if:

1.  The response respects the user's stated preference.
2.  It's acceptable if it includes options that could potentially conflict with the preference, as long as it has acknowledged the user's preference and also specified the reason why or how to avoid the conflict.
3.  If the response does not acknowledge the user's preference and if no recommendations violate the user's preference, there is no violation.

Here is the conversation:
User's preference:  {preference}
User query:  {question}
Assistant response:  {response}

Examine the response meticulously and answer.  Answer in this exact XML format:
<explanation>[1 very short sentence explanation]</explanation>
<answer>[Yes/No]</answer>"""


# ---------- 2) Acknowledgement of Preference (Figure 16) ----------

USER_PROMPT_TEMPLATE_ACKNOWLEDGEMENT = """\
You will analyze a conversation between a user and an assistant, focusing on whether the assistant acknowledges any user preference in answering the user's query.

Evaluate the response based on these stringent criteria:
1.  Check if the response explicitly or implicitly mentions or references a user preference.
2.  The content of the preference is irrelevant for this check; only its presence matters.

Answer "Yes" if:

1.  The response explicitly mentions or refers to a user preference in answering the user's question.  Examples include: "Based on your previous preference for xxxx"; "Based on your previous preference, I would suggest you xxxx"; "Since you mentioned you prefer/dislike xxxx"; "Since you are a xxxx", "I will recommend the following given your preference for xxx", etc.  You should extract the "xxxx" in your answer.  If it only says "Based on our previous conversation, I recommend..." and does not explicitly mention any preference, you should answer 'No'.
2.  The response assumes the user preference in answering the user's question implicitly.  For example, when the user asks 'Can you recommend me cars to drive?', if the response is 'Based on your preference, I will recommend non-EV cars, ...', then this indicates the assistant assumes that the user's preference is a dislike of EV cars, and you should answer 'Yes'.

Answer "No" if the response does not mention or refer to any user preference explicitly or implicitly.  If you cannot extract the sentence stating what the preference is, answer 'No'.

Here is the assistant's response:
The user's question is: {question}
Assistant. response: {response}

Examine the response meticulously and answer.  Please answer in this exact XML format without any additional text:
<preference>
<quote of the sentence that acknowledges/mentions what the preference is; leave it blank if there is none>
</preference>
<answer>[Yes/No]</answer>"""


# ---------- 3) Preference Hallucination (Figure 17) ----------
#
# In the paper this check operates on a short "restatement" quote extracted in
# a prior step (Fig 16's <preference> output). Per user direction we feed the
# full model answer as the restatement; the rubric still applies because the
# "Yes" criteria all hinge on meaning-level divergence vs. the GT preference,
# and the "empty -> No" clause naturally covers responses that don't restate.

USER_PROMPT_TEMPLATE_HALLUCINATION = """\
You will analyze a conversation between a user and an assistant, focusing on whether the assistant's restatement of the user's stated preference is the same preference.  Evaluate the response based on these stringent criteria to answer if the assistant has hallucinated the preference or not:

Answer "Yes" if:

1.  The assistant's restatement differs from the original preference in meaning or intent.
2.  The assistant's restatement contradicts the original preference.
3.  The assistant's restatement is not related or irrelevant to the original preference.

Answer "No" if has not hallucinated:

1.  The assistant's restatement matches the original preference.
2.  The assistant's restatement is a minor paraphrase that fully preserves the meaning and intent of the original preference.
3.  The restatement is just empty, no hallucination.

Here is the information:
Original user preference:  {preference}
Assistant's restatement:  {restatement}

Examine the original preference and the assistant's restatement meticulously and answer.  Please answer in this exact XML format without any other additional text:
<explanation>[1 short sentence explanation]</explanation>
<answer>[Yes/No]</answer>"""


# ---------- 4) Helpful Response (Figure 18) ----------

USER_PROMPT_TEMPLATE_HELPFUL = """\
You will analyze a conversation between a user and an assistant, focusing on whether the assistant provides any substantive helpful response to the user's query.

Evaluate the response based on these stringent criteria:

Answer "Yes" if any of the following:

1.  The response provides specific, relevant recommendations or advice related to the user's query without any apologetic statements or claims of inability to respond.
2.  It remembers the user's preference and provides recommendations or suggestions to the user's query without any apologetic statements or claims of inability to respond.

Answer "No" if any of the following:

1.  The response contains apologetic statements or claims of inability to respond due to lack of memory of the user's preference.
2.  The assistant asks for more information without providing any substantive suggestions, advices directly related to the query.
3.  The response doesn't address the user's specific query.
4.  Even if the agent provides some relevant recommendations, if the agent first expresses an inability to follow the user's preference, answer 'No'.

Examples of unhelpful responses (Answer:  No):
"Unfortunately, I don't actually have enough context about your specific preferences for xxx"
"Unfortunately, we haven't had a previous discussion about your preferences for xxx.  Could you let me know your preference for xxx?"
"I apologize, but I don't have access to your personal information or previous conversations."
"I'm sorry, but I can't provide a specific answer without more details."

Here is the conversation:
User query:  {question}
Assistant response:  {response}

Examine the response meticulously and answer.  Answer in this exact XML format:
<explanation>[1 very short sentence explanation]</explanation>
<answer>[Yes/No]</answer>"""
