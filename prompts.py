# Default values for compatible mode
DEFAULT_EVAL_CRITERIA = """Does the model provide relevant and useful responses to the user's needs or questions?"""

DEFAULT_SCORE_1 = "The model's responses are irrelevant or unhelpful to the user's needs or queries."
DEFAULT_SCORE_2 = "The model sometimes provides helpful information, but often fails to address the user's actual needs or questions."
DEFAULT_SCORE_3 = "The model generally provides helpful responses that address the user's needs, though it may occasionally miss the mark."
DEFAULT_SCORE_4 = "The model regularly provides helpful responses that are well-aligned with the user's inquiries, with only rare inaccuracies."
DEFAULT_SCORE_5 = "The model consistently offers highly relevant and useful responses that perfectly cater to the user's needs and inquiries."

# Default Eval Prompt
DEFAULT_EVAL_PROMPT = """Does the model provide relevant and useful responses to the user's needs or questions?

Scoring Rubric:
Score 1: The model's responses are irrelevant or unhelpful to the user's needs or queries.
Score 2: The model sometimes provides helpful information, but often fails to address the user's actual needs or questions.
Score 3: The model generally provides helpful responses that address the user's needs, though it may occasionally miss the mark.
Score 4: The model regularly provides helpful responses that are well-aligned with the user's inquiries, with only rare inaccuracies.
Score 5: The model consistently offers highly relevant and useful responses that perfectly cater to the user's needs and inquiries.

[User Query]: {{input}}

[AI Response]: {{response}}"""

# Split the eval prompt into editable and fixed parts
DEFAULT_EVAL_PROMPT_EDITABLE = """Does the model provide relevant and useful responses to the user's needs or questions?

Scoring Rubric:
Score 1: The model's responses are irrelevant or unhelpful to the user's needs or queries.
Score 2: The model sometimes provides helpful information, but often fails to address the user's actual needs or questions.
Score 3: The model generally provides helpful responses that address the user's needs, though it may occasionally miss the mark.
Score 4: The model regularly provides helpful responses that are well-aligned with the user's inquiries, with only rare inaccuracies.
Score 5: The model consistently offers highly relevant and useful responses that perfectly cater to the user's needs and inquiries."""

# Fixed suffix that will always be appended
FIXED_EVAL_SUFFIX = """
[User Query]: {{human_input}}

[AI Response]: {{ai_response}}"""

# Define the Prometheus prompt used by default (without reference)
PROMETHEUS_PROMPT = """###Task Description:
An instruction (might include an Input inside it) and a response to evaluate are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other openings, closings, or explanations.

###The instruction to evaluate:
{human_input}

###Response to evaluate:
{ai_response}

###Score Rubrics:
[{eval_criteria}]
Score 1: {score1_desc}
Score 2: {score2_desc}
Score 3: {score3_desc}
Score 4: {score4_desc}
Score 5: {score5_desc}

###Feedback:
"""

# Define the Prometheus prompt with reference response
PROMETHEUS_PROMPT_WITH_REFERENCE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other openings, closings, or explanations.

###The instruction to evaluate:
{human_input}

###Response to evaluate:
{ai_response}

###Reference Answer (Score 5):
{ground_truth_input}

###Score Rubrics:
[{eval_criteria}]
Score 1: {score1_desc}
Score 2: {score2_desc}
Score 3: {score3_desc}
Score 4: {score4_desc}
Score 5: {score5_desc}

###Feedback:
"""

# Define the Flow Judge prompt
FLOW_JUDGE_PROMPT = """# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large \
language model.

You will be provided with the inputs and output of the task, as well as the evaluation criteria \
and scoring rubric. Your task is to evaluate the output of the AI system based on the evaluation \
criteria and scoring rubric provided.

# INPUT
Below are the inputs required for performing the task:
<inputs>
{INPUTS}
</inputs>

# OUTPUT
Below is the output of the task:
<output>
{OUTPUT}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
Here are the evaluation criteria and the rubric that you need to use for evaluating the task:
<evaluation_criteria>
{EVALUATION_CRITERIA}
</evaluation_criteria>

<scoring_rubric>
{RUBRIC}
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. \
Review the evaluation criteria and scoring rubric to understand the different levels of \
performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output \
generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score \
descriptions in the scoring rubric. For each criterion,decide which description best matches the \
output.
4. After comparing the output to the score descriptions, pay attention to the small details that \
might impact the final score that you assign. Sometimes a small difference can dictate the final \
score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring \
to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
- Write the verbal feedback inside <feedback> tags without any additional surrounding text.
- Write the numeric score inside <score> tags, without any additional surrounding text and always \
after the feedback.

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric."""

# Judge system prompt for non-Prometheus models
JUDGE_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate based on the user's instruction. Your output format should strictly adhere to JSON as follows: {"feedback": "<write feedback>", "result": <numerical score>}. Ensure the output is valid JSON, without additional formatting or explanations.""" 

ATLA_PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|> You are tasked with evaluating a response based on a given instruction (which may contain an Input) and a scoring rubric that serve as the evaluation standard. Provide a comprehensive feedback on the response quality strictly adhering to the scoring rubric, without any general evaluation. Follow this with a score between 1 and 5, referring to the scoring rubric. Avoid generating any additional opening, closing, or explanations.
  Here are some rules of the evaluation:
  (1) You should prioritize evaluating whether the response satisfies the provided rubric. The basis of your score should depend exactly on the rubric. However, the response does not need to explicitly address points raised in the rubric. Rather, evaluate the response based on the criteria outlined in the rubric.

  Your reply should strictly follow this format:
  **Reasoning:** <Your feedback>

  **Result:** <an integer between 1 and 5>

  Here is the data:

  Instruction:
  ```
  {human_input}
  ```

  Response:
  ```
  {ai_response}
  ```

  Score Rubrics:
  [{eval_criteria}]
  Score 1: {score1_desc}
  Score 2: {score2_desc}
  Score 3: {score3_desc}
  Score 4: {score4_desc}
  Score 5: {score5_desc}
  <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

ATLA_PROMPT_WITH_REFERENCE = """You are tasked with evaluating a response based on a given instruction (which may contain an Input) and a scoring rubric and reference answer that serve as the evaluation standard. Provide a comprehensive feedback on the response quality strictly adhering to the scoring rubric, without any general evaluation. Follow this with a score between 1 and 5, referring to the scoring rubric. Avoid generating any additional opening, closing, or explanations.

  Here are some rules of the evaluation:
  (1) You should prioritize evaluating whether the response satisfies the provided rubric. The basis of your score should depend exactly on the rubric. However, the response does not need to explicitly address points raised in the rubric. Rather, evaluate the response based on the criteria outlined in the rubric.

  Your reply should strictly follow this format:
  **Reasoning:** <Your feedback>

  **Result:** <an integer between 1 and 5>

  Here is the data:

  Instruction:
  ```
  {human_input}
  ```

  Response:
  ```
  {ai_response}
  ```

  Score Rubrics:
  [{eval_criteria}]
  Score 1: {score1_desc}
  Score 2: {score2_desc}
  Score 3: {score3_desc}
  Score 4: {score4_desc}
  Score 5: {score5_desc}

  Reference answer:
  {ground_truth_input}"""