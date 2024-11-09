# Page Headers
MAIN_TITLE = "# Judge Arena - Free LLM Evals to test your GenAI application"

# How it works section
HOW_IT_WORKS = """
- **Run any form of evaluation:** from simple hallucination detection to qualitative interpretations
- **Evaluate anything:** coding, analysis, creative writing, math, or general knowledge
"""

BATTLE_RULES = """
## 🤺 Battle Rules:
- Both AIs stay anonymous - if either reveals its identity, the duel is void
- Choose the LLM judge that most aligns with your judgement
- If both score the same - choose the critique that you prefer more!
<br><br>
"""

# CSS Styles
CSS_STYLES = """
    .prompt-row {
        align-items: flex-start !important;
    }
    .send-button-row {
        display: flex;
        justify-content: flex-end;
        margin-top: 8px;
    }
    /* Style for metric buttons */
    .metric-button-active {
        background-color: #2B3A55 !important;
        color: white !important;
    }
    /* Add this to ensure proper button spacing */
    .metric-buttons-row {
        gap: 8px;
    }
"""

# Default Eval Prompt
EVAL_DESCRIPTION = """
## 📝 Instructions
**Precise evaluation criteria leads to more consistent and reliable judgments.** A good evaluation prompt should include the following elements:
- Evaluation criteria
- Scoring rubric 
- (Optional) Examples\n

**Any variables you define in your prompt using {{double curly braces}} will automatically map to the corresponding input fields under "Sample to evaluate" section on the right.**

<br><br>
"""

DEFAULT_EVAL_PROMPT = """You are assessing a chat bot response to a user's input based on [INSERT CRITERIA]

Score:
A score of 1 means that the response's answer meets all of the evaluation criteria.
A score of 0 means that the response's answer does not meet all of the evaluation criteria.

Here is the data:
[BEGIN DATA]
***
[User Query]: {{input}}
***
[Response]: {{response}}
***
[END DATA]"""

# Default Variable Values
DEFAULT_INPUT = """Which of these animals is least likely to be found in a rainforest?"
A) Jaguar
B) Toucan
C) Polar Bear
D) Sloth"""
DEFAULT_RESPONSE = "C) Polar Bear"

# Voting Section Header
VOTING_HEADER = """
# Start Voting Now
"""

# Acknowledgements
ACKNOWLEDGEMENTS = """
<br><br><br>
# Acknowledgements

We thank [LMSYS Org](https://lmsys.org/) for their hard work on the Chatbot Arena and fully credit them for the inspiration to build this.

We thank [Clementine Fourrier](https://huggingface.co/clefourrier) and Hugging Face for their guidance and partnership in setting this up.
"""

# Policy Content
POLICY_CONTENT = """
# About Atla

Atla is an applied research organization that trains models as evaluators to capture human preferences. We're a team of researchers, engineers, and operational leaders, with experience spanning a variety of disciplines, all working together to build reliable and understandable AI systems. Our research is informed by our experiences conducting AI safety research at the UK AI Task Force, OpenAI and the Stanford Existential Risks Initiative.
<br><br>
# Our Mission

By creating advanced evaluation models, we enable AI developers to identify and fix risks, leading to safer, more reliable AI that can be trusted and widely used. Our aim is to surpass the current state-of-the-art evaluation methods by training models specifically for evaluation. AIs will probably become very powerful, and perform tasks that are difficult for us to verify. We want to enable humans to oversee AI systems that are solving tasks too difficult for humans to evaluate. We have written more about [our approach to scalable oversight](https://www.atla-ai.com/post/scaling-alignment) on our blog.
<br><br>
# Judge Arena Policy

## Overview

Judge Arena is an open-source platform dedicated to improving the standard of evaluation of generative AI models in their role as judges. Users can run evals and assess anonymized responses from two competing model judges, choosing the better judgement or declaring a tie. This policy outlines our commitments to maintain a fair, open, and collaborative environment :)

## Transparency

- **Open-Source**: Judge Arena's code is open-source and available on GitHub. We encourage contributions from the community and anyone can replicate or modify the platform to suit their needs. We use proprietary model provider APIs where provided and Together AI's API to serve leading open-source models.
- **Methodology**: All processes related to model evaluation, rating calculations, and model selection are openly documented. We'd like to ensure that our ranking system is understandable and reproducible by others!
- **Data Sharing**: Periodically, we'll share 20% of the collected evaluation data with the community. The data collected from Judge Arena is restricted to an anonymized user ID, the final prompt sent, the model responses, the user vote, and the timestamp.

## Model Inclusion Criteria

Judge Arena is specifically designed to assess AI models that function as evaluators (a.k.a judges). This includes but is not limited to powerful general-purpose models and the latest language models designed for evaluation tasks. Models are eligible for inclusion if they meet the following criteria:

- **Judge Capability**: The model should possess the ability to score AND critique responses, content, or other models' outputs effectively.
- **Adaptable:** The model must be prompt-able to be evaluate in different scoring formats, for different criteria.
- **Accessibility**:
   - **Public API Access**: Models accessible through public APIs without restrictive barriers.
   - **Open-Source Models**: Models with publicly available weights that can be downloaded and run by the community.

## Leaderboard Management

- **ELO Ranking System**: Models are ranked on a public leaderboard based on aggregated user evaluations. We use an ELO rating system to rank AI judges on the public leaderboard. Each model begins with an initial rating of 1500 (as is used by the International Chess Federation), and we use a K-factor of 32 to determine the maximum rating adjustment after each evaluation.
- **Minimum Period**: Listed models remain accessible on Judge Arena for a minimum period of two weeks so they can be comprehensively evaluated.
- **Deprecation Policy**: Models may be removed from the leaderboard if they become inaccessible or are no longer publicly available.

This policy might be updated to reflect changes in our practices or in response to community feedback.

# FAQ

**Isn't this the same as Chatbot Arena?**

We are big fans of what the LMSYS team have done with Chatbot Arena and fully credit them for the inspiration to develop this. We were looking for a dynamic leaderboard that graded on AI judge capabilities and didn't manage to find one, so we created Judge Arena. This UI is designed especially for evals; to match the format of the model-based eval prompts that you would use in your LLM evaluation / monitoring tool.

**What are the Evaluator Prompt Templates based on?**

As a quick start, we've set up templates that cover the most popular evaluation metrics out there on LLM evaluation / monitoring tools, often known as 'base metrics'. The data samples used in these were randomly picked from popular datasets from academia - [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [Preference Collection](https://huggingface.co/datasets/prometheus-eval/Preference-Collection), [RewardBench](https://huggingface.co/datasets/allenai/reward-bench), [RAGTruth](https://arxiv.org/abs/2401.00396).

These templates are designed as a starting point to showcase how to interact with the Judge Arena, especially for those less familiar with using LLM judges.

**Why should I trust this leaderboard?**

We have listed out our efforts to be fully transparent in the policies above. All of the code for this leaderboard is open-source and can be found on our [Github](https://github.com/atla-ai/judge-arena).

**Who funds this effort?**

Atla currently funds this out of our own pocket. We are looking for API credits (with no strings attached) to support this effort - please get in touch if you or someone you know might be able to help.

**What is Atla working on?**

We are training a general-purpose evaluator that you will soon be able to run in this Judge Arena. Our next step will be to open-source a powerful model that the community can use to run fast and accurate evaluations.
<br><br>
# Get in touch
Feel free to email us at [support@atla-ai.com](mailto:support@atla-ai.com) or leave feedback on our [Github](https://github.com/atla-ai/judge-arena)!"""
