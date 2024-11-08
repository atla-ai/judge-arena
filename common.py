# Page Headers
MAIN_TITLE = "# Judge Arena - Test anonymous LLM judges side-by-side"
SUBTITLE = "*Free LLM Evals to test your GenAI application.*"

# How it works section
HOW_IT_WORKS = """
# How it works:
- **Run any form of evaluation:** from simple hallucination detection to qualitative interpretations
- **Evaluate anything:** coding, analysis, creative writing, math, or general knowledge
"""

BATTLE_RULES = """
## 🤺 Battle Rules:
- Both AIs stay anonymous - if either reveals its identity, the duel is void
- Choose the LLM judge that most aligns with your judgement
- If both score the same - choose the critique that you prefer more!\n
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
"""

# Default Eval Prompt
DEFAULT_EVAL_PROMPT = """You are assessing a chat bot response to a user's input based on the helpfulness of the response.\n                    

Score:

A score of 1 means that the response's answer meets all of the evaluation criteria.

A score of 0 means that the response's answer does not meet all of the evaluation criteria.

Here is the data:\n

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

# Our Mission

By creating advanced evaluation models, we enable AI developers to identify and fix risks, leading to safer, more reliable AI that can be trusted and widely used. Our aim is to surpass the current state-of-the-art evaluation methods by training models specifically for evaluation. AIs will probably become very powerful, and perform tasks that are difficult for us to verify. We want to enable humans to oversee AI systems that are solving tasks too difficult for humans to evaluate. We have written more about [our approach to scalable oversight](https://www.atla-ai.com/post/scaling-alignment) on our blog.

# Judge Arena Policy

## Overview

Judge Arena is an open-source platform dedicated to improving the standard of evaluation of generative AI models in their role as judges. Users can run evals and assess anonymized responses from two competing model judges, choosing the better judgement or declaring a tie. This policy outlines our commitments and guidelines to ensure a fair, open, and collaborative environment for both users and model providers.

## Transparency

- **Open-Source**: Judge Arena's code is open-source and available on GitHub. This approach allows anyone to review, replicate, or modify the platform to suit their needs. We use proprietary model provider APIs where provided and Together AI's API to serve leading open-source models.
- **Community Engagement**: We actively encourage contributions from the community. Feedback, code contributions, and discussions are welcome to improve the platform's functionality, fairness, and transparency.
- **Methodology**: All processes related to model evaluation, rating calculations, and model selection are openly documented. This transparency ensures that our processes are understandable and reproducible by others.
- **Data Sharing**: Periodically, we will share 20% of the collected evaluation data with the community. This data includes anonymized prompts, model responses, and aggregated evaluation results.

## Model Inclusion Criteria

Judge Arena is specifically designed to assess AI models that function as evaluators (a.k.a judges), including but not limited to powerful general-purpose models and the latest language models designed for evaluation tasks. Models are eligible for inclusion if they meet the following criteria:

- **Judge Capability**: The model must possess the ability to score AND critique responses, content, or other models' outputs effectively.
- **Adaptable:** The model must be prompt-able to be evaluate in different scoring formats, for different criteria.
- **Accessibility**:
   - **Public API Access**: Models accessible through public APIs without restrictive barriers.
   - **Open-Source Models**: Models with publicly available weights that can be downloaded and run by the community.

## Evaluation Methodology

- **User Participation**: Users run evaluations and select preferred model responses based on quality, relevance, and accuracy contributing to the model's overall rating.
- **Blind Testing**: All model evaluations are conducted blindly. Users are not informed which model produced which response to eliminate bias.
- **Data Collection**: We collect sufficient data to ensure statistical significance in our evaluations. We additionally show the 95% confidence interval in the leaderboard to provide a signal of reliability.
- **Anomaly Detection**: We monitor user activity to detect and mitigate anomalous behavior or voting patterns that could skew results.

## Leaderboard Management

- **ELO Ranking System**: Models are ranked on a public leaderboard based on aggregated user evaluations. We use an ELO rating system to rank AI judges on the public leaderboard. Each model begins with an initial rating of 1500 (as is used by the International Chess Federation), and we use a K-factor of 32 to determine the maximum rating adjustment after each evaluation.
- **Minimum Period**: Listed models remain accessible on Judge Arena for a minimum period of two weeks to allow for comprehensive community evaluation.
- **Deprecation Policy**: Models may be removed from the leaderboard if they become inaccessible, are no longer publicly available.

## Privacy and Data Protection

- **Anonymization**: All shared data is anonymized to prevent the identification of individual users.

## Policy Updates and Communication

- **Ongoing Revisions**: This policy may be updated to reflect changes in our practices or in response to community feedback.
- **Notification of Changes**: Policy changes will be communicated to users and stakeholders on this page.

# FAQ

**Isn't this the same as Chatbot Arena?**

- We are big fans of what the LMSYS team have done with Chatbot Arena and fully credit them for the inspiration to develop this. We were looking for a dynamic leaderboard that graded on AI judge capabilities and didn't manage to find one, so we created Judge Arena. This UI is designed especially for evals; to match the format of the model-based eval prompts that you would use in your LLM evaluation / monitoring tool.

\n\n**Why should I trust this leaderboard?**

- We have listed out our efforts to be fully transparent in the policies above. All of the code for this leaderboard is open-source and can be found on our [Github](https://github.com/atla-ai/judge-arena).

\n\n**Who funds this effort?**

- Atla currently funds this out of our own pocket. We are looking for API credits (with no strings attached) to support this effort - please get in touch if you or someone you know might be able to help.

\n\n**What is Atla working on?**

- We are training a general-purpose evaluator that you will soon be able to run in this Judge Arena. Our next step will be to open-source a powerful model that the community can use to run fast and accurate evaluations.

## Get in touch
Feel free to email us at [support@atla-ai.com](mailto:support@atla-ai.com) or leave feedback on our [Github](https://github.com/atla-ai/judge-arena)!"""
