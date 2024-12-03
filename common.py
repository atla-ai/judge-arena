# Page Headers
MAIN_TITLE = "# Judge Arena: Benchmarking LLMs as Evaluators"

# How it works section
HOW_IT_WORKS = """
Vote to help the community find the best LLM-as-a-judge to use!
"""

BATTLE_RULES = """
## 🤺 Choose the winner
1. Define your scoring criteria in the **Evaluator Prompt** 
2. Add a test case to the **Sample to evaluate**
3. Test the evaluators & vote for the model that best aligns with your judgement!
\n
Variables defined in your prompt with {{double curly braces}} map to input fields under **Sample to evaluate**.

<br>
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
## 📝 Tips
**Precise evaluation criteria leads to more consistent and reliable judgments.** A good evaluation prompt should include the following elements:
- Evaluation criteria
- Scoring rubric 
- Examples (Optional)
"""

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
[User Query]: {{input}}

[AI Response]: {{response}}"""

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
<br><br>
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

By creating advanced evaluation models, we enable AI developers to identify and fix risks, leading to safer, more reliable AI that can be trusted and widely used. Our aim is to surpass the current state-of-the-art evaluation methods by training models specifically for evaluation. AIs will probably become very powerful, and perform tasks that are difficult for us to verify. We want to enable humans to oversee AI systems that are solving tasks too difficult for humans to evaluate. 
Read more about [our approach to scalable oversight](https://www.atla-ai.com/post/scaling-alignment) on our blog.
<br><br>
# Judge Arena Policy

## Overview

Judge Arena is an open-source platform dedicated to determining which models make the best judges. Users can run evals and assess anonymized responses from two competing model judges, choosing the better judgement or declaring a tie. This policy outlines our commitments to maintain a fair and open environment :)

## Transparency

- **Open-Source**: Judge Arena's code is open-source and available on GitHub. We encourage contributions from the community and anyone can replicate or modify the platform to suit their needs. We use proprietary model provider APIs where provided and Together AI's API to serve leading open-source models.
- **Methodology**: All processes related to model evaluation, rating calculations, and model selection are openly documented. 
- **Data Sharing**: Periodically, we'll share 20% of the collected evaluation data with the community. The data collected from Judge Arena is restricted to an anonymized user ID, the final prompt sent, the model responses, the user vote, and the timestamp.

## Model Inclusion Criteria

Judge Arena is specifically designed to assess AI models that function as evaluators (a.k.a judges). This includes but is not limited to powerful general-purpose models and the latest language models designed for evaluation tasks. Models are eligible for inclusion if they meet the following criteria:

- **Judge Capability**: The model should possess the ability to score AND critique other models' outputs effectively.
- **Promptable:** The model must be promptable to be evaluate in different scoring formats, for different criteria.
- **Accessibility**:
   - **Public API Access**: Models accessible through public APIs without restrictive barriers.
   - **Open-Source Models**: Models with publicly available weights that can be downloaded and run by the community.

## Leaderboard Management

- **ELO Ranking System**: Models are ranked on a public leaderboard based on aggregated user evaluations. We use an ELO rating system to rank AI judges on the public leaderboard. Each model begins with an initial rating of 1200, and we use a K-factor of 32 to determine the maximum rating adjustment after each evaluation.
- **Minimum Period**: Listed models remain accessible on Judge Arena for a minimum period of two weeks so they can be comprehensively evaluated.
- **Deprecation Policy**: Models may be removed from the leaderboard if they become inaccessible or are no longer publicly available.

*This policy might be updated to reflect changes in our practices or in response to community feedback.*
<br><br>
# FAQ

**Isn't this the same as Chatbot Arena?**

We are big fans of what the LMSYS team have done with Chatbot Arena and fully credit them for the inspiration to develop this. We were looking for a dynamic leaderboard that graded on AI judge capabilities and didn't manage to find one, so we created Judge Arena. This UI is designed especially for evals; to match the format of the model-based eval prompts that you would use in your LLM evaluation / monitoring tool.

**Why should I trust this leaderboard?**

We have listed out our efforts to be fully transparent in the policies above. All of the code for this leaderboard is open-source and can be found on our [Github](https://github.com/atla-ai/judge-arena). Check out our [blog](https://www.atla-ai.com/blog) to stay up to date as we analyse the results from the leaderboard.

**Who funds this effort?**

Atla currently funds this out of our own pocket. We are looking for API credits (with no strings attached) to support this effort - please get in touch if you or someone you know might be able to help.

**What is Atla working on?**

We are training a general-purpose evaluator that you will soon be able to run in this Judge Arena. Our next step will be to open-source a powerful model that the community can use to run fast and accurate evaluations.
<br><br>
# Get in touch
We’d love to hear your feedback! For general feature requests or to submit / suggest new models to add to the arena, please open up a discussion in the [community](https://huggingface.co/spaces/AtlaAI/judge-arena/discussions) tab. You can also contact us directly on [X](https://x.com/Atla_AI) or [Discord](https://discord.gg/V6TTGSTYHC).
\nPlease file any issues on our [Github](https://github.com/atla-ai/judge-arena)."""


# Default values for compatible mode
DEFAULT_EVAL_CRITERIA = """Does the model provide relevant and useful responses to the user's needs or questions?"""

DEFAULT_SCORE_1 = "The model's responses are irrelevant or unhelpful to the user's needs or queries."

DEFAULT_SCORE_2 = "The model sometimes provides helpful information, but often fails to address the user's actual needs or questions."

DEFAULT_SCORE_3 = "The model generally provides helpful responses that address the user's needs, though it may occasionally miss the mark."

DEFAULT_SCORE_4 = "The model regularly provides helpful responses that are well-aligned with the user's inquiries, with only rare inaccuracies."

DEFAULT_SCORE_5 = "The model consistently offers highly relevant and useful responses that perfectly cater to the user's needs and inquiries."

#**What are the Evaluator Prompt Templates based on?**

#As a quick start, we've set up templates that cover the most popular evaluation metrics out there on LLM evaluation / monitoring tools, often known as 'base metrics'. The data samples used in these were randomly picked from popular datasets from academia - [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [Preference Collection](https://huggingface.co/datasets/prometheus-eval/Preference-Collection), [RewardBench](https://huggingface.co/datasets/allenai/reward-bench), [RAGTruth](https://arxiv.org/abs/2401.00396).

#These templates are designed as a starting point to showcase how to interact with the Judge Arena, especially for those less familiar with using LLM judges.
