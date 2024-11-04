import gradio as gr
import random
import os
from collections import defaultdict
import openai
from openai import OpenAI
import anthropic

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key="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")  # Replace with your actual Anthropic key
openai_client = OpenAI(api_key="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")

# Model and ELO score data
elo_scores = defaultdict(lambda: 1500)
vote_counts = defaultdict(int)
model_data = {
    'gpt-4': {'organization': 'OpenAI', 'license': 'OpenAI API'},
    'gpt-3.5-turbo': {'organization': 'OpenAI', 'license': 'OpenAI API'},
    'claude-3-5-haiku-20241022': {'organization': 'Anthropic', 'license': 'Anthropic API'},
    'claude-3-5-sonnet-20241022': {'organization': 'Anthropic', 'license': 'Anthropic API'},
}

def get_openai_response(model_name, prompt):
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message
    except Exception as e:
        return f"Error with OpenAI model {model_name}: {str(e)}"

def get_anthropic_response(model_name, prompt):
    try:
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=1000,
            temperature=0,
            system="You are a helpful assistant.",
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error with Anthropic model {model_name}: {str(e)}"

def get_model_response(model_name, prompt):
    if model_name.startswith('gpt'):
        return get_openai_response(model_name, prompt)
    elif model_name.startswith('claude'):
        return get_anthropic_response(model_name, prompt)
    else:
        return "Model not found or unsupported."

def submit_prompt(prompt):
    models = list(model_data.keys())
    model1, model2 = random.sample(models, 2)
    response_a = get_model_response(model1, prompt)
    response_b = get_model_response(model2, prompt)
    return response_a, response_b, model1, model2

def vote(choice, model_a, model_b):
    elo_a, elo_b = elo_scores[model_a], elo_scores[model_b]
    K = 32
    Ea, Eb = 1 / (1 + 10 ** ((elo_b - elo_a) / 400)), 1 / (1 + 10 ** ((elo_a - elo_b) / 400))
    Sa, Sb = (1, 0) if choice == 'A' else (0, 1) if choice == 'B' else (0.5, 0.5)
    elo_scores[model_a] += K * (Sa - Ea)
    elo_scores[model_b] += K * (Sb - Eb)
    vote_counts[model_a] += 1
    vote_counts[model_b] += 1
    return gr.update(visible=False)

def leaderboard_data():
    leaderboard = sorted(
        [{"Model": model, "ELO Score": round(elo_scores[model], 2), "95% CI": f"±{1.96 * (400 / (vote_counts[model] + 1) ** 0.5):.2f}", 
          "# Votes": vote_counts[model], "Organization": model_data[model]["organization"], "License": model_data[model]["license"]}
         for model in elo_scores],
        key=lambda x: x["ELO Score"], reverse=True
    )
    return leaderboard

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Chatbot Arena")
    
    with gr.Tabs():
        with gr.TabItem("Arena"):
            prompt_input = gr.Textbox(label="Enter your prompt")
            response_a, response_b = gr.Textbox(label="Response A", interactive=False), gr.Textbox(label="Response B", interactive=False)
            submit_button = gr.Button("Submit")
            vote_a_button, vote_tie_button, vote_b_button = gr.Button("Vote A"), gr.Button("Vote Tie"), gr.Button("Vote B")

        with gr.TabItem("Leaderboard"):
            refresh_button = gr.Button("Refresh Leaderboard")
            leaderboard_table = gr.Dataframe(headers=["Model", "ELO Score", "95% CI", "# Votes", "Organization", "License"], datatype=["str", "number", "str", "number", "str", "str"])

    submit_button.click(submit_prompt, inputs=prompt_input, outputs=[response_a, response_b])
    vote_a_button.click(lambda model_a, model_b: vote('A', model_a, model_b), inputs=[], outputs=[])
    vote_b_button.click(lambda model_a, model_b: vote('B', model_a, model_b), inputs=[], outputs=[])
    vote_tie_button.click(lambda model_a, model_b: vote('Tie', model_a, model_b), inputs=[], outputs=[])
    refresh_button.click(leaderboard_data, inputs=None, outputs=leaderboard_table)

demo.launch()