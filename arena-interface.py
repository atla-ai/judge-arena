from datetime import datetime
import json
import gradio as gr
import re
import random
import time
from collections import defaultdict
from functools import partial
import openai 
from openai import OpenAI
import anthropic
import pandas as pd
from together import Together
import os




# Initialize clients
os.environ["TOGETHER_API_KEY"] = "89fc43189ac1782cd65e42bdf80343099c5bef78a121da2eff5e7e1a500aac72" 
anthropic_client = anthropic.Anthropic(api_key="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")  # Replace with your actual Anthropic key
openai_client = OpenAI(api_key="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")
together_client = Together()  # No API key needed in constructor

# Model and ELO score data
elo_scores = defaultdict(lambda: 1500)
vote_counts = defaultdict(int)
model_data = {
    'gpt-4o': {'organization': 'OpenAI', 'license': 'Proprietary'},
    'gpt-4o-mini': {'organization': 'OpenAI', 'license': 'Proprietary'},
    'gpt-3.5-turbo': {'organization': 'OpenAI', 'license': 'Proprietary'},
    'claude-3-5-haiku-20241022': {'organization': 'Anthropic', 'license': 'Proprietary'},
    'claude-3-5-sonnet-20241022': {'organization': 'Anthropic', 'license': 'Proprietary'},
    'claude-3-opus-20240229': {'organization': 'Anthropic', 'license': 'Proprietary'},
    'Meta Llama 3.1 8B Instruct Turbo': {
        'organization': 'Meta', 
        'license': 'Open Source',
        'api_model': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
    },
    'Meta Llama 3.1 70B Instruct Turbo': {
        'organization': 'Meta', 
        'license': 'Open Source',
        'api_model': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    },
}

current_session_id = 0
voting_data = []

def get_new_session_id():
    global current_session_id
    current_session_id += 1
    return f"user{current_session_id}"

def store_vote_data(prompt, response_a, response_b, model_a, model_b, winner, judge_id):
    vote_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "model_a": model_a,
        "model_b": model_b,
        "winner": winner,
        "judge_id": judge_id,
    }
    voting_data.append(vote_entry)
    
    # Optionally save to file after each vote
    with open('voting_data.json', 'w') as f:
        json.dump(voting_data, f, indent=2)

def parse_variables(prompt):
    # Extract variables enclosed in double curly braces
    variables = re.findall(r'{{(.*?)}}', prompt)
    # Remove duplicates while preserving order
    seen = set()
    variables = [x.strip() for x in variables if not (x.strip() in seen or seen.add(x.strip()))]
    return variables

def get_final_prompt(eval_prompt, variable_values):
    # Replace variables in the eval prompt with their values
    for var, val in variable_values.items():
        eval_prompt = eval_prompt.replace('{{' + var + '}}', val)
    return eval_prompt

def get_openai_response(model_name, prompt):
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
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
    elif model_name in model_data and 'api_model' in model_data[model_name]:
        return get_together_response(model_data[model_name]['api_model'], prompt)
    else:
        return "Model not found or unsupported."

def submit_prompt(eval_prompt, *variable_values):
    try:
        variables = parse_variables(eval_prompt)
        variable_values_dict = {var: val for var, val in zip(variables, variable_values)}
        final_prompt = get_final_prompt(eval_prompt, variable_values_dict)

        models = list(model_data.keys())
        model1, model2 = random.sample(models, 2)
        model_a, model_b = (model1, model2) if random.random() < 0.5 else (model2, model1)

        response_a = get_model_response(model_a, final_prompt)
        response_b = get_model_response(model_b, final_prompt)

        return (
            response_a,                  # response_a textbox
            response_b,                  # response_b textbox
            gr.update(visible=True),     # action_buttons_row
            gr.update(visible=True),     # regenerate_button
            model_a,                     # model_a_state
            model_b                      # model_b_state
        )
    except Exception as e:
        print(f"Error in submit_prompt: {str(e)}")
        # Return default values in case of error
        return (
            "Error generating response",
            "Error generating response",
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            None
        )

def vote(choice, model_a, model_b, prompt, response_a, response_b, judge_id):
    # Update ELO scores based on user choice
    elo_a = elo_scores[model_a]
    elo_b = elo_scores[model_b]
    K = 32  # ELO K-factor

    # Calculate expected scores
    Ea = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    Eb = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

    # Assign actual scores
    if choice == 'A':
        Sa, Sb = 1, 0
    elif choice == 'B':
        Sa, Sb = 0, 1
    else:
        Sa, Sb = 0.5, 0.5

    # Update scores and vote counts
    elo_scores[model_a] += K * (Sa - Ea)
    elo_scores[model_b] += K * (Sb - Eb)
    vote_counts[model_a] += 1
    vote_counts[model_b] += 1

    # Store the vote data
    store_vote_data(prompt, response_a, response_b, model_a, model_b, choice, judge_id)

    # Return updates for vote buttons and model names
    return [
        gr.update(visible=False),  # action_buttons_row
        gr.update(value=f"*Model: {model_a}*"),  # model_name_a
        gr.update(value=f"*Model: {model_b}*")   # model_name_b
    ]

def get_leaderboard():
    # Generate leaderboard data
    leaderboard = []
    for model, elo in elo_scores.items():
        votes = vote_counts[model]
        ci = 1.96 * (400 / (votes + 1) ** 0.5)  # Approximate 95% confidence interval
        data = {
            'Model': model,
            'ELO Score': f"{elo:.2f}",
            '95% CI': f"±{ci:.2f}",
            '# Votes': votes,
            'Organization': model_data[model]['organization'],
            'License': model_data[model]['license'],
        }
        leaderboard.append(data)
    # Sort by ELO score
    leaderboard.sort(key=lambda x: float(x['ELO Score']), reverse=True)
    return leaderboard

def regenerate_prompt(model_a, model_b, eval_prompt, *variable_values):
    # Similar to submit_prompt but with guaranteed different models
    variables = parse_variables(eval_prompt)
    variable_values_dict = {var: val for var, val in zip(variables, variable_values)}
    final_prompt = get_final_prompt(eval_prompt, variable_values_dict)

    # Get available models excluding the previous ones
    available_models = [m for m in model_data.keys() if m not in (model_a, model_b)]
    
    # If we have enough models for new pairs
    if len(available_models) >= 2:
        model1, model2 = random.sample(available_models, 2)
    else:
        # Fallback to allowing previous models if necessary
        model1, model2 = random.sample(list(model_data.keys()), 2)
    
    response_a = get_model_response(model1, final_prompt)
    response_b = get_model_response(model2, final_prompt)

    return (
        response_a,
        response_b,
        gr.update(visible=True),  # Show voting buttons
        gr.update(value="*Model: Unknown*"),  # Hide model A name
        gr.update(value="*Model: Unknown*"),  # Hide model B name
        model1,  # Return new model_a for state
        model2   # Return new model_b for state
    )

# Add these constants at the top of your file
K_FACTOR = 32  # Standard chess K-factor, adjust as needed
DEFAULT_ELO = 1200  # Starting ELO for new models

def calculate_elo_change(rating_a, rating_b, winner):
    """Calculate ELO rating changes for both players."""
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    
    if winner == "A":
        score_a, score_b = 1, 0
    elif winner == "B":
        score_a, score_b = 0, 1
    else:  # Handle ties
        score_a, score_b = 0.5, 0.5
    
    change_a = K_FACTOR * (score_a - expected_a)
    change_b = K_FACTOR * (score_b - expected_b)
    
    return change_a, change_b

def update_leaderboard():
    """Calculate current ELO ratings from voting history."""
    ratings = defaultdict(lambda: DEFAULT_ELO)
    matches = defaultdict(int)
    wins = defaultdict(int)
    
    # Load voting data
    try:
        with open('voting_data.json', 'r') as f:
            voting_data = json.load(f)
    except FileNotFoundError:
        return pd.DataFrame()
    
    # Process each vote
    for vote in voting_data:
        model_a = vote['model_a']
        model_b = vote['model_b']
        winner = vote['winner']
        
        # Skip if models aren't in current model_data
        if model_a not in model_data or model_b not in model_data:
            continue
        
        # Update match counts
        matches[model_a] += 1
        matches[model_b] += 1
        if winner == "A":
            wins[model_a] += 1
        elif winner == "B":
            wins[model_b] += 1
        else:  # Handle ties
            wins[model_a] += 0.5
            wins[model_b] += 0.5
        
        # Update ELO ratings
        change_a, change_b = calculate_elo_change(ratings[model_a], ratings[model_b], winner)
        ratings[model_a] += change_a
        ratings[model_b] += change_b
    
    # Create leaderboard DataFrame
    leaderboard_data = []
    for model in model_data.keys():  # Only include current models
        win_rate = (wins[model] / matches[model] * 100) if matches[model] > 0 else 0
        leaderboard_data.append({
            'Model': model,
            'ELO': round(ratings[model], 1),
            'Matches': matches[model],
            'Wins': wins[model],
            'Win Rate': f"{win_rate:.1f}%",
            'Organization': model_data[model]['organization'],
            'License': model_data[model]['license']
        })
    
    # Sort by ELO rating
    df = pd.DataFrame(leaderboard_data)
    return df.sort_values('ELO', ascending=False).reset_index(drop=True)

# Update your display_leaderboard function
def display_leaderboard():
    df = update_leaderboard()
    return gr.DataFrame(
        value=df,
        headers=['Model', 'ELO', 'Matches', 'Wins', 'Win Rate', 'Organization', 'License'],
        datatype=['str', 'number', 'number', 'number', 'str', 'str', 'str'],
        row_count=(len(df) + 1, 'dynamic'),
    )

def get_together_response(model_name, prompt):
    try:
        response = together_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with Together model {model_name}: {str(e)}"

with gr.Blocks(theme='default', css="""
    .prompt-row {
        align-items: flex-start !important;
    }
    .send-button {
        margin-top: 40px;  /* Adjust this to match the height of the TextArea label */
    }
""") as demo:
    judge_id = gr.State(get_new_session_id())
    gr.Markdown("# Judge Arena")
    gr.Markdown("*(Add your scoring details and other information here.)*")
    with gr.Tabs():
        with gr.TabItem("Judge Arena"):
            # Model Responses at the top, side-by-side
            with gr.Row():
                with gr.Column():
                    response_a = gr.TextArea(label="Response A", lines=10, interactive=False)
                    model_name_a = gr.Markdown("*Model: Unknown*")
                with gr.Column():
                    response_b = gr.TextArea(label="Response B", lines=10, interactive=False)
                    model_name_b = gr.Markdown("*Model: Unknown*")
            # Initially hide vote buttons and regenerate button
            with gr.Row(visible=False) as action_buttons_row:
                vote_a = gr.Button("Choose A", variant="primary")
                vote_tie = gr.Button("Tie", variant="secondary")
                vote_b = gr.Button("Choose B", variant="primary")
            regenerate_button = gr.Button("Regenerate", variant="secondary", visible=False)
            # Eval Prompt and Variables below
            with gr.Row(elem_classes="prompt-row"):
                eval_prompt = gr.TextArea(
                    label="Eval Prompt", 
                    lines=1, 
                    placeholder="Type your eval prompt here... denote variables like a ground truth response with {{variable}} to be populated below.",
                    show_label=True,
                    scale=8  # Adjust this value to control width ratio
                )
                send_btn = gr.Button(
                    value="Send",
                    variant="primary",
                    scale=1,  # Adjust this value to control width ratio
                    min_width=50,
                    elem_classes="send-button"
                )
            gr.Markdown("### Eval Variables denoted by {{variable}}")
            # Create inputs for up to 5 variables
            variable_rows = []
            for i in range(5):
                with gr.Row(visible=False) as var_row:
                    with gr.Column(scale=0.2, min_width=80):  # Reduced scale and min_width
                        var_label = gr.Markdown("Variable")
                    with gr.Column(scale=1):  # Reduced scale
                        var_input = gr.Textbox(label="", container=False)
                    variable_rows.append((var_row, var_label, var_input))
        with gr.TabItem("Leaderboard"):
            refresh_button = gr.Button("Refresh")
            leaderboard_table = gr.Dataframe(
                headers=['Model', 'ELO', 'Matches', 'Wins', 'Win Rate', 'Organization', 'License'],
                datatype=['str', 'number', 'number', 'number', 'str', 'str', 'str']
            )

    # Define state variables for model tracking
    model_a_state = gr.State()
    model_b_state = gr.State()

    # Update variable inputs based on the eval prompt
    def update_variables(eval_prompt):
        variables = parse_variables(eval_prompt)
        updates = []
        for i in range(5):
            var_row, var_label, var_input = variable_rows[i]
            if i < len(variables):
                updates.extend([
                    gr.update(visible=True),  # var_row
                    gr.update(value=f"**{variables[i]}:**"),  # var_label
                    gr.update(visible=True)  # var_input
                ])
            else:
                updates.extend([
                    gr.update(visible=False),  # var_row
                    gr.update(),  # var_label
                    gr.update(visible=False, value="")  # var_input
                ])
        return updates

    eval_prompt.change(fn=update_variables, inputs=eval_prompt, outputs=[item for sublist in variable_rows for item in sublist])

    # Regenerate button functionality
    regenerate_button.click(
        fn=regenerate_prompt,
        inputs=[model_a_state, model_b_state, eval_prompt] + [var_input for _, _, var_input in variable_rows],
        outputs=[response_a, response_b, action_buttons_row, model_name_a, model_name_b, model_a_state, model_b_state],
        queue=True
    )

    # Update model names after responses are generated
    def update_model_names(model_a, model_b):
        return gr.update(value=f"*Model: {model_a}*"), gr.update(value=f"*Model: {model_b}*")

    vote_a.click(
        fn=vote,
        inputs=[
            gr.State('A'), model_a_state, model_b_state,
            eval_prompt, response_a, response_b, judge_id
        ],
        outputs=[action_buttons_row, model_name_a, model_name_b]
    )

    vote_b.click(
        fn=vote,
        inputs=[
            gr.State('B'), model_a_state, model_b_state,
            eval_prompt, response_a, response_b, judge_id
        ],
        outputs=[action_buttons_row, model_name_a, model_name_b]
    )

    vote_tie.click(
        fn=vote,
        inputs=[
            gr.State('Tie'), model_a_state, model_b_state,
            eval_prompt, response_a, response_b, judge_id
        ],
        outputs=[action_buttons_row, model_name_a, model_name_b]
    )

    send_btn.click(
        fn=submit_prompt,
        inputs=[eval_prompt] + [var_input for _, _, var_input in variable_rows],
        outputs=[response_a, response_b, action_buttons_row, regenerate_button, model_a_state, model_b_state],
        queue=True
    )

    # Update the leaderboard
    def update_leaderboard():
        leaderboard = get_leaderboard()
        data = [
            [
                entry['Model'],
                float(entry['ELO Score']),
                entry['95% CI'],
                entry['# Votes'],
                entry['Organization'],
                entry['License']
            ] for entry in leaderboard
        ]
        return gr.update(value=data)

    refresh_button.click(fn=update_leaderboard, inputs=None, outputs=leaderboard_table)

demo.launch()


