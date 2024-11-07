from datetime import datetime
import json
import gradio as gr
import re
import random
from collections import defaultdict
import pandas as pd
import os   
from dotenv import load_dotenv
from gen_api_answer import get_model_response
from common import *

load_dotenv() 

# Model and ELO score data
DEFAULT_ELO = 1500  # Starting ELO for new models
elo_scores = defaultdict(lambda: DEFAULT_ELO)
vote_counts = defaultdict(int)

# Load the model_data from JSONL
def load_model_data():
    model_data = {}
    try:
        with open('data/models.jsonl', 'r') as f:
            for line in f:
                model = json.loads(line)
                model_data[model['name']] = {
                    'organization': model['organization'],
                    'license': model['license'],
                    'api_model': model['api_model']
                }
    except FileNotFoundError:
        print("Warning: models.jsonl not found")
        return {}
    return model_data

model_data = load_model_data()

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
    
    # Save to file after each vote
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

def submit_prompt(eval_prompt, *variable_values):
    try:
        variables = parse_variables(eval_prompt)
        variable_values_dict = {var: val for var, val in zip(variables, variable_values)}
        final_prompt = get_final_prompt(eval_prompt, variable_values_dict)

        models = list(model_data.keys())
        model1, model2 = random.sample(models, 2)
        model_a, model_b = (model1, model2) if random.random() < 0.5 else (model2, model1)

        response_a = get_model_response(model_a, model_data.get(model_a), final_prompt)
        response_b = get_model_response(model_b, model_data.get(model_b), final_prompt)

        return (
            response_a,
            response_b,
            gr.update(visible=True),
            gr.update(visible=True),
            model_a,
            model_b
        )
    except Exception as e:
        print(f"Error in submit_prompt: {str(e)}")
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
    elo_scores[model_a] += K_FACTOR * (Sa - Ea)
    elo_scores[model_b] += K_FACTOR * (Sb - Eb)
    vote_counts[model_a] += 1
    vote_counts[model_b] += 1

    # Store the vote data
    store_vote_data(prompt, response_a, response_b, model_a, model_b, choice, judge_id)

    # Return updates for UI components
    return {
        action_buttons_row: gr.update(visible=False),
        model_name_a: gr.update(value=f"*Model: {model_a}*"),
        model_name_b: gr.update(value=f"*Model: {model_b}*"),
        send_btn: gr.update(interactive=True),
        regenerate_button: gr.update(visible=True, interactive=True)
    }



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
    
    response_a = get_model_response(model1, model_data.get(model1), final_prompt)
    response_b = get_model_response(model2, model_data.get(model2), final_prompt)

    # Parse the responses
    score_a, critique_a = parse_model_response(response_a)
    score_b, critique_b = parse_model_response(response_b)

    return (
        score_a,              # score_a textbox
        critique_a,           # critique_a textbox
        score_b,              # score_b textbox
        critique_b,           # critique_b textbox
        gr.update(visible=True),  # action_buttons_row
        gr.update(value="*Model: Unknown*"),  # model_name_a
        gr.update(value="*Model: Unknown*"),  # model_name_b
        model1,              # model_a_state
        model2               # model_b_state
    )

# Add these constants at the top of your file
K_FACTOR = 32  # Standard chess K-factor, adjust as needed
DEFAULT_ELO = 1500  # Starting ELO for new models

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
        ci = 1.96 * (400 / (matches[model] + 1) ** 0.5) if matches[model] > 0 else 0  # Confidence interval
        leaderboard_data.append({
            'Model': model,
            'ELO': round(ratings[model], 1),
            '95% CI': f"±{ci:.1f}",
            'Matches': matches[model],
            'Win Rate': f"{win_rate:.1f}%",
            'Organization': model_data[model]['organization'],
            'License': model_data[model]['license']
        })
    
    # Sort by ELO rating
    df = pd.DataFrame(leaderboard_data)
    return df.sort_values('ELO', ascending=False).reset_index(drop=True)

# Update the display_leaderboard function
def display_leaderboard():
    df = update_leaderboard()
    return gr.DataFrame(
        value=df,
        headers=['Model', 'ELO', '95% CI', 'Matches', 'Organization', 'License'],
        datatype=['str', 'number', 'str', 'number', 'str', 'str', 'str'],
        row_count=(len(df) + 1, 'dynamic'),
    )

# Update the leaderboard table definition in the UI
leaderboard_table = gr.Dataframe(
    headers=['Model', 'ELO', '95% CI', 'Matches', 'Organization', 'License'],
    datatype=['str', 'number', 'str', 'number', 'str', 'str', 'str']
)

def parse_model_response(response):
    try:
        # Debug print
        print(f"Raw model response: {response}")
        
        # First try to parse the entire response as JSON
        try:
            data = json.loads(response)
            return str(data.get('result', 'N/A')), data.get('feedback', 'N/A')
        except json.JSONDecodeError:
            # If that fails (typically for smaller models), try to find JSON within the response
            json_match = re.search(r'{.*}', response)
            if json_match:
                data = json.loads(json_match.group(0))
                return str(data.get('result', 'N/A')), data.get('feedback', 'N/A')
            else:
                return 'Error', f"Failed to parse response: {response}"
                
    except Exception as e:
        # Debug print for error case
        print(f"Failed to parse response: {str(e)}")
        return 'Error', f"Failed to parse response: {response}"

def get_leaderboard_stats():
    """Get summary statistics for the leaderboard."""
    try:
        with open('voting_data.json', 'r') as f:
            voting_data = json.load(f)
        
        total_votes = len(voting_data)
        total_models = len(model_data)
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        return f"""
### Leaderboard Stats
- **Total Models**: {total_models}
- **Total Votes**: {total_votes}
- **Last Updated**: {last_updated}
"""
    except FileNotFoundError:
        return "No voting data available"

def initialize_voting_data():
    """Initialize or clear the voting data file."""
    empty_data = []
    with open('voting_data.json', 'w') as f:
        json.dump(empty_data, f)

# Add this near the start of your app initialization, before the Gradio interface setup
if __name__ == "__main__":
    initialize_voting_data()
    
    # ... rest of your Gradio app setup ...

with gr.Blocks(theme='default', css=CSS_STYLES) as demo:
    judge_id = gr.State(get_new_session_id())
    gr.Markdown(MAIN_TITLE)
    gr.Markdown(SUBTITLE)
    
    with gr.Tabs():
        with gr.TabItem("Judge Arena"):
            gr.Markdown(HOW_IT_WORKS)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown(BATTLE_RULES)
            
            # Add heading for Eval Prompt
            gr.Markdown("\n")
            
            # Eval Prompt and Variables side by side
            with gr.Row():
                # Left column - Eval Prompt
                with gr.Column(scale=1):
                    eval_prompt = gr.TextArea(
                        label="Eval Prompt",
                        lines=1, 
                        value=DEFAULT_EVAL_PROMPT,
                        placeholder="Type your eval prompt here... denote variables in {{curly brackets}} to be populated on the right.",
                        show_label=True
                    )

                # Right column - Variable Mapping
                with gr.Column(scale=1):
                    gr.Markdown("### Variable Mapping")
                    # Create inputs for up to 5 variables, with first two visible by default
                    variable_rows = []
                    for i in range(5):
                        initial_visibility = True if i < 2 else False
                        with gr.Group(visible=initial_visibility) as var_row:
                            # Variable input with direct label
                            initial_value = DEFAULT_INPUT if i == 0 else DEFAULT_RESPONSE
                            initial_label = "input" if i == 0 else "response" if i == 1 else f"variable_{i+1}"
                            var_input = gr.Textbox(
                                label=initial_label,
                                value=initial_value,
                                container=True
                            )
                            variable_rows.append((var_row, var_input))

            # Send button
            with gr.Row(elem_classes="send-button-row"):
                send_btn = gr.Button(
                    value="Send",
                    variant="primary",
                    size="lg",
                    scale=1
                )
            
            # Add divider heading for model outputs
            gr.Markdown(VOTING_HEADER)
            
            # Model Responses side-by-side
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model A")
                    score_a = gr.Textbox(label="Score", interactive=False)
                    critique_a = gr.TextArea(label="Critique", lines=8, interactive=False)
                    model_name_a = gr.Markdown("*Model: Unknown*")
                with gr.Column():
                    gr.Markdown("### Model B")
                    score_b = gr.Textbox(label="Score", interactive=False)
                    critique_b = gr.TextArea(label="Critique", lines=8, interactive=False)
                    model_name_b = gr.Markdown("*Model: Unknown*")
            
            # Initially hide vote buttons and regenerate button
            with gr.Row(visible=False) as action_buttons_row:
                vote_a = gr.Button("Choose A", variant="primary")
                vote_tie = gr.Button("Tie", variant="secondary")
                vote_b = gr.Button("Choose B", variant="primary")
            regenerate_button = gr.Button("Regenerate with different models", variant="secondary", visible=False)
            
            # Add spacing and acknowledgements at the bottom
            gr.Markdown(ACKNOWLEDGEMENTS)

        with gr.TabItem("Leaderboard"):
            refresh_button = gr.Button("Refresh")
            stats_display = gr.Markdown()
            leaderboard_table = gr.Dataframe(
                headers=['Model', 'ELO', '95% CI', 'Matches', 'Organization', 'License'],
                datatype=['str', 'number', 'str', 'number', 'str', 'str']
            )

        with gr.TabItem("Policy"):
            gr.Markdown(POLICY_CONTENT)

    # Define state variables for model tracking
    model_a_state = gr.State()
    model_b_state = gr.State()

    # Update variable inputs based on the eval prompt
    def update_variables(eval_prompt):
        variables = parse_variables(eval_prompt)
        updates = []
        for i in range(5):
            var_row, var_input = variable_rows[i]
            if i < len(variables):
                updates.extend([
                    gr.update(visible=True),  # var_row
                    gr.update(value=f"**{variables[i]}:**"),  # var_input
                    gr.update(visible=True)  # var_input
                ])
            else:
                updates.extend([
                    gr.update(visible=False),  # var_row
                    gr.update(),  # var_input
                    gr.update(visible=False, value="")  # var_input
                ])
        return updates

    eval_prompt.change(fn=update_variables, inputs=eval_prompt, outputs=[item for sublist in variable_rows for item in sublist])

    # Regenerate button functionality
    regenerate_button.click(
        fn=regenerate_prompt,
        inputs=[model_a_state, model_b_state, eval_prompt] + [var_input for _, var_input in variable_rows],
        outputs=[
            score_a,
            critique_a,
            score_b,
            critique_b,
            action_buttons_row,
            model_name_a,
            model_name_b,
            model_a_state,
            model_b_state
        ]
    )

    # Update model names after responses are generated
    def update_model_names(model_a, model_b):
        return gr.update(value=f"*Model: {model_a}*"), gr.update(value=f"*Model: {model_b}*")

    # Store the last submitted prompt and variables for comparison
    last_submission = gr.State({})


    # Update the vote button click handlers
    vote_a.click(
        fn=lambda *args: vote('A', *args),
        inputs=[model_a_state, model_b_state, eval_prompt, score_a, score_b, judge_id],
        outputs=[action_buttons_row, model_name_a, model_name_b, send_btn, regenerate_button]
    )

    vote_b.click(
        fn=lambda *args: vote('B', *args),
        inputs=[model_a_state, model_b_state, eval_prompt, score_a, score_b, judge_id],
        outputs=[action_buttons_row, model_name_a, model_name_b, send_btn, regenerate_button]
    )

    vote_tie.click(
        fn=lambda *args: vote('Tie', *args),
        inputs=[model_a_state, model_b_state, eval_prompt, score_a, score_b, judge_id],
        outputs=[action_buttons_row, model_name_a, model_name_b, send_btn, regenerate_button]
    )

    # Update the send button handler to store the submitted inputs
    def submit_and_store(prompt, *variables):
        last_submission.value = {"prompt": prompt, "variables": variables}
        response_a, response_b, buttons_visible, regen_visible, model_a, model_b = submit_prompt(prompt, *variables)
        
        # Parse the responses
        score_a, critique_a = parse_model_response(response_a)
        score_b, critique_b = parse_model_response(response_b)
        
        return (
            score_a,
            critique_a,
            score_b,
            critique_b,
            buttons_visible,
            gr.update(visible=True),  # Changed from False to True to show regenerate button
            model_a,
            model_b,
            gr.update(value="*Model: Unknown*"),
            gr.update(value="*Model: Unknown*")
        )

    send_btn.click(
        fn=submit_and_store,
        inputs=[eval_prompt] + [var_input for _, var_input in variable_rows],
        outputs=[
            score_a,
            critique_a,
            score_b,
            critique_b,
            action_buttons_row,
            regenerate_button,
            model_a_state,
            model_b_state,
            model_name_a,    # Add model name outputs
            model_name_b
        ]
    )

    # Update the input change handlers to also disable regenerate button
    def handle_input_changes(prompt, *variables):
        """Enable send button and manage regenerate button based on input changes"""
        last_inputs = last_submission.value
        current_inputs = {"prompt": prompt, "variables": variables}
        inputs_changed = last_inputs != current_inputs
        return [
            gr.update(interactive=True),                    # send button always enabled
            gr.update(interactive=not inputs_changed)       # regenerate button disabled if inputs changed
        ]

    # Update the change handlers for prompt and variables
    eval_prompt.change(
        fn=handle_input_changes,
        inputs=[eval_prompt] + [var_input for _, var_input in variable_rows],
        outputs=[send_btn, regenerate_button]
    )

    for _, var_input in variable_rows:
        var_input.change(
            fn=handle_input_changes,
            inputs=[eval_prompt] + [var_input for _, var_input in variable_rows],
            outputs=[send_btn, regenerate_button]
        )

    # Update the leaderboard
    def refresh_leaderboard():
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
        stats = get_leaderboard_stats()
        return [gr.update(value=data), gr.update(value=stats)]

    refresh_button.click(
        fn=refresh_leaderboard,
        inputs=None,
        outputs=[leaderboard_table, stats_display]
    )

    # Add the load event at the very end, just before demo.launch()
    demo.load(
        fn=refresh_leaderboard,
        inputs=None,
        outputs=[leaderboard_table, stats_display]
    )

demo.launch()