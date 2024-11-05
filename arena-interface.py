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
    'GPT-4o': {
        'organization': 'OpenAI',
        'license': 'Proprietary',
        'api_model': 'gpt-4o'
    },
    'GPT-4 Turbo': {
        'organization': 'OpenAI',
        'license': 'Proprietary',
        'api_model': 'gpt-4-turbo'
    },
    'GPT-3.5 Turbo': {
        'organization': 'OpenAI',
        'license': 'Proprietary',
        'api_model': 'gpt-3.5-turbo'
    },
    'Claude 3 Haiku': {
        'organization': 'Anthropic',
        'license': 'Proprietary',
        'api_model': 'claude-3-haiku-20240307'
    },
    'Claude 3 Sonnet': {
        'organization': 'Anthropic',
        'license': 'Proprietary',
        'api_model': 'claude-3-sonnet-20240229'
    },
    'Claude 3 Opus': {
        'organization': 'Anthropic',
        'license': 'Proprietary',
        'api_model': 'claude-3-opus-20240229'
    },
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

# Add this near the top with other constants
SYSTEM_PROMPT = """Please act as an impartial judge and evaluate based on the user's instruction. Your output format should be a JSON as follows: {{"feedback": "(write a feedback for the evaluation criteria)", "result": "(a score based on the evaluation criteria)"}}"""

def get_openai_response(model_name, prompt):
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error with Anthropic model {model_name}: {str(e)}"

def get_model_response(model_name, prompt):
    model_info = model_data.get(model_name)
    if not model_info:
        return "Model not found or unsupported."
    
    api_model = model_info['api_model']
    organization = model_info['organization']
    
    try:
        if organization == 'OpenAI':
            return get_openai_response(api_model, prompt)
        elif organization == 'Anthropic':
            return get_anthropic_response(api_model, prompt)
        elif organization == 'Meta':
            return get_together_response(api_model, prompt)
        else:
            return "Organization not supported."
    except Exception as e:
        return f"Error with {organization} model {model_name}: {str(e)}"

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
    
    response_a = get_model_response(model1, final_prompt)
    response_b = get_model_response(model2, final_prompt)

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

def get_together_response(model_name, prompt):
    try:
        response = together_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with Together model {model_name}: {str(e)}"

def parse_model_response(response):
    try:
        # Parse JSON response
        data = json.loads(response)
        return data.get('result', 'N/A'), data.get('feedback', 'N/A')
    except:
        # If JSON parsing fails, return original response
        return 'Error', response

with gr.Blocks(theme='default', css="""
    .prompt-row {
        align-items: flex-start !important;
    }
    .send-button-row {
        display: flex;
        justify-content: flex-end;
        margin-top: 8px;
    }
""") as demo:
    judge_id = gr.State(get_new_session_id())
    gr.Markdown("# Judge Arena")
    gr.Markdown("*Free LLM Evals to test your GenAI application.*")
    
    with gr.Tabs():
        with gr.TabItem("Judge Arena"):
            # Add introduction section with side-by-side rules and scoring
            gr.Markdown("""
            # How the Arena Works:

            ## Test two anonymous LLM judges side by side
            Try out different eval metrics - from simple hallucination detection to qualitative interpretations
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ## Battle Rules:
                    - Both AIs stay anonymous - if either reveals its identity, the duel is void
                    - Evaluate anything: coding, analysis, creative writing, math, or general knowledge
                    """)
                with gr.Column():
                    gr.Markdown("""
                    ## Scoring System:
                    - Choose the LLM judge that most aligned with your choice as a human
                    - If both score the same - choose the critique that you prefer more!
                    - Your votes shape our real-time leaderboard
                    - Every arena match adds to the growing stats
                    """)
            
            # Add divider heading
            gr.Markdown("""
            # Start Voting Now
            """)
            
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
            # Eval Prompt and Variables below
            with gr.Row(elem_classes="prompt-row"):
                eval_prompt = gr.TextArea(
                    label="Eval Prompt", 
                    lines=1, 
                    value="""You are assessing a chat bot response to a user's input based on the helpfulness of the response.\n                    

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

[END DATA]""",
                    placeholder="Type your eval prompt here... denote variables like a ground truth response with {{variable}} to be populated below.",
                    show_label=True,
                    scale=8
                )
            with gr.Row(elem_classes="send-button-row"):
                send_btn = gr.Button(
                    value="Send",
                    variant="primary",
                    size="lg",
                    scale=1  # Make button larger
                )
            gr.Markdown("### Variable Mapping")
            # Create inputs for up to 5 variables, with first two visible by default
            variable_rows = []
            for i in range(5):
                # Set initial visibility True for first two rows (input and response)
                initial_visibility = True if i < 2 else False
                with gr.Row(visible=initial_visibility) as var_row:
                    with gr.Column(scale=0.2, min_width=80):
                        # Set initial labels for input and response
                        initial_label = "**input:**" if i == 0 else "**response:**" if i == 1 else "Variable"
                        var_label = gr.Markdown(initial_label)
                    with gr.Column(scale=1):
                        # Set initial values for input and response
                        initial_value = "Hello! Can you tell me the weather today?" if i == 0 else \
                                      "Hi there! It is 27 degrees Celsius today. Would you like the weather for the week ahead?" if i == 1 else ""
                        var_input = gr.Textbox(label="", container=False, value=initial_value)
                    variable_rows.append((var_row, var_label, var_input))
        with gr.TabItem("Leaderboard"):
            refresh_button = gr.Button("Refresh")
            leaderboard_table = gr.Dataframe(
                headers=['Model', 'ELO', '95% CI', 'Matches', 'Organization', 'License'],
                datatype=['str', 'number', 'str', 'number', 'str', 'str']
            )

        with gr.TabItem("Policy"):
            gr.Markdown("""
                # About Atla
                *Atla is an applied research organisation that trains models as evaluators to capture human preferences. Developers use Atla's models to run fast and accurate evaluations, so they can ship quickly and with confidence.*

                # Our Policy
                *(Add your policy information here)*

                # Frequently Asked Questions
                *(Add your FAQs here)*

                # Contact Us
                *(Add your contact information here)*
            """)

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

    def handle_input_changes(prompt, *variables):
        """Enable send button and disable regenerate button if inputs have changed"""
        last_inputs = last_submission.value
        current_inputs = {"prompt": prompt, "variables": variables}
        inputs_changed = last_inputs != current_inputs
        return [
            gr.update(interactive=True),  # Always keep send button enabled
            gr.update(visible=False)      # Hide regenerate button when inputs change
        ]

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
            gr.update(visible=False),  # Hide regenerate button on new submission
            model_a,
            model_b,
            gr.update(value="*Model: Unknown*"),
            gr.update(value="*Model: Unknown*")
        )

    send_btn.click(
        fn=submit_and_store,
        inputs=[eval_prompt] + [var_input for _, _, var_input in variable_rows],
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
        """Enable send button and disable regenerate button if inputs have changed"""
        last_inputs = last_submission.value
        current_inputs = {"prompt": prompt, "variables": variables}
        inputs_changed = last_inputs != current_inputs
        return [
            gr.update(interactive=inputs_changed),           # send button
            gr.update(interactive=not inputs_changed)        # regenerate button
        ]

    # Update the change handlers for prompt and variables
    eval_prompt.change(
        fn=handle_input_changes,
        inputs=[eval_prompt] + [var_input for _, _, var_input in variable_rows],
        outputs=[send_btn, regenerate_button]
    )

    for _, _, var_input in variable_rows:
        var_input.change(
            fn=handle_input_changes,
            inputs=[eval_prompt] + [var_input for _, _, var_input in variable_rows],
            outputs=[send_btn, regenerate_button]
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


