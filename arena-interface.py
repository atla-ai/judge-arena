import json
import re
import random
from collections import defaultdict
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from gen_api_answer import get_model_response, parse_model_response
from db import add_vote, create_db_connection, get_votes
from utils import Vote
from common import (
    POLICY_CONTENT,
    ACKNOWLEDGEMENTS,
    DEFAULT_EVAL_PROMPT,
    DEFAULT_INPUT,
    DEFAULT_RESPONSE,
    CSS_STYLES,
    MAIN_TITLE,
    HOW_IT_WORKS,
    BATTLE_RULES,
    EVAL_DESCRIPTION,
    VOTING_HEADER,
)
from example_metrics import EXAMPLE_METRICS


# Model and ELO score data
DEFAULT_ELO = 1500  # Starting ELO for new models
K_FACTOR = 32  # Standard chess K-factor, adjust as needed
elo_scores = defaultdict(lambda: DEFAULT_ELO)
vote_counts = defaultdict(int)

db = create_db_connection()
votes_collection = get_votes(db)

current_time = datetime.now()


# Load the model_data from JSONL
def load_model_data():
    model_data = {}
    try:
        with open("data/models.jsonl", "r") as f:
            for line in f:
                model = json.loads(line)
                model_data[model["name"]] = {
                    "organization": model["organization"],
                    "license": model["license"],
                    "api_model": model["api_model"],
                }
    except FileNotFoundError:
        print("Warning: models.jsonl not found")
        return {}
    return model_data


model_data = load_model_data()

current_session_id = 0


def get_new_session_id():
    global current_session_id
    current_session_id += 1
    return f"user{current_session_id}"


def store_vote_data(prompt, response_a, response_b, model_a, model_b, winner, judge_id):
    vote = Vote(
        timestamp=datetime.now().isoformat(),
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        model_a=model_a,
        model_b=model_b,
        winner=winner,
        judge_id=judge_id,
    )
    add_vote(vote, db)


def parse_variables(prompt):
    # Extract variables enclosed in double curly braces
    variables = re.findall(r"{{(.*?)}}", prompt)
    # Remove duplicates while preserving order
    seen = set()
    variables = [
        x.strip() for x in variables if not (x.strip() in seen or seen.add(x.strip()))
    ]
    return variables


def get_final_prompt(eval_prompt, variable_values):
    # Replace variables in the eval prompt with their values
    for var, val in variable_values.items():
        eval_prompt = eval_prompt.replace("{{" + var + "}}", val)
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
            model_b,
            final_prompt,
        )
    except Exception as e:
        print(f"Error in submit_prompt: {str(e)}")
        return (
            "Error generating response",
            "Error generating response",
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            None,
            None,
        )


def vote(
    choice,
    model_a,
    model_b,
    final_prompt,
    score_a,
    critique_a,
    score_b,
    critique_b,
    judge_id,
):
    # Update ELO scores based on user choice
    elo_a = elo_scores[model_a]
    elo_b = elo_scores[model_b]

    # Calculate expected scores
    Ea = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    Eb = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

    # Assign actual scores
    if choice == "A":
        Sa, Sb = 1, 0
    elif choice == "B":
        Sa, Sb = 0, 1
    else:
        Sa, Sb = 0.5, 0.5

    # Update scores and vote counts
    elo_scores[model_a] += K_FACTOR * (Sa - Ea)
    elo_scores[model_b] += K_FACTOR * (Sb - Eb)
    vote_counts[model_a] += 1
    vote_counts[model_b] += 1

    # Format the full responses with score and critique
    response_a = f"""{score_a}

{critique_a}"""

    response_b = f"""{score_b}

{critique_b}"""

    # Store the vote data with the final prompt
    store_vote_data(
        final_prompt, response_a, response_b, model_a, model_b, choice, judge_id
    )

    # Return updates for UI components
    return [
        gr.update(visible=False),  # action_buttons_row
        gr.update(value=f"*Model: {model_a}*"),  # model_name_a
        gr.update(value=f"*Model: {model_b}*"),  # model_name_b
        gr.update(interactive=True),  # send_btn
        gr.update(visible=True, interactive=True),  # regenerate_button
    ]


def get_current_votes():
    """Get current votes from database."""
    return get_votes(db)


def get_leaderboard():
    """Generate leaderboard data using fresh votes from MongoDB."""
    # Get fresh voting data
    voting_data = get_current_votes()
    print(f"Fetched {len(voting_data)} votes from database")  # Debug log

    # Initialize dictionaries for tracking
    ratings = defaultdict(lambda: DEFAULT_ELO)
    matches = defaultdict(int)

    # Process each vote
    for vote in voting_data:
        try:
            model_a = vote.get("model_a")
            model_b = vote.get("model_b")
            winner = vote.get("winner")

            # Skip if models aren't in current model_data
            if (
                not all([model_a, model_b, winner])
                or model_a not in model_data
                or model_b not in model_data
            ):
                continue

            # Update match counts
            matches[model_a] += 1
            matches[model_b] += 1

            # Calculate ELO changes
            elo_a = ratings[model_a]
            elo_b = ratings[model_b]

            # Expected scores
            expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
            expected_b = 1 - expected_a

            # Actual scores
            score_a = 1 if winner == "A" else 0 if winner == "B" else 0.5
            score_b = 1 - score_a

            # Update ratings
            ratings[model_a] += K_FACTOR * (score_a - expected_a)
            ratings[model_b] += K_FACTOR * (score_b - expected_b)

        except Exception as e:
            print(f"Error processing vote: {e}")
            continue

    # Generate leaderboard data
    leaderboard = []
    for model in model_data.keys():
        votes = matches[model]
        elo = ratings[model]
        ci = 1.96 * (400 / (votes + 1) ** 0.5) if votes > 0 else 0
        data = {
            "Model": model,
            "ELO Score": f"{elo:.2f}",
            "95% CI": f"±{ci:.2f}",
            "# Votes": votes,
            "Organization": model_data[model]["organization"],
            "License": model_data[model]["license"],
        }
        leaderboard.append(data)

    # Sort leaderboard by ELO score in descending order
    leaderboard.sort(key=lambda x: float(x["ELO Score"]), reverse=True)

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
        score_a,  # score_a textbox
        critique_a,  # critique_a textbox
        score_b,  # score_b textbox
        critique_b,  # critique_b textbox
        gr.update(visible=True),  # action_buttons_row
        gr.update(value="*Model: Unknown*"),  # model_name_a
        gr.update(value="*Model: Unknown*"),  # model_name_b
        model1,  # model_a_state
        model2,  # model_b_state
    )


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
    """Generate leaderboard DataFrame using fresh votes from MongoDB."""
    # Get fresh voting data
    voting_data = get_current_votes()
    print(f"Found {len(voting_data)} votes in database")
    matches = defaultdict(int)

    # Process each vote chronologically
    for vote in voting_data:
        # Extract model names from the vote document
        try:
            model_a = vote.get("model_a")
            model_b = vote.get("model_b")
            winner = vote.get("winner")

            print(f"Processing vote: {model_a} vs {model_b}, winner: {winner}")

            # Skip if any required field is missing or models aren't in current model_data
            if not all([model_a, model_b, winner]):
                print(f"Missing required fields in vote: {vote}")
                continue

            if model_a not in model_data:
                print(f"Model A '{model_a}' not found in model_data")
                continue

            if model_b not in model_data:
                print(f"Model B '{model_b}' not found in model_data")
                continue

            # Update match counts
            matches[model_a] += 1
            matches[model_b] += 1
            print(
                f"Updated matches - {model_a}: {matches[model_a]}, {model_b}: {matches[model_b]}"
            )
        except Exception as e:
            print(f"Error processing vote: {e}")
            print(f"Problematic vote data: {vote}")
            continue


# Update the display_leaderboard function
def display_leaderboard():
    df = update_leaderboard()
    return gr.DataFrame(
        value=df,
        headers=["Model", "ELO", "95% CI", "Matches", "Organization", "License"],
        datatype=["str", "number", "str", "number", "str", "str", "str"],
        row_count=(len(df) + 1, "dynamic"),
    )


# Update the leaderboard table definition in the UI
leaderboard_table = gr.Dataframe(
    headers=["Model", "ELO", "95% CI", "Matches", "Organization", "License"],
    datatype=["str", "number", "str", "number", "str", "str", "str"],
)


def get_leaderboard_stats():
    """Get summary statistics for the leaderboard."""
    now = datetime.now(timezone.utc)
    total_votes = len(get_current_votes())
    total_models = len(model_data)
    last_updated = now.replace(minute=0, second=0, microsecond=0).strftime(
        "%B %d, %Y at %H:00 UTC"
    )

    return f"""
### Leaderboard Stats
- **Total Models**: {total_models}
- **Total Votes**: {total_votes}
- **Last Updated**: {last_updated}
"""


def set_example_metric(metric_name):
    if metric_name == "Custom":
        variables = parse_variables(DEFAULT_EVAL_PROMPT)
        variable_values = []
        for var in variables:
            if var == "input":
                variable_values.append(DEFAULT_INPUT)
            elif var == "response":
                variable_values.append(DEFAULT_RESPONSE)
            else:
                variable_values.append("")  # Default empty value
        # Pad variable_values to match the length of variable_rows
        while len(variable_values) < len(variable_rows):
            variable_values.append("")
        return [DEFAULT_EVAL_PROMPT] + variable_values

    metric_data = EXAMPLE_METRICS[metric_name]
    variables = parse_variables(metric_data["prompt"])
    variable_values = []
    for var in variables:
        value = metric_data.get(var, "")  # Default to empty string if not found
        variable_values.append(value)
    # Pad variable_values to match the length of variable_rows
    while len(variable_values) < len(variable_rows):
        variable_values.append("")
    return [metric_data["prompt"]] + variable_values


# Select random metric at startup
def get_random_metric():
    metrics = list(EXAMPLE_METRICS.keys())
    return set_example_metric(random.choice(metrics))


with gr.Blocks(theme="default", css=CSS_STYLES) as demo:
    judge_id = gr.State(get_new_session_id())
    gr.Markdown(MAIN_TITLE)
    gr.Markdown(HOW_IT_WORKS)

    with gr.Tabs():
        with gr.TabItem("Judge Arena"):

            with gr.Row():
                with gr.Column():
                    gr.Markdown(BATTLE_RULES)
                    gr.Markdown(EVAL_DESCRIPTION)

            # Add Example Metrics Section
            with gr.Accordion("Evaluator Prompt Templates", open=False):
                with gr.Row():
                    custom_btn = gr.Button("Custom", variant="secondary")
                    hallucination_btn = gr.Button("Hallucination")
                    precision_btn = gr.Button("Precision")
                    recall_btn = gr.Button("Recall")
                    coherence_btn = gr.Button("Logical coherence")
                    faithfulness_btn = gr.Button("Faithfulness")

            # Eval Prompt and Variables side by side
            with gr.Row():
                # Left column - Eval Prompt
                with gr.Column(scale=1):
                    eval_prompt = gr.TextArea(
                        label="Evaluator Prompt",
                        lines=1,
                        value=DEFAULT_EVAL_PROMPT,
                        placeholder="Type your eval prompt here... denote variables in {{curly brackets}} to be populated on the right.",
                        show_label=True,
                    )

                # Right column - Variable Mapping
                with gr.Column(scale=1):
                    gr.Markdown("### Sample to test the evaluator")
                    # Create inputs for up to 5 variables, with first two visible by default
                    variable_rows = []
                    for i in range(5):
                        initial_visibility = True if i < 2 else False
                        with gr.Group(visible=initial_visibility) as var_row:
                            # Set default labels for the first two inputs
                            default_label = (
                                "input" if i == 0 else "response" if i == 1 else ""
                            )
                            var_input = gr.Textbox(
                                container=True,
                                label=default_label,  # Add default label here
                            )
                            variable_rows.append((var_row, var_input))

            # Send button
            with gr.Row(elem_classes="send-button-row"):
                send_btn = gr.Button(
                    value="Test the evaluators", variant="primary", size="lg", scale=1
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
            regenerate_button = gr.Button(
                "Regenerate with different models", variant="secondary", visible=False
            )

            # Add spacing and acknowledgements at the bottom
            gr.Markdown(ACKNOWLEDGEMENTS)

        with gr.TabItem("Leaderboard"):
            stats_display = gr.Markdown()
            leaderboard_table = gr.Dataframe(
                headers=["Model", "ELO", "95% CI", "Matches", "Organization", "License"],
                datatype=["str", "number", "str", "number", "str", "str", "str"],
            )

        with gr.TabItem("Policy"):
            gr.Markdown(POLICY_CONTENT)

    # Define state variables for model tracking
    model_a_state = gr.State()
    model_b_state = gr.State()
    final_prompt_state = gr.State()

    # Update variable inputs based on the eval prompt
    def update_variables(eval_prompt):
        variables = parse_variables(eval_prompt)
        updates = []

        for i in range(len(variable_rows)):
            var_row, var_input = variable_rows[i]
            if i < len(variables):
                var_name = variables[i]
                # Set the number of lines based on the variable name
                if var_name == "response":
                    lines = 4  # Adjust this number as needed
                else:
                    lines = 1  # Default to single line for other variables
                updates.extend(
                    [
                        gr.update(visible=True),  # Show the variable row
                        gr.update(
                            label=var_name, visible=True, lines=lines
                        ),  # Update label and lines
                    ]
                )
            else:
                updates.extend(
                    [
                        gr.update(visible=False),  # Hide the variable row
                        gr.update(value="", visible=False),  # Clear value when hidden
                    ]
                )
        return updates

    eval_prompt.change(
        fn=update_variables,
        inputs=eval_prompt,
        outputs=[item for sublist in variable_rows for item in sublist],
    )

    # Regenerate button functionality
    regenerate_button.click(
        fn=regenerate_prompt,
        inputs=[model_a_state, model_b_state, eval_prompt]
        + [var_input for _, var_input in variable_rows],
        outputs=[
            score_a,
            critique_a,
            score_b,
            critique_b,
            action_buttons_row,
            model_name_a,
            model_name_b,
            model_a_state,
            model_b_state,
        ],
    )

    # Update model names after responses are generated
    def update_model_names(model_a, model_b):
        return gr.update(value=f"*Model: {model_a}*"), gr.update(
            value=f"*Model: {model_b}*"
        )

    # Store the last submitted prompt and variables for comparison
    last_submission = gr.State({})

    # Update the vote button click handlers
    vote_a.click(
        fn=lambda *args: vote("A", *args),
        inputs=[
            model_a_state,
            model_b_state,
            final_prompt_state,
            score_a,
            critique_a,
            score_b,
            critique_b,
            judge_id,
        ],
        outputs=[
            action_buttons_row,
            model_name_a,
            model_name_b,
            send_btn,
            regenerate_button,
        ],
    )

    vote_b.click(
        fn=lambda *args: vote("B", *args),
        inputs=[
            model_a_state,
            model_b_state,
            final_prompt_state,
            score_a,
            critique_a,
            score_b,
            critique_b,
            judge_id,
        ],
        outputs=[
            action_buttons_row,
            model_name_a,
            model_name_b,
            send_btn,
            regenerate_button,
        ],
    )

    vote_tie.click(
        fn=lambda *args: vote("Tie", *args),
        inputs=[
            model_a_state,
            model_b_state,
            final_prompt_state,
            score_a,
            critique_a,
            score_b,
            critique_b,
            judge_id,
        ],
        outputs=[
            action_buttons_row,
            model_name_a,
            model_name_b,
            send_btn,
            regenerate_button,
        ],
    )

    # Update the send button handler to store the submitted inputs
    def submit_and_store(prompt, *variables):
        # Create a copy of the current submission
        current_submission = {"prompt": prompt, "variables": variables}

        # Get the responses
        (
            response_a,
            response_b,
            buttons_visible,
            regen_visible,
            model_a,
            model_b,
            final_prompt,
        ) = submit_prompt(prompt, *variables)

        # Parse the responses
        score_a, critique_a = parse_model_response(response_a)
        score_b, critique_b = parse_model_response(response_b)

        # Update the last_submission state with the current values
        last_submission.value = current_submission

        return (
            score_a,
            critique_a,
            score_b,
            critique_b,
            buttons_visible,
            gr.update(
                visible=True, interactive=True
            ),  # Show and enable regenerate button
            model_a,
            model_b,
            final_prompt,  # Add final_prompt to state
            gr.update(value="*Model: Unknown*"),
            gr.update(value="*Model: Unknown*"),
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
            final_prompt_state,  # Add final_prompt_state to outputs
            model_name_a,
            model_name_b,
        ],
    )

    # Update the input change handlers to also disable regenerate button
    def handle_input_changes(prompt, *variables):
        """Enable send button and manage regenerate button based on input changes"""
        last_inputs = last_submission.value
        current_inputs = {"prompt": prompt, "variables": variables}
        inputs_changed = last_inputs != current_inputs
        return [
            gr.update(interactive=True),  # send button always enabled
            gr.update(
                interactive=not inputs_changed
            ),  # regenerate button disabled if inputs changed
        ]

    # Update the change handlers for prompt and variables
    eval_prompt.change(
        fn=handle_input_changes,
        inputs=[eval_prompt] + [var_input for _, var_input in variable_rows],
        outputs=[send_btn, regenerate_button],
    )

    for _, var_input in variable_rows:
        var_input.change(
            fn=handle_input_changes,
            inputs=[eval_prompt] + [var_input for _, var_input in variable_rows],
            outputs=[send_btn, regenerate_button],
        )

    # Update the leaderboard
    def refresh_leaderboard():
        """Refresh the leaderboard data and stats."""
        leaderboard = get_leaderboard()
        data = [
            [
                entry["Model"],
                float(entry["ELO Score"]),
                entry["95% CI"],
                entry["# Votes"],
                entry["Organization"],
                entry["License"],
            ]
            for entry in leaderboard
        ]
        stats = get_leaderboard_stats()
        return [gr.update(value=data), gr.update(value=stats)]

    # Add the load event at the very end, just before demo.launch()
    demo.load(
        fn=refresh_leaderboard, inputs=None, outputs=[leaderboard_table, stats_display]
    )

    # Add click handlers for metric buttons
    outputs_list = [eval_prompt] + [var_input for _, var_input in variable_rows]

    custom_btn.click(fn=lambda: set_example_metric("Custom"), outputs=outputs_list)

    hallucination_btn.click(
        fn=lambda: set_example_metric("Hallucination"), outputs=outputs_list
    )

    precision_btn.click(fn=lambda: set_example_metric("Precision"), outputs=outputs_list)

    recall_btn.click(fn=lambda: set_example_metric("Recall"), outputs=outputs_list)

    coherence_btn.click(
        fn=lambda: set_example_metric("Logical_Coherence"), outputs=outputs_list
    )

    faithfulness_btn.click(
        fn=lambda: set_example_metric("Faithfulness"), outputs=outputs_list
    )

    # Set default metric at startup
    demo.load(
        fn=lambda: set_example_metric("Custom"),
        outputs=[eval_prompt] + [var_input for _, var_input in variable_rows],
    )

if __name__ == "__main__":
    demo.launch()
