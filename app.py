import json
import re
import random
from collections import defaultdict
from datetime import datetime, timezone
import hashlib

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from gen_api_answer import get_model_response, parse_model_response, get_random_human_ai_pair
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
DEFAULT_ELO = 1200  # Starting ELO for new models
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


def get_ip(request: gr.Request) -> str:
    """Get and hash the IP address from the request."""
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    elif "x-forwarded-for" in request.headers:
        ip = request.headers["x-forwarded-for"]
        if "," in ip:
            ip = ip.split(",")[0]
    else:
        ip = request.client.host
    
    # Hash the IP address for privacy
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def vote(
    choice,
    model_a,
    model_b,
    final_prompt,
    score_a,
    critique_a,
    score_b,
    critique_b,
    request: gr.Request,
):
    # Get hashed IP as judge_id
    judge_id = get_ip(request)
    
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
        gr.update(visible=False),  # vote_a
        gr.update(visible=False),  # vote_b
        gr.update(visible=False),  # tie_button_row
        gr.update(value=f"*Model: {model_a}*"),  # model_name_a
        gr.update(value=f"*Model: {model_b}*"),  # model_name_b
        gr.update(interactive=True, value="Run the evaluators", variant="primary"),  # send_btn
        gr.update(visible=True),  # spacing_div
    ]


def get_current_votes():
    """Get current votes from database."""
    return get_votes(db)


def get_leaderboard(show_preliminary=True):
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
        # Skip models with < 500 votes if show_preliminary is False
        if not show_preliminary and votes < 500:
            continue
            
        elo = ratings[model]
        ci = 1.96 * (400 / (votes + 1) ** 0.5) if votes > 0 else 0
        data = {
            "Model": model,
            "ELO Score": f"{int(elo)}",
            "95% CI": f"±{int(ci)}",
            "# Votes": votes,
            "Organization": model_data[model]["organization"],
            "License": model_data[model]["license"],
        }
        leaderboard.append(data)

    # Sort leaderboard by ELO score in descending order
    leaderboard.sort(key=lambda x: float(x["ELO Score"]), reverse=True)

    return leaderboard


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


#def set_example_metric(metric_name):
#    if metric_name == "Custom":
#        variables = parse_variables(DEFAULT_EVAL_PROMPT)
#        variable_values = []
#        for var in variables:
#            if var == "input":
#                variable_values.append(DEFAULT_INPUT)
#            elif var == "response":
#                variable_values.append(DEFAULT_RESPONSE)
#            else:
#                variable_values.append("")  # Default empty value
        # Pad variable_values to match the length of variable_rows
#        while len(variable_values) < len(variable_rows):
#            variable_values.append("")
#        return [DEFAULT_EVAL_PROMPT] + variable_values

#    metric_data = EXAMPLE_METRICS[metric_name]
#    variables = parse_variables(metric_data["prompt"])
#    variable_values = []
#    for var in variables:
#        value = metric_data.get(var, "")  # Default to empty string if not found
#        variable_values.append(value)
    # Pad variable_values to match the length of variable_rows
#    while len(variable_values) < len(variable_rows):
#        variable_values.append("")
#    return [metric_data["prompt"]] + variable_values


# Select random metric at startup
#  def get_random_metric():
#    metrics = list(EXAMPLE_METRICS.keys())
#    return set_example_metric(random.choice(metrics))


def populate_random_example(request: gr.Request):
    """Generate a random human-AI conversation example."""
    human_msg, ai_msg = get_random_human_ai_pair()
    return [
        gr.update(value=human_msg),
        gr.update(value=ai_msg)
    ]


with gr.Blocks(theme="default", css=CSS_STYLES) as demo:
    gr.Markdown(MAIN_TITLE)
    gr.Markdown(HOW_IT_WORKS)
    
    # Hidden eval prompt that will always contain DEFAULT_EVAL_PROMPT
    eval_prompt = gr.Textbox(
        value=DEFAULT_EVAL_PROMPT,
        visible=False
    )

    with gr.Tabs():
        with gr.TabItem("Judge Arena"):
            random_btn = gr.Button("🎲", scale=0)
            with gr.Row():
                # Left side - Input section
                with gr.Column(scale=1):
                    with gr.Group():
                        human_input = gr.TextArea(
                            label="👩 Human Input",
                            lines=13,
                            placeholder="Enter the human message here..."
                        )
                        
                        ai_response = gr.TextArea(
                            label="🤖 AI Response", 
                            lines=13,
                            placeholder="Enter the AI response here..."
                        )
                        
                        send_btn = gr.Button(
                            value="Run the evaluators",
                            variant="primary",
                            size="lg"
                        )

                # Right side - Model outputs
                with gr.Column(scale=1):
                    gr.Markdown("### 👩‍⚖️ Judge A")
                    with gr.Group():
                        model_name_a = gr.Markdown("*Model: Hidden*")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):  # Fixed narrow width for score
                                score_a = gr.Textbox(label="Score", lines=6, interactive=False)
                                vote_a = gr.Button("Vote A", variant="primary", visible=False)
                            with gr.Column(scale=9, min_width=400):  # Wider width for critique
                                critique_a = gr.TextArea(label="Critique", lines=8, interactive=False)
                
                    # Spacing div that's visible only when tie button is hidden
                    spacing_div = gr.HTML('<div style="height: 42px;"></div>', visible=True, elem_id="spacing-div")
                
                    # Tie button row
                    with gr.Row(visible=False) as tie_button_row:
                        with gr.Column():
                            vote_tie = gr.Button("Tie", variant="secondary")
                    
                
                    gr.Markdown("### 🧑‍⚖️ Judge B")
                    with gr.Group():
                        model_name_b = gr.Markdown("*Model: Hidden*")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):  # Fixed narrow width for score
                                score_b = gr.Textbox(label="Score", lines=6, interactive=False)
                                vote_b = gr.Button("Vote B", variant="primary", visible=False)
                            with gr.Column(scale=9, min_width=400):  # Wider width for critique
                                critique_b = gr.TextArea(label="Critique", lines=8, interactive=False)
                    # Place Vote B button directly under Judge B
                
            gr.Markdown("<br>")

            # Add spacing and acknowledgements at the bottom
            gr.Markdown(ACKNOWLEDGEMENTS)

        with gr.TabItem("Leaderboard"):
            with gr.Row():
                with gr.Column(scale=1):
                    show_preliminary = gr.Checkbox(
                        label="Reveal preliminary results",
                        value=True,  # Checked by default
                        info="Show all models, including models with less few human ratings (< 500 votes)",
                        interactive=True
                    )
            stats_display = gr.Markdown()
            leaderboard_table = gr.Dataframe(
                headers=["Model", "ELO", "95% CI", "Matches", "Organization", "License"],
                datatype=["str", "number", "str", "number", "str", "str", "str"],
            )

            # Update refresh_leaderboard to use the checkbox value
            def refresh_leaderboard(show_preliminary):
                """Refresh the leaderboard data and stats."""
                leaderboard = get_leaderboard(show_preliminary)
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

            # Add change handler for checkbox
            show_preliminary.change(
                fn=refresh_leaderboard,
                inputs=[show_preliminary],
                outputs=[leaderboard_table, stats_display]
            )

            # Update the load event
            demo.load(
                fn=refresh_leaderboard,
                inputs=[show_preliminary],
                outputs=[leaderboard_table, stats_display]
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

    #eval_prompt.change(
    #    fn=update_variables,
    #    inputs=eval_prompt,
    #    outputs=[item for sublist in variable_rows for item in sublist],
    #)

    # Regenerate button functionality
    #regenerate_button.click(
    #    fn=regenerate_prompt,
    #    inputs=[model_a_state, model_b_state, eval_prompt, human_input, ai_response],
    #    outputs=[
    #        score_a,
    #        critique_a,
    #        score_b,
    #        critique_b,
    #        vote_a,
    #        vote_b,
    #        tie_button_row,
    #        model_name_a,
    #        model_name_b,
    #        model_a_state,
    #        model_b_state,
    #    ],
    #)

    # Update model names after responses are generated
    def update_model_names(model_a, model_b):
        return gr.update(value=f"*Model: {model_a}*"), gr.update(
            value=f"*Model: {model_b}*"
        )

    # Store the last submitted prompt and variables for comparison
    last_submission = gr.State({})

    # Update the vote button click handlers
    vote_a.click(
        fn=vote,
        inputs=[
            gr.State("A"),  # Choice
            model_a_state,
            model_b_state,
            final_prompt_state,
            score_a,
            critique_a,
            score_b,
            critique_b,
        ],
        outputs=[
            vote_a,
            vote_b,
            tie_button_row,
            model_name_a,
            model_name_b,
            send_btn,
            spacing_div,
        ],
    )

    vote_b.click(
        fn=vote,
        inputs=[
            gr.State("B"),  # Choice
            model_a_state,
            model_b_state,
            final_prompt_state,
            score_a,
            critique_a,
            score_b,
            critique_b,
        ],
        outputs=[
            vote_a,
            vote_b,
            tie_button_row,
            model_name_a,
            model_name_b,
            send_btn,
            spacing_div,
        ],
    )

    vote_tie.click(
        fn=vote,
        inputs=[
            gr.State("Tie"),  # Choice
            model_a_state,
            model_b_state,
            final_prompt_state,
            score_a,
            critique_a,
            score_b,
            critique_b,
        ],
        outputs=[
            vote_a,
            vote_b,
            tie_button_row,
            model_name_a,
            model_name_b,
            send_btn,
            spacing_div,
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

        # Format scores with "/ 5"
        score_a = f"{score_a} / 5"
        score_b = f"{score_b} / 5"

        # Update the last_submission state with the current values
        last_submission.value = current_submission

        return (
            score_a,
            critique_a,
            score_b,
            critique_b,
            gr.update(visible=True),  # vote_a
            gr.update(visible=True),  # vote_b
            gr.update(visible=True),  # tie_button_row
            model_a,
            model_b,
            final_prompt,  # Add final_prompt to state
            gr.update(value="*Model: Hidden*"),
            gr.update(value="*Model: Hidden*"),
            # Change the button to "Regenerate" mode after evaluation
            gr.update(
                value="Regenerate with different models",
                variant="secondary",
                interactive=True
            ),
            gr.update(visible=False),  # spacing_div
        )

    send_btn.click(
        fn=submit_and_store,
        inputs=[eval_prompt, human_input, ai_response],
        outputs=[
            score_a,
            critique_a,
            score_b,
            critique_b,
            vote_a,
            vote_b,
            tie_button_row,
            model_a_state,
            model_b_state,
            final_prompt_state,
            model_name_a,
            model_name_b,
            send_btn,
            spacing_div,
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
    #eval_prompt.change(
    #    fn=handle_input_changes,
    #    inputs=[eval_prompt] + [var_input for _, var_input in variable_rows],
    #    outputs=[send_btn, regenerate_button],
    #)

    # for _, var_input in variable_rows:
    #    var_input.change(
    #        fn=handle_input_changes,
    #        inputs=[eval_prompt] + [var_input for _, var_input in variable_rows],
    #        outputs=[send_btn, regenerate_button],
    #    )

    # Add click handlers for metric buttons
    #outputs_list = [eval_prompt] + [var_input for _, var_input in variable_rows]

    #custom_btn.click(fn=lambda: set_example_metric("Custom"), outputs=outputs_list)

    #hallucination_btn.click(
    #    fn=lambda: set_example_metric("Hallucination"), outputs=outputs_list
    #)

    #precision_btn.click(fn=lambda: set_example_metric("Precision"), outputs=outputs_list)

    #recall_btn.click(fn=lambda: set_example_metric("Recall"), outputs=outputs_list)

    #coherence_btn.click(
    #    fn=lambda: set_example_metric("Logical_Coherence"), outputs=outputs_list
    #)

    #faithfulness_btn.click(
    #    fn=lambda: set_example_metric("Faithfulness"), outputs=outputs_list
    #)

    # Set default metric at startup
    demo.load(
        #fn=lambda: set_example_metric("Hallucination"),
        #outputs=[eval_prompt] + [var_input for _, var_input in variable_rows],
    )

    # Add random button handler
    random_btn.click(
        fn=populate_random_example,
        inputs=[],
        outputs=[human_input, ai_response]
    )

    # Add new input change handlers
    def handle_input_change():
        return gr.update(value="Run the evaluators", variant="primary")

    # Update the change handlers for inputs
    human_input.change(
        fn=handle_input_change,
        inputs=[],
        outputs=[send_btn]
    )

    ai_response.change(
        fn=handle_input_change,
        inputs=[],
        outputs=[send_btn]
    )

    # Update the demo.load to include the random example population
    demo.load(
        fn=populate_random_example,
        inputs=[],
        outputs=[human_input, ai_response]
    )

if __name__ == "__main__":
    demo.launch()