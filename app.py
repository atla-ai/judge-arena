import json
import re
import random
from collections import defaultdict
from datetime import datetime
import hashlib

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from gen_api_answer import (
    get_model_response, 
    parse_model_response,
    prometheus_parse_model_response,
    atla_parse_model_response
)

from random_sample_generation import (
    get_random_human_ai_pair,
    get_random_human_ai_ground_truth_pair,
    generate_ai_response
)   
from db import add_vote, create_db_connection, get_votes
from utils import Vote
from common import (
    POLICY_CONTENT,
    ACKNOWLEDGEMENTS,
    CSS_STYLES,
    MAIN_TITLE,
    HOW_IT_WORKS,
)
from prompts import (
    DEFAULT_EVAL_PROMPT,
    DEFAULT_EVAL_PROMPT_EDITABLE,
    FIXED_EVAL_SUFFIX,
    DEFAULT_EVAL_CRITERIA,
    DEFAULT_SCORE_1,
    DEFAULT_SCORE_2,
    DEFAULT_SCORE_3,
    DEFAULT_SCORE_4,
    DEFAULT_SCORE_5,
)
from leaderboard import (
    get_leaderboard,
    get_leaderboard_stats,
    get_model_rankings,
    DEFAULT_ELO,
    K_FACTOR
)


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
    prompt_value = prompt.value if hasattr(prompt, 'value') else prompt
    
    vote = Vote(
        timestamp=datetime.now().isoformat(),
        prompt=prompt_value,
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


def get_vote_message(choice: str, model_a: str, model_b: str) -> tuple[str, str]:
    """Generate appropriate message based on vote and model rankings.
    Returns (title, message) tuple."""
    # Get current rankings
    voting_data = get_current_votes()
    leaderboard = get_leaderboard(model_data, voting_data, show_preliminary=True)
    rankings = get_model_rankings(leaderboard)
    pos_a = rankings.get(model_a, 0)
    pos_b = rankings.get(model_b, 0)
    
    if choice == "Tie":
        return "It's a tie!", "Keep voting responsibly 🤗"
    
    # Check if vote aligns with leaderboard
    if (choice == "A" and pos_a < pos_b) or (choice == "B" and pos_b < pos_a):
        return "The favourite wins!", "Keep voting responsibly 🤗"
    else:
        return "The underdog wins!", "Keep voting responsibly 🤗"


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
    
    # Get model positions for display
    voting_data = get_current_votes()
    leaderboard = get_leaderboard(model_data, voting_data, show_preliminary=True)
    rankings = get_model_rankings(leaderboard)
    pos_a = rankings.get(model_a, 0)
    pos_b = rankings.get(model_b, 0)
    
    # Format model names with positions and win/loss indicators
    if choice == "Tie":
        model_a_display = f"*Model: {model_a} (Position #{pos_a})*"
        model_b_display = f"*Model: {model_b} (Position #{pos_b})*"
    else:
        winner = model_a if choice == "A" else model_b
        loser = model_b if choice == "A" else model_a
        winner_pos = pos_a if choice == "A" else pos_b
        loser_pos = pos_b if choice == "A" else pos_a
        
        model_a_display = f"*Model: {model_a} {'✅' if choice == 'A' else '❌'} (Position #{pos_a})*"
        model_b_display = f"*Model: {model_b} {'✅' if choice == 'B' else '❌'} (Position #{pos_b})*"
    
    # Generate vote message
    title, message = get_vote_message(choice, model_a, model_b)
    
    return [
        gr.update(interactive=False, variant="primary" if choice == "A" else "secondary"),  # vote_a
        gr.update(interactive=False, variant="primary" if choice == "B" else "secondary"),  # vote_b
        gr.update(interactive=False, variant="primary" if choice == "Tie" else "secondary"),  # vote_tie
        gr.update(value=model_a_display),  # model_name_a
        gr.update(value=model_b_display),  # model_name_b
        gr.update(interactive=True, value="Regenerate judges", variant="secondary"),  # send_btn
        gr.update(value="🎲 New round", variant="primary"),  # random_btn
        gr.Info(message, title=title),  # success message
    ]


def get_current_votes():
    """Get current votes from database."""
    return get_votes(db)


# Update the refresh_leaderboard function
def refresh_leaderboard(show_preliminary):
    """Refresh the leaderboard data and stats."""
    voting_data = get_current_votes()
    leaderboard = get_leaderboard(model_data, voting_data, show_preliminary)
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
    stats = get_leaderboard_stats(model_data, voting_data)
    return [gr.update(value=data), gr.update(value=stats)]


# Update the leaderboard table definition in the UI
leaderboard_table = gr.Dataframe(
    headers=["Model", "ELO", "95% CI", "Matches", "Organization", "License"],
    datatype=["str", "number", "str", "number", "str", "str", "str"],
)


def populate_random_example(request: gr.Request, compatible_mode: bool):
    """Generate a random human-AI conversation example and reset judge outputs."""
    if compatible_mode:
        # Generate all three components when compatible mode is enabled
        human_msg, ai_msg, ground_truth_msg = get_random_human_ai_ground_truth_pair()
    else:
        # Generate only human and AI messages when compatible mode is disabled
        human_msg, ai_msg = get_random_human_ai_pair()
        ground_truth_msg = ""
    
    return [
        gr.update(value=human_msg),
        gr.update(value=ai_msg),
        gr.update(value="🎲", variant="secondary"),  # Reset random button appearance
        gr.update(value=""),  # Clear score A
        gr.update(value=""),  # Clear critique A
        gr.update(value=""),  # Clear score B
        gr.update(value=""),  # Clear critique B
        gr.update(interactive=False, variant="primary"),  # Reset vote A
        gr.update(interactive=False, variant="primary"),  # Reset vote B
        gr.update(interactive=False, variant="primary"),  # Reset vote tie
        gr.update(value="*Model: Hidden*"),  # Reset model name A
        gr.update(value="*Model: Hidden*"),  # Reset model name B
        gr.update(value=ground_truth_msg, visible=compatible_mode),  # Set ground truth and visibility
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
            with gr.Row():
                # Left side - Input section
                with gr.Column(scale=1):
                    with gr.Group():
                        human_input = gr.TextArea(
                            label="👩 User Input",
                            lines=10,
                            placeholder="Enter the human message here..."
                        )
                        with gr.Row():
                            generate_btn = gr.Button(
                                "Generate AI Response",
                                size="sm",
                                interactive=False
                            )
                        
                        ai_response = gr.TextArea(
                            label="🤖 AI Response", 
                            lines=15,
                            placeholder="Enter the AI response here..."
                        )
                        
                        # Ground truth response (initially hidden)
                        ground_truth = gr.TextArea(
                            label="🎯 Ground truth response",
                            lines=12,
                            placeholder="Enter the ground truth response here...",
                            visible=False
                        )
                        
                    with gr.Row():
                        random_btn = gr.Button("🎲", scale=2)
                        send_btn = gr.Button(
                            value="Run judges",
                            variant="primary",
                            size="lg",
                            scale=8
                        )

                # Right side - Model outputs
                with gr.Column(scale=1):
                    gr.Markdown("### 👩‍⚖️ Judge A")
                    with gr.Group():
                        model_name_a = gr.Markdown("*Model: Hidden*")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):  # Fixed narrow width for score
                                score_a = gr.Textbox(label="Score", lines=6, interactive=False)
                                vote_a = gr.Button("Vote A", variant="primary", interactive=False)
                            with gr.Column(scale=9, min_width=400):  # Wider width for critique
                                critique_a = gr.TextArea(label="Critique", lines=8, interactive=False)
                
                    # Tie button row
                    with gr.Row() as tie_button_row:
                        with gr.Column():
                            vote_tie = gr.Button("Tie", variant="primary", interactive=False)
                    
                
                    gr.Markdown("### 🧑‍⚖️ Judge B")
                    with gr.Group():
                        model_name_b = gr.Markdown("*Model: Hidden*")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):  # Fixed narrow width for score
                                score_b = gr.Textbox(label="Score", lines=6, interactive=False)
                                vote_b = gr.Button("Vote B", variant="primary", interactive=False)
                            with gr.Column(scale=9, min_width=400):  # Wider width for critique
                                critique_b = gr.TextArea(label="Critique", lines=8, interactive=False)
                        # Place Vote B button directly under Judge B
                
            gr.Markdown("<br>")
            

            # Replace the "Edit Judge Prompt" Accordion section with:
            with gr.Accordion("📝 Edit Judge Prompt", open=False) as prompt_accordion:
                gr.Markdown("<br>")
                use_reference_toggle = gr.Checkbox(
                    label="Use a reference response",
                    value=False
                )
                
                # Hide the default prompt editor
                with gr.Column(visible=False) as default_prompt_editor:
                    eval_prompt_editable = gr.TextArea(
                        value=DEFAULT_EVAL_PROMPT_EDITABLE,
                        label="Evaluation Criteria",
                        lines=12
                    )

                    with gr.Row(visible=False) as edit_buttons_row:
                        cancel_prompt_btn = gr.Button("Cancel")
                        save_prompt_btn = gr.Button("Save", variant="primary")
                    gr.Markdown("*The sample being evaluated is always appended as:*")
                    gr.Markdown(f"```{FIXED_EVAL_SUFFIX}")
                
                # Show the compatible mode editor
                with gr.Column(visible=True) as compatible_prompt_editor:
                    with gr.Row():
                        # Left column - Evaluation Criteria
                        with gr.Column(scale=1):
                            eval_criteria_text = gr.TextArea(
                                label="Evaluation Criteria",
                                lines=12,
                                value=DEFAULT_EVAL_CRITERIA,
                                placeholder="Enter the evaluation criteria..."
                            )
                            prometheus_reference = gr.Markdown(
                                "<br> *By default, we use the Prometheus absolute grading prompt template - see [here](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0).*",
                                visible=True 
                            )
                        
                        # Right column - Score Descriptions
                        with gr.Column(scale=1):
                            score1_description = gr.TextArea(
                                label="Score 1",
                                value=DEFAULT_SCORE_1,
                                placeholder="Description for score 1",
                                lines=2
                            )
                            score2_description = gr.TextArea(
                                label="Score 2", 
                                value=DEFAULT_SCORE_2,
                                placeholder="Description for score 2",
                                lines=2
                            )
                            score3_description = gr.TextArea(
                                label="Score 3",
                                value=DEFAULT_SCORE_3,
                                placeholder="Description for score 3",
                                lines=2
                            )
                            score4_description = gr.TextArea(
                                label="Score 4",
                                value=DEFAULT_SCORE_4,
                                placeholder="Description for score 4",
                                lines=2
                            )
                            score5_description = gr.TextArea(
                                label="Score 5",
                                value=DEFAULT_SCORE_5,
                                placeholder="Description for score 5",
                                lines=2
                            )

                    # Add save/cancel buttons for compatible mode
                    with gr.Row(visible=False) as compatible_edit_buttons_row:
                        compatible_cancel_btn = gr.Button("Cancel")
                        compatible_save_btn = gr.Button("Save", variant="primary")

        with gr.TabItem("Leaderboard"):
            with gr.Row():
                with gr.Column(scale=1):
                    show_preliminary = gr.Checkbox(
                        label="Reveal preliminary results",
                        value=True,  # Checked by default
                        info="Show all models, including models with less human ratings (< 300 votes)",
                        interactive=True
                    )
            stats_display = gr.Markdown()
            leaderboard_table = gr.Dataframe(
                headers=["Model", "ELO", "95% CI", "Matches", "Organization", "License"],
                datatype=["str", "number", "str", "number", "str", "str", "str"],
            )
            
            gr.Markdown("""<br>
                        <br>
                        Judge Arena uses Together AI for inference of open-source models. FP8 models are named as -- "Turbo" where the performance of the FP16 reference models is closely matched:

                        [*"Together Turbo achieves this performance while maintaining full accuracy compared to Meta's reference implementation across all models. Llama-3.1-405B-Instruct-Turbo matches the accuracy of Meta reference models."*](https://www.together.ai/blog/together-inference-engine-2)
            """)

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
            gr.Markdown(ACKNOWLEDGEMENTS)

    # Define state variables for model tracking
    model_a_state = gr.State()
    model_b_state = gr.State()
    final_prompt_state = gr.State()
    eval_prompt_previous = gr.State(value=DEFAULT_EVAL_PROMPT_EDITABLE)  # Initialize with default value
    is_editing = gr.State(False)  # Track editing state
    compatible_mode_state = gr.State(False)  # Track compatible mode state

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
            gr.State("A"),
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
            vote_tie,
            model_name_a,
            model_name_b,
            send_btn,
            random_btn,
            gr.State(),  # placeholder for success message
        ],
    )

    vote_b.click(
        fn=vote,
        inputs=[
            gr.State("B"),
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
            vote_tie,
            model_name_a,
            model_name_b,
            send_btn,
            random_btn,
            gr.State(),  # placeholder for success message
        ],
    )

    vote_tie.click(
        fn=vote,
        inputs=[
            gr.State("Tie"),
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
            vote_tie,
            model_name_a,
            model_name_b,
            send_btn,
            random_btn,
            gr.State(),  # placeholder for success message
        ],
    )

    # Add handlers for save/cancel buttons
    def save_prompt(new_prompt, previous_prompt):
        return [
            gr.update(value=new_prompt),  # Update the prompt
            new_prompt,  # Update the previous prompt state
            gr.update(visible=False)  # Hide the buttons
        ]

    def cancel_prompt(previous_prompt):
        return [
            gr.update(value=previous_prompt),  # Revert to previous prompt
            previous_prompt,  # Keep the previous prompt state
            gr.update(visible=False)  # Hide the buttons
        ]

    def show_edit_buttons(current_value, previous_value):
        # Show buttons only if the current value differs from the previous value
        return gr.update(visible=current_value != previous_value)

    # Add handlers for save/cancel buttons and prompt changes
    save_prompt_btn.click(
        fn=save_prompt,
        inputs=[eval_prompt_editable, eval_prompt_previous],
        outputs=[eval_prompt_editable, eval_prompt_previous, edit_buttons_row]
    )

    cancel_prompt_btn.click(
        fn=cancel_prompt,
        inputs=[eval_prompt_previous],
        outputs=[eval_prompt_editable, eval_prompt_previous, edit_buttons_row]
    )

    eval_prompt_editable.change(
        fn=show_edit_buttons,
        inputs=[eval_prompt_editable, eval_prompt_previous],
        outputs=edit_buttons_row
    )

    # Function to toggle visibility based on compatible mode
    def toggle_use_reference(checked):
        if checked:
            # Get new random samples with ground truth when enabling reference mode
            human_msg, ai_msg, ground_truth_msg = get_random_human_ai_ground_truth_pair()
            return {
                ground_truth: gr.update(visible=True, value=ground_truth_msg),
                human_input: gr.update(value=human_msg),
                ai_response: gr.update(value=ai_msg),
                # Reset other UI elements
                score_a: gr.update(value=""),
                critique_a: gr.update(value=""),
                score_b: gr.update(value=""),
                critique_b: gr.update(value=""),
                vote_a: gr.update(interactive=False, variant="primary"),
                vote_b: gr.update(interactive=False, variant="primary"),
                vote_tie: gr.update(interactive=False, variant="primary"),
                model_name_a: gr.update(value="*Model: Hidden*"),
                model_name_b: gr.update(value="*Model: Hidden*"),
                random_btn: gr.update(value="🎲", variant="secondary"),
            }
        else:
            # Just hide ground truth when disabling reference mode
            return {
                ground_truth: gr.update(visible=False)
            }

    # Update the change handler to include all necessary outputs
    use_reference_toggle.change(
        fn=toggle_use_reference,
        inputs=[use_reference_toggle],
        outputs=[
            ground_truth,
            human_input,
            ai_response,
            score_a,
            critique_a,
            score_b,
            critique_b,
            vote_a,
            vote_b,
            vote_tie,
            model_name_a,
            model_name_b,
            random_btn,
        ]
    )

    # Update the submit function to handle different prompts
    def submit_and_store(
        use_reference,
        eval_criteria_text_input,
        human_input,
        ai_response,
        ground_truth_input,
        score1_description,
        score2_description,
        score3_description,
        score4_description,
        score5_description,
        is_first_game=False
    ):
        # Build prompt data dictionary
        prompt_data = {
            'human_input': human_input,
            'ai_response': ai_response,
            'ground_truth_input': ground_truth_input,
            'eval_criteria': eval_criteria_text_input,
            'score1_desc': score1_description,
            'score2_desc': score2_description,
            'score3_desc': score3_description,
            'score4_desc': score4_description,
            'score5_desc': score5_description,
        }

        # Get list of active models only for matches
        active_models = [name for name, info in model_data.items() 
                        if info.get("active", True)]  # Default to True for backward compatibility
        
        # Modified model selection logic
        atla_model = "Atla-8B-preview-2024-01-08"
        
        if is_first_game:
            # For the first game, ensure Atla is one of the models
            other_models = [m for m in active_models if m != atla_model]
            other_model = random.choice(other_models)
            
            # Randomly assign Atla to either position A or B
            if random.random() < 0.5:
                model_a, model_b = atla_model, other_model
            else:
                model_a, model_b = other_model, atla_model
        else:
            # For subsequent games, Atla appears 30% of the time
            if random.random() < 0.3:
                # Include Atla in this battle
                other_models = [m for m in active_models if m != atla_model]
                other_model = random.choice(other_models)
                
                # Randomly assign Atla to either position A or B
                if random.random() < 0.5:
                    model_a, model_b = atla_model, other_model
                else:
                    model_a, model_b = other_model, atla_model
            else:
                # Battle between two non-Atla models
                non_atla_models = [m for m in active_models if m != atla_model]
                model1, model2 = random.sample(non_atla_models, 2)
                model_a, model_b = (model1, model2) if random.random() < 0.5 else (model2, model1)

        # Get responses from models
        response_a = get_model_response(
            model_a,
            model_data.get(model_a),
            prompt_data,
            use_reference=use_reference
        )
        response_b = get_model_response(
            model_b,
            model_data.get(model_b),
            prompt_data,
            use_reference=use_reference
        )

        # Parse the responses based on model, using appropriate parsing for different models
        is_prometheus_a = (model_data.get(model_a)['organization'] == 'Prometheus')
        is_prometheus_b = (model_data.get(model_b)['organization'] == 'Prometheus')
        is_atla_a = (model_data.get(model_a)['organization'] == 'Atla')
        is_atla_b = (model_data.get(model_b)['organization'] == 'Atla')

        if is_prometheus_a:
            score_a_val, critique_a_val = prometheus_parse_model_response(response_a)
            score_a_val = f"{score_a_val} / 5"
        elif is_atla_a:
            score_a_val, critique_a_val = atla_parse_model_response(response_a)
            score_a_val = f"{score_a_val} / 5"
        else:
            score_a_val, critique_a_val = parse_model_response(response_a)
            score_a_val = f"{score_a_val} / 5"

        if is_prometheus_b:
            score_b_val, critique_b_val = prometheus_parse_model_response(response_b)
            score_b_val = f"{score_b_val} / 5"
        elif is_atla_b:
            score_b_val, critique_b_val = atla_parse_model_response(response_b)
            score_b_val = f"{score_b_val} / 5"
        else:
            score_b_val, critique_b_val = parse_model_response(response_b)
            score_b_val = f"{score_b_val} / 5"

        return (
            score_a_val,
            critique_a_val,
            score_b_val,
            critique_b_val,
            gr.update(interactive=True, variant="primary"),  # vote_a
            gr.update(interactive=True, variant="primary"),  # vote_b
            gr.update(interactive=True, variant="primary"),  # vote_tie
            model_a,
            model_b,
            eval_prompt,
            gr.update(value="*Model: Hidden*"),
            gr.update(value="*Model: Hidden*"),
            gr.update(value="Regenerate judges", variant="secondary", interactive=True),
            gr.update(value="🎲"),  # random_btn
        )

    # Update the click handler to use False for is_first_game after first submission
    def create_submit_handler():
        first_game = True
        
        def handler(*args):
            nonlocal first_game
            result = submit_and_store(*args, first_game)
            first_game = False  # Set to False after first submission
            return result
        
        return handler

    # Update the send_btn click handler
    send_btn.click(
        fn=create_submit_handler(),
        inputs=[
            use_reference_toggle,
            eval_criteria_text,
            human_input,
            ai_response,
            ground_truth,
            score1_description,
            score2_description,
            score3_description,
            score4_description,
            score5_description,
        ],
        outputs=[
            score_a,
            critique_a,
            score_b,
            critique_b,
            vote_a,
            vote_b,
            vote_tie,
            model_a_state,
            model_b_state,
            final_prompt_state,
            model_name_a,
            model_name_b,
            send_btn,
            random_btn,
        ],
    )

    # Add random button handler
    random_btn.click(
        fn=populate_random_example,
        inputs=[use_reference_toggle],  # Use compatible mode toggle to decide behavior
        outputs=[
            human_input, 
            ai_response,
            random_btn,
            score_a,
            critique_a,
            score_b,
            critique_b,
            vote_a,
            vote_b,
            vote_tie,
            model_name_a,
            model_name_b,
            ground_truth,  # Set ground truth
        ]
    )

    # Add new input change handlers
    def handle_input_change():
        """Reset UI state when inputs are changed"""
        return [
            gr.update(interactive=False),  # vote_a
            gr.update(interactive=False),  # vote_b
            gr.update(interactive=False),  # vote_tie
            gr.update(value="Run judges", variant="primary"),  # send_btn
            gr.update(value="🎲", variant="secondary"),  # random_btn
        ]

    # Update the change handlers for inputs
    human_input.change(
        fn=handle_input_change,
        inputs=[],
        outputs=[vote_a, vote_b, vote_tie, send_btn, random_btn]
    )

    ai_response.change(
        fn=handle_input_change,
        inputs=[],
        outputs=[vote_a, vote_b, vote_tie, send_btn, random_btn]
    )

    generate_btn.click(
        fn=lambda msg: (
            generate_ai_response(msg)[0],  # Only take the response text
            gr.update(
                value="Generate AI Response",  # Keep the label
                interactive=False  # Disable the button
            )
        ),
        inputs=[human_input],
        outputs=[ai_response, generate_btn]
    )

    human_input.change(
        fn=lambda x: gr.update(interactive=bool(x.strip())),
        inputs=[human_input],
        outputs=[generate_btn]
    )

    # Update the demo.load to include the random example population
    demo.load(
        fn=lambda: populate_random_example(None, False),  # Pass False for initial compatible_mode
        inputs=[],
        outputs=[
            human_input,
            ai_response,
            random_btn,
            score_a,
            critique_a,
            score_b,
            critique_b,
            vote_a,
            vote_b,
            vote_tie,
            model_name_a,
            model_name_b,
            ground_truth,
        ]
    )

    # Add new state variables for compatible mode
    eval_criteria_previous = gr.State(value=DEFAULT_EVAL_CRITERIA)
    score1_previous = gr.State(value=DEFAULT_SCORE_1)
    score2_previous = gr.State(value=DEFAULT_SCORE_2)
    score3_previous = gr.State(value=DEFAULT_SCORE_3)
    score4_previous = gr.State(value=DEFAULT_SCORE_4)
    score5_previous = gr.State(value=DEFAULT_SCORE_5)

    # Add new functions to handle compatible mode saves/cancels
    def save_compatible_prompt(criteria, score1, score2, score3, score4, score5):
        return [
            gr.update(value=criteria),  # Update criteria
            criteria,  # Update previous criteria state
            gr.update(value=score1),
            score1,
            gr.update(value=score2),
            score2,
            gr.update(value=score3),
            score3,
            gr.update(value=score4),
            score4,
            gr.update(value=score5),
            score5,
            gr.update(visible=False)  # Hide buttons
        ]

    def cancel_compatible_prompt(prev_criteria, prev_score1, prev_score2, prev_score3, prev_score4, prev_score5):
        return [
            gr.update(value=prev_criteria),
            prev_criteria,
            gr.update(value=prev_score1),
            prev_score1,
            gr.update(value=prev_score2),
            prev_score2,
            gr.update(value=prev_score3),
            prev_score3,
            gr.update(value=prev_score4),
            prev_score4,
            gr.update(value=prev_score5),
            prev_score5,
            gr.update(visible=False)
        ]

    def show_compatible_edit_buttons(*current_values):
        previous_values = current_values[1::2]  # Get previous values
        current_values = current_values[::2]    # Get current values
        return gr.update(visible=any(curr != prev for curr, prev in zip(current_values, previous_values)))

    # Add click handlers for compatible mode buttons
    compatible_save_btn.click(
        fn=save_compatible_prompt,
        inputs=[
            eval_criteria_text,
            score1_description,
            score2_description,
            score3_description,
            score4_description,
            score5_description
        ],
        outputs=[
            eval_criteria_text,
            eval_criteria_previous,
            score1_description,
            score1_previous,
            score2_description,
            score2_previous,
            score3_description,
            score3_previous,
            score4_description,
            score4_previous,
            score5_description,
            score5_previous,
            compatible_edit_buttons_row
        ]
    )

    compatible_cancel_btn.click(
        fn=cancel_compatible_prompt,
        inputs=[
            eval_criteria_previous,
            score1_previous,
            score2_previous,
            score3_previous,
            score4_previous,
            score5_previous
        ],
        outputs=[
            eval_criteria_text,
            eval_criteria_previous,
            score1_description,
            score1_previous,
            score2_description,
            score2_previous,
            score3_description,
            score3_previous,
            score4_description,
            score4_previous,
            score5_description,
            score5_previous,
            compatible_edit_buttons_row
        ]
    )

    # Add change handlers for all compatible mode inputs
    for component in [eval_criteria_text, score1_description, score2_description, 
                     score3_description, score4_description, score5_description]:
        component.change(
            fn=show_compatible_edit_buttons,
            inputs=[
                eval_criteria_text,
                eval_criteria_previous,
                score1_description,
                score1_previous,
                score2_description,
                score2_previous,
                score3_description,
                score3_previous,
                score4_description,
                score4_previous,
                score5_description,
                score5_previous
            ],
            outputs=compatible_edit_buttons_row
        )

if __name__ == "__main__":
    demo.launch()