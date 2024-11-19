import json
import re
import random
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from gen_api_answer import (
    get_model_response, 
    parse_model_response, 
    get_random_human_ai_pair,
    generate_ai_response
)
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
    DEFAULT_EVAL_PROMPT_EDITABLE,
    FIXED_EVAL_SUFFIX,
)
from leaderboard import (
    get_leaderboard,
    get_leaderboard_stats,
    calculate_elo_change,
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


def populate_random_example(request: gr.Request):
    """Generate a random human-AI conversation example and reset judge outputs."""
    human_msg, ai_msg = get_random_human_ai_pair()
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

            # Update Evaluator Prompt Accordion
            with gr.Accordion("📝 Judge Prompt", open=False):
                eval_prompt_editable = gr.TextArea(
                    value=DEFAULT_EVAL_PROMPT_EDITABLE,
                    label="Evaluation Criteria",
                    lines=12
                )
                with gr.Row(visible=False) as edit_buttons_row:  # Make buttons row initially hidden
                    cancel_prompt_btn = gr.Button("Cancel")
                    save_prompt_btn = gr.Button("Save", variant="primary")
                gr.Markdown("*The sample being evaluated is always appended as:*")
                gr.Markdown(f"```{FIXED_EVAL_SUFFIX}")

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

    # Update variable inputs based on the eval prompt
    #def update_variables(eval_prompt):
    #    variables = parse_variables(eval_prompt)
    #    updates = []

    #    for i in range(len(variable_rows)):
    #        var_row, var_input = variable_rows[i]
    #        if i < len(variables):
    #            var_name = variables[i]
    #            # Set the number of lines based on the variable name
    #            if var_name == "response":
    #                lines = 4  # Adjust this number as needed
    #            else:
    #                lines = 1  # Default to single line for other variables
    #            updates.extend(
    #                [
    #                    gr.update(visible=True),  # Show the variable row
    #                    gr.update(
    #                        label=var_name, visible=True, lines=lines
    #                    ),  # Update label and lines
    #                ]
    #            )
    #        else:
    #            updates.extend(
    #                [
    #                        gr.update(visible=False),  # Hide the variable row
    #                        gr.update(value="", visible=False),  # Clear value when hidden
    #                    ]
    #            )
    #    return updates

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

    # Update the submit function to combine editable and fixed parts
    def submit_and_store(editable_prompt, *variables):
        # Combine the editable prompt with fixed suffix
        full_prompt = editable_prompt + FIXED_EVAL_SUFFIX
        
        # Get the responses using the full prompt
        (
            response_a,
            response_b,
            buttons_visible,
            regen_visible,
            model_a,
            model_b,
            final_prompt,
        ) = submit_prompt(full_prompt, *variables)

        # Parse the responses
        score_a, critique_a = parse_model_response(response_a)
        score_b, critique_b = parse_model_response(response_b)

        # Only append "/ 5" if using the default prompt
        if editable_prompt.strip() == DEFAULT_EVAL_PROMPT_EDITABLE.strip():
            score_a = f"{score_a} / 5"
            score_b = f"{score_b} / 5"

        # Update the last_submission state with the current values
        last_submission.value = {"prompt": full_prompt, "variables": variables}

        return (
            score_a,
            critique_a,
            score_b,
            critique_b,
            gr.update(interactive=True, variant="primary"),  # vote_a
            gr.update(interactive=True, variant="primary"),  # vote_b
            gr.update(interactive=True, variant="primary"),  # vote_tie
            model_a,
            model_b,
            final_prompt,
            gr.update(value="*Model: Hidden*"),
            gr.update(value="*Model: Hidden*"),
            gr.update(
                value="Regenerate judges",
                variant="secondary",
                interactive=True
            ),
            gr.update(value="🎲"),  # random_btn
        )

    # Update the click handler to use the editable prompt
    send_btn.click(
        fn=submit_and_store,
        inputs=[eval_prompt_editable, human_input, ai_response],
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

    # Update the input change handlers to also disable regenerate button
    # def handle_input_changes(prompt, *variables):
    #    """Enable send button and manage regenerate button based on input changes"""
    #    last_inputs = last_submission.value
    #    current_inputs = {"prompt": prompt, "variables": variables}
    #    inputs_changed = last_inputs != current_inputs
    #    return [
    #        gr.update(interactive=True),  # send button always enabled
    #        gr.update(
    #            interactive=not inputs_changed
    #        ),  # regenerate button disabled if inputs changed
    #    ]

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
        fn=populate_random_example,
        inputs=[],
        outputs=[human_input, ai_response]
    )

if __name__ == "__main__":
    demo.launch()