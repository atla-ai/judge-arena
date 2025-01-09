from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

# Constants
DEFAULT_ELO = 1200  # Starting ELO for new models
K_FACTOR = 32  # Standard chess K-factor

def get_leaderboard(model_data: Dict, voting_data: List, show_preliminary=True):
    """Generate leaderboard data using votes from MongoDB."""
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
        # Skip models with < 300 votes if show_preliminary is False
        if not show_preliminary and votes < 300:
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

def get_leaderboard_stats(model_data: Dict, voting_data: List) -> str:
    """Get summary statistics for the leaderboard."""
    now = datetime.now(timezone.utc)
    total_votes = len(voting_data)
    total_models = len(model_data)
    # last_updated = now.strftime("%B %d, %Y at %H:%M:%S UTC")

    last_updated = now.replace(minute=0, second=0, microsecond=0).strftime(
        "%B %d, %Y at %H:00 UTC"
    )

    return f"""
### Leaderboard Stats
- **Total Models**: {total_models}
- **Total Votes**: {total_votes}
- **Last Updated**: {last_updated}
"""

def calculate_elo_change(rating_a: float, rating_b: float, winner: str) -> tuple[float, float]:
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

def get_model_rankings(leaderboard: List[Dict]) -> Dict[str, int]:
    """Get current rankings of all models from leaderboard data."""
    return {entry["Model"]: idx + 1 for idx, entry in enumerate(leaderboard)}