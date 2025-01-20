from leaderboard import get_leaderboard
from db import create_db_connection, get_votes
import csv
from pathlib import Path
import json

def save_leaderboard_csv(output_path: str = "leaderboard.csv"):
    """
    Save the current leaderboard data to a CSV file.
    
    Args:
        output_path: Path where the CSV should be saved (default: 'leaderboard.csv')
    """
    # Initialize database connection
    db = create_db_connection()
    votes = get_votes(db)
    
    # Load model data
    model_data = {}
    try:
        with open("data/models.jsonl", "r") as f:
            for line in f:
                model = json.loads(line)
                model_data[model["name"]] = {
                    "organization": model["organization"],
                    "license": model["license"],
                }
    except FileNotFoundError:
        print("Warning: models.jsonl not found")
        return
    
    # Get leaderboard data
    leaderboard = get_leaderboard(model_data, votes, show_preliminary=True)
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write to CSV
    fieldnames = ["Model", "ELO Score", "95% CI", "# Votes", "Organization", "License"]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(leaderboard)
    
    print(f"Leaderboard saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Save leaderboard data to CSV')
    parser.add_argument('--output', '-o', 
                       default='leaderboard.csv',
                       help='Output path for CSV file')
    
    args = parser.parse_args()
    save_leaderboard_csv(args.output)