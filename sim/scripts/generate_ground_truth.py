import pandas as pd
import numpy as np
from pathlib import Path

def load_match_data(year: str):
    """Load match data from the goal-filtered CSV files"""
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent.parent / "data" / "processed" / "goal-filtered"
    filename = f"{year}.csv"
    full_path = data_path / filename
    
    if not full_path.exists():
        raise FileNotFoundError(f"Cannot find file: {full_path}")
    
    return pd.read_csv(full_path)

def calculate_league_table(df_matches):
    """Calculate actual league table from match results"""
    teams = set(df_matches['HomeTeam'].unique()) | set(df_matches['AwayTeam'].unique())
    standings = {team: {'Pts': 0, 'GF': 0, 'GA': 0, 'MP': 0} for team in teams}
    
    for _, match in df_matches.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        home_goals = match['HomeGoal']
        away_goals = match['AwayGoal']
        
        # Update matches played
        standings[home_team]['MP'] += 1
        standings[away_team]['MP'] += 1
        
        # Update goals for/against
        standings[home_team]['GF'] += home_goals
        standings[home_team]['GA'] += away_goals
        standings[away_team]['GF'] += away_goals
        standings[away_team]['GA'] += home_goals
        
        # Update points
        if home_goals > away_goals:
            standings[home_team]['Pts'] += 3
        elif home_goals < away_goals:
            standings[away_team]['Pts'] += 3
        else:
            standings[home_team]['Pts'] += 1
            standings[away_team]['Pts'] += 1
    
    # Build table
    table = []
    for team, stats in standings.items():
        GD = stats['GF'] - stats['GA']
        table.append([team, stats['Pts'], stats['GF'], stats['GA'], GD])
    
    df_table = pd.DataFrame(table, columns=['Team', 'Pts', 'GF', 'GA', 'GD'])
    df_table.sort_values(by=['Pts', 'GD', 'GF'], ascending=False, inplace=True)
    df_table.reset_index(drop=True, inplace=True)
    
    # Insert rank as the first column (starting from 1)
    df_table.insert(0, 'rank', df_table.index + 1)
    
    return df_table

def main(year: str, save_dir: Path):
    """Generate ground-truth table for a specific year"""
    try:
        # Load match data
        df_matches = load_match_data(year)
        
        # Calculate actual league table
        league_table = calculate_league_table(df_matches)
        
        # Save to CSV
        save_path = save_dir / f"{year}-ground-truth.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        league_table.to_csv(save_path, index=False)
        print(f"Saved ground-truth table for {year} to {save_path}")
        
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    # Output directory
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent / "results" / "ground-truth"
    
    # Get all available years from goal-filtered directory
    data_dir = script_dir.parent.parent / "data" / "processed" / "goal-filtered"
    csv_files = list(data_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        year = csv_file.stem  # Remove .csv extension
        print("=" * 50)
        print(f"Processing season: {year}")
        main(year, results_dir)