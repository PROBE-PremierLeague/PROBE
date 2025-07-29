import pandas as pd
import numpy as np
from itertools import permutations
from scipy.stats import poisson
from pathlib import Path

def load_team_parameters(year: str):
    script_dir = Path(__file__).resolve().parent
    lambdas_path = (
        script_dir.parent.parent / "model" / "poisson" / "lambdas" / "mle-L-BFGS-B-lambdas"
    )
    filename = f"{year}-parameters.csv"
    full_path = lambdas_path / filename
    if not full_path.exists():
        raise FileNotFoundError(f"Cannot find file: {full_path}")
    return pd.read_csv(full_path)

def simulate_expectation(df, goal_cutoff=10):
    teams = df['Team'].tolist()
    attack = dict(zip(df['Team'], df['Attack']))
    defense = dict(zip(df['Team'], df['Defend']))
    home_adv = df['HomeAdvantage'].iloc[0]

    standings = {team: {'Pts': 0.0, 'GF': 0.0, 'GA': 0.0} for team in teams}

    for home, away in permutations(teams, 2):
        if home == away:
            continue

        lambda_home = max(attack[home] - defense[away] + home_adv, 0.1)
        lambda_away = max(attack[away] - defense[home], 0.1)

        standings[home]['GF'] += lambda_home
        standings[home]['GA'] += lambda_away
        standings[away]['GF'] += lambda_away
        standings[away]['GA'] += lambda_home

        P_home_win = 0.0
        P_draw = 0.0
        P_away_win = 0.0

        for h in range(goal_cutoff + 1):
            for a in range(goal_cutoff + 1):
                p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
                if h > a:
                    P_home_win += p
                elif h == a:
                    P_draw += p
                else:
                    P_away_win += p

        standings[home]['Pts'] += 3 * P_home_win + 1 * P_draw
        standings[away]['Pts'] += 3 * P_away_win + 1 * P_draw

    return standings

def build_league_table(standings):
    table = []
    for team, s in standings.items():
        GD = s['GF'] - s['GA']
        table.append([team, round(s['Pts'], 2), round(s['GF'], 2), round(s['GA'], 2), round(GD, 2)])

    df_table = pd.DataFrame(table, columns=['Team', 'Pts', 'GF', 'GA', 'GD'])
    df_table.sort_values(by=['Pts', 'GD', 'GF'], ascending=False, inplace=True)
    df_table.reset_index(drop=True, inplace=True)

    # Insert rank as the first column
    df_table.insert(0, 'rank', df_table.index)
    return df_table


def main(year: str, save_dir: Path):
    try:
        df = load_team_parameters(year)
        standings = simulate_expectation(df)
        league_table = build_league_table(standings)

        # Save to CSV
        save_path = save_dir / f"table-{year}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        league_table.to_csv(save_path, index=False)  # Now 'rank' is a proper column
        print(f"Saved table for {year} to {save_path}")

    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    # Output directory
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent.parent / "sim" / "results" / "past"

    # Simulate for all seasons from 2012-2013 to 2023-2024
    for start_year in range(2012, 2024):
        end_year = start_year + 1
        year_str = f"{start_year}-{end_year}"
        print("=" * 50)
        print(f"Simulating season: {year_str}")
        main(year_str, results_dir)
