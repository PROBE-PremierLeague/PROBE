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

def compute_match_probabilities(df, goal_cutoff=10):
    teams = df['Team'].tolist()
    attack = dict(zip(df['Team'], df['Attack']))
    defense = dict(zip(df['Team'], df['Defend']))
    home_adv = df['HomeAdvantage'].iloc[0]

    results = []

    for home, away in permutations(teams, 2):
        if home == away:
            continue

        lambda_home = max(attack[home] - defense[away] + home_adv, 0.1)
        lambda_away = max(attack[away] - defense[home], 0.1)

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

        results.append({
            'home_team': home,
            'away_team': away,
            'P_home_win': round(P_home_win, 4),
            'P_draw': round(P_draw, 4),
            'P_away_win': round(P_away_win, 4)
        })

    return pd.DataFrame(results)

def main(year: str):
    try:
        df = load_team_parameters(year)
        prob_table = compute_match_probabilities(df)

        # Save to CSV
        script_dir = Path(__file__).resolve().parent
        save_dir = script_dir.parent.parent / "sim" / "results" / "probabilities"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"match_probs-{year}.csv"
        prob_table.to_csv(save_path, index=False)
        print(f"Saved match probabilities for {year} to {save_path}")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    for start_year in range(2012, 2024):
        end_year = start_year + 1
        year_str = f"{start_year}-{end_year}"
        print("=" * 50)
        print(f"Calculating match win/draw/loss probabilities for season: {year_str}")
        main(year_str)
