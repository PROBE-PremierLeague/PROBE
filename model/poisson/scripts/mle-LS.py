import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares

# Set paths
script_dir = Path(__file__).parent
poisson_dir = script_dir.parent
model_dir = poisson_dir.parent
root_dir = model_dir.parent
input_dir = root_dir / "data" / "processed" / "goal-filtered"
output_dir = poisson_dir / "lambdas" / "mle-LS-lambdas"

# Seasons to process
season = {
    '1993-1994', '1994-1995', '1995-1996', '1996-1997', '1997-1998', 
    '1998-1999', '1999-2000', '2000-2001', '2001-2002', '2002-2003', 
    '2003-2004', '2004-2005', '2005-2006', '2006-2007', '2007-2008', 
    '2008-2009', '2009-2010', '2010-2011', '2011-2012', '2012-2013', 
    '2013-2014', '2014-2015', '2015-2016', '2016-2017', '2017-2018',
    '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023',
    '2023-2024'
}

def likelihood_derivatives(x, data, team_dict, num_teams):
    F = np.zeros(num_teams * 2 + 1)

    # Attack residuals
    for i in range(num_teams):
        team = [k.split('_')[0] for k, v in team_dict.items() if v == i][0]
        
        # Home attack
        home_rows = data[data['HomeTeam'] == team]
        for _, row in home_rows.iterrows():
            opp_def = team_dict[f"{row['AwayTeam']}_Defend"]
            mu = max(0.1, x[i] - x[opp_def] + x[-1])
            F[i] += -1 + row['HomeGoal'] / mu
        
        # Away attack
        away_rows = data[data['AwayTeam'] == team]
        for _, row in away_rows.iterrows():
            opp_def = team_dict[f"{row['HomeTeam']}_Defend"]
            mu = max(0.1, x[i] - x[opp_def])
            F[i] += -1 + row['AwayGoal'] / mu

    # Defense residuals
    for i in range(num_teams, num_teams * 2):
        team = [k.split('_')[0] for k, v in team_dict.items() if v == i][0]

        # Home defend (opponent is away team)
        away_rows = data[data['AwayTeam'] == team]
        for _, row in away_rows.iterrows():
            opp_atk = team_dict[f"{row['HomeTeam']}_Attack"]
            mu = max(0.1, x[opp_atk] - x[i] + x[-1])
            F[i] += 1 - row['HomeGoal'] / mu

        # Away defend
        home_rows = data[data['HomeTeam'] == team]
        for _, row in home_rows.iterrows():
            opp_atk = team_dict[f"{row['AwayTeam']}_Attack"]
            mu = max(0.1, x[opp_atk] - x[i])
            F[i] += 1 - row['AwayGoal'] / mu

    # Home advantage residual
    for _, row in data.iterrows():
        home_atk = team_dict[f"{row['HomeTeam']}_Attack"]
        away_def = team_dict[f"{row['AwayTeam']}_Defend"]
        mu = max(0.1, x[home_atk] - x[away_def] + x[-1])
        F[-1] += -1 + row['HomeGoal'] / mu

    return F

def calculate_team_parameters(year):
    # Load data
    input_file = input_dir / f"{year}.csv"
    if not input_file.exists():
        print(f"Warning: {input_file} does not exist. Skipped.")
        return

    data = pd.read_csv(input_file)
    teams = sorted(set(data['HomeTeam']).union(data['AwayTeam']))
    num_teams = len(teams)
    print(f"Processing {year} with {num_teams} teams.")

    # Create team index map
    team_dict = {}
    for idx, team in enumerate(teams):
        team_dict[f"{team}_Attack"] = idx
        team_dict[f"{team}_Defend"] = idx + num_teams

    # Initial guess
    attack_init = np.linspace(1.2, 2.0, num_teams)
    defend_init = np.linspace(0.5, 1.5, num_teams)
    init_guess = np.concatenate([attack_init, defend_init, [0.25]])

    # Bounds
    lower_bounds = [0.5] * num_teams + [-0.5] * num_teams + [0.0]
    upper_bounds = [3.0] * num_teams + [2.0] * num_teams + [0.5]
    bounds = (lower_bounds, upper_bounds)

    # Optimization
    result = least_squares(
        fun=likelihood_derivatives,
        x0=init_guess,
        args=(data, team_dict, num_teams),
        bounds=bounds,
        max_nfev=2000,
        verbose=1
    )

    if not result.success:
        print(f"Optimization failed for {year}: {result.message}")
        return

    # Save result
    params = result.x
    result_df = pd.DataFrame({
        'Team': teams,
        'Attack': params[:num_teams],
        'Defend': params[num_teams:num_teams*2],
        'HomeAdvantage': params[-1]
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{year}-parameters.csv"
    result_df.to_csv(output_file, index=False)
    print(result_df)
    print(f"{output_file.name} saved successfully!\n")

    return result_df

if __name__ == "__main__":
    for year in sorted(season):
        print(f"\n=== Calculating parameters for {year} ===")
        calculate_team_parameters(year)

    print("All seasons processed.")
