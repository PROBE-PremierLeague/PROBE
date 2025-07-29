import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path

# Set paths
script_dir = Path(__file__).parent
root_dir = script_dir.parent.parent.parent
data_path = root_dir /"data"/ "processed" / "goal-filtered"
output_path = root_dir /"model"/"poisson"/ "lambdas" / "mle-L-BFGS-B-lambdas"
print(data_path)
# Create output directory
output_path.mkdir(parents=True, exist_ok=True)

# Seasons to process
start_year = 1993
end_year = 2023
seasons = [f"{year}-{year+1}" for year in range(start_year, end_year+1)]
#seasons = ["2018-2019","2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]


def calculate_lambda(season):

    # Read season data from CSV
    input_file = data_path/ f"{season}.csv"
    
    if not input_file.exists():
        print(f"warning: file {season} does not exist, skip")
        return

    year_data = pd.read_csv(input_file)
    

    # Get unique home teams and sort them alphabetically
    home_teams = year_data['HomeTeam'].unique()
    sorted_teams = sorted(home_teams)
    num_team = len(sorted_teams)
    # Create mapping from team names to indices
    team_to_index = {team: idx for idx, team in enumerate(sorted_teams)}

    # Add team indices to the DataFrame
    year_data['HomeTeamInd'] = year_data['HomeTeam'].map(team_to_index)
    year_data['AwayTeamInd'] = year_data['AwayTeam'].map(team_to_index)

    # Initialize goal DataFrame to store home and away goals for each match
    goal_df = pd.DataFrame(index=range(num_team), columns=range(num_team))
    for _, row in year_data.iterrows():
        home = row["HomeTeamInd"]
        away = row["AwayTeamInd"]
        home_goals = row["HomeGoal"]
        away_goals = row["AwayGoal"]
        goal_df.at[home, away] = (home_goals, away_goals)

    # Objective function for optimization (sum of squared residuals)
    def objective_function(params):
        """
        Objective function: Calculate the sum of squared residuals for all equations.

        Args:
            params (np.ndarray): Array of parameters to optimize:
                - First 20 elements: Attack strengths (a_0 to a_19)
                - Next 20 elements: Defense strengths (b_0 to b_19)
                - Last element: Home advantage parameter (c)

        Returns:
            float: Sum of squared residuals.
        """
        a = params[:num_team]          # Attack strengths
        d = params[num_team:num_team*2]        # Defense strengths
        home_gain = params[num_team*2]   # Home advantage

        # Initialize total error (sum of squared residuals)
        total_error = 0.0

        # Calculate residuals for attack and defense equations for each team
        for i in range(num_team):
            # Attack equation residual
            attack_sum = 0
            for j in range(num_team):
                if i != j and goal_df.at[i, j] is not None:
                    home_goals = goal_df.at[i, j][0]
                    attack_sum += home_goals / max(0.1, a[i] - d[j] + home_gain)

                if i != j and goal_df.at[j, i] is not None:
                    away_goals = goal_df.at[j, i][1]
                    attack_sum += away_goals / max(0.1, a[i] - d[j])

            # Add squared residual to total error
            total_error += (attack_sum - num_team * 2 + 2) ** 2

            # Defense equation residual
            defense_sum = 0
            for j in range(num_team):
                if i != j and goal_df.at[i, j] is not None:
                    away_goals = goal_df.at[i, j][1]
                    defense_sum += away_goals / max(0.1, a[j] - d[i])

                if i != j and goal_df.at[j, i] is not None:
                    home_goals = goal_df.at[j, i][0]
                    defense_sum += home_goals / max(0.1, a[j] - d[i] + home_gain)

            # Add squared residual to total error
            total_error += (defense_sum - num_team * 2 + 2) ** 2

        # League-wide goal balance equation residual
        gain_sum = 0
        for _, row in year_data.iterrows():
            home_ind = row["HomeTeamInd"]
            away_ind = row["AwayTeamInd"]
            home_goals = row["HomeGoal"]
            gain_sum += home_goals / max(0.1, a[home_ind] - d[away_ind] + home_gain) 

        # Add squared residual to total error
        total_error += (gain_sum - num_team * (num_team - 1)) ** 2

        return total_error

    # Initial parameter guess 
    initial_guess = np.ones(num_team*2+1)
    initial_guess[num_team * 2] = 0.5  # Home advantage
    lower_bounds = [0.1] * num_team + [-0.5] * num_team + [0.0]  
    upper_bounds = [5.0] * num_team + [5.0] * num_team + [1.5]   
    bounds = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]

    # Optimize using L-BFGS-B method (suitable for constrained optimization)
    print(f"Starting optimization for {input_file.name}...")
    result = minimize(
        objective_function,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 1000,    # Maximum iterations
            'ftol': 1e-8,       # Convergence tolerance
            'disp': True        # Display optimization progress
        }
    )

    # Process optimization results
    if result.success:
        # Extract optimized parameters
        solution = result.x
        attack_strength = solution[:num_team]
        defense_strength = solution[num_team:num_team*2]
        home_advantage = solution[num_team*2]

        # Create result DataFrame
        teams = list(team_to_index.keys())
        results_df = pd.DataFrame({
            'Team': teams,
            'Attack': attack_strength,
            'Defend': defense_strength,
            'HomeAdvantage': home_advantage
        })

        # Save results to CSV with the same name in the output directory
        output_file = output_path / f"{season}-parameters.csv"
        results_df.to_csv(str(output_file), index=False)
        print(f"\nSuccessfully processed {input_file.name}")
        print(f"Results saved to {output_file}")
        print(f"Home Advantage parameter: {home_advantage:.4f}")
    else:
        print(f"\nOptimization failed for {input_file.name}: {result.message}")
        print(f"Final parameters: {result.x}")

def main():
    for season in seasons:
        calculate_lambda(season)

    print("Process completed!")


if __name__ == "__main__":
    main()
