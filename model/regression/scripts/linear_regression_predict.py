import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from pathlib import Path

def load_parameters_data():
    model_dir = Path(__file__).parent.parent.parent
    params_path = os.path.join(model_dir,'poisson', 'lambdas', 'mle-L-BFGS-B-lambdas')
    params_files = [f for f in os.listdir(params_path) if f.endswith('-parameters.csv')]
    all_params = []
    
    # Specify 6 teams
    selected_teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham']
    
    for file in params_files:
        year = file.split('-parameters.csv')[0]
        df = pd.read_csv(os.path.join(params_path, file))
        df = df[df['Team'].isin(selected_teams)]
        df['Season'] = year
        all_params.append(df)
    
    return pd.concat(all_params).reset_index(drop=True)


def prepare_training_data(params_df, team1, team2):

    team1_data = params_df[params_df['Team'] == team1].sort_values('Season')
    team2_data = params_df[params_df['Team'] == team2].sort_values('Season')
    
    if len(team1_data) < 4 or len(team2_data) < 4:
        return None, None
    
    all_X = []  
    all_y = []  
    
    for i in range(len(team1_data) - 3):
        window1 = team1_data.iloc[i:i+4]
        window2 = team2_data.iloc[i:i+4]
        
        # Ensure seasons match
        if not all(window1['Season'].values == window2['Season'].values):
            continue
        
        # Features: data from previous 3 years
        X = np.array([
            # team1's attack λ from previous 3 years
            window1.iloc[0]['Attack'],  # t-3 year
            window1.iloc[1]['Attack'],  # t-2 year
            window1.iloc[2]['Attack'],  # t-1 year
            # team2's defense λ from previous 3 years
            window2.iloc[0]['Defend'],  # t-3 year
            window2.iloc[1]['Defend'],  # t-2 year
            window2.iloc[2]['Defend'],  # t-1 year
            window1.iloc[0]['HomeAdvantage'],
            window1.iloc[1]['HomeAdvantage'],
            window1.iloc[2]['HomeAdvantage']
        ])
        
        # Target: team1's attack λ - team2's defense λ in the 4th year
        y = window1.iloc[3]['Attack'] - window2.iloc[3]['Defend'] 
        # + window1.iloc[3]['HomeAdvantage']
        
        all_X.append(X)
        all_y.append(y)
    
    if not all_X:  # If no valid training data found
        return None, None
        
    return np.array(all_X), np.array(all_y)

def calculate_predictions(X, params):
    """Calculate predictions"""
    intercept, beta1, beta2, beta3 = params
    predictions = (
        intercept + 
        beta1 * X[:, 0] + beta2 * X[:, 1] + beta3 * X[:, 2] -  # team1's attack parameters
        beta1 * X[:, 3] - beta2 * X[:, 4] - beta3 * X[:, 5] 
        # + beta1 * X[:, 6] + beta2 * X[:, 7] + beta3 * X[:, 8]
    )
    return predictions

def calculate_loss(params, params_df):
    """
    Calculate total loss for all team pair combinations
    params: Model parameters [intercept, beta1, beta2, beta3, gamma1, gamma2, gamma3]
    params_df: DataFrame containing λ parameters for all teams across seasons
    """
    total_loss = 0
    # Specify 6 teams
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham']
    
    # Calculate loss for each pair of teams
    for i in range(len(teams)):
        for j in range(len(teams)):
            if i != j:  # Different teams
                team1, team2 = teams[i], teams[j]
                
                X, y = prepare_training_data(params_df, team1, team2)
                if X is not None and y is not None:
                    y_pred = calculate_predictions(X, params)
                    loss = np.sum((y - y_pred) ** 2)
                    total_loss += loss
    
    return total_loss

def train_global_model(params_df):
    """
    Train global model, optimize a set of shared coefficients
    params_df: DataFrame containing λ parameters for all teams across seasons
    """
    # Initial parameter guess
    initial_params = np.zeros(4)  # [intercept, beta1, beta2, beta3, gamma1, gamma2, gamma3]
    
    # Optimization
    result = minimize(
        fun=calculate_loss,
        x0=initial_params,
        args=(params_df,),
        method='L-BFGS-B'
    )
    
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    
    return result.x

def predict_next_season(params_df, coefficients, target_season):
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham']
    predictions = {}
    
    print(f"\n========== {target_season} Season Predictions ==========")
    print("\nLinear Regression Predictions:")
    print("Team1 (Attack) vs Team2 (Defense) -> Predicted Difference")
    print("-" * 50)
    
    for team1 in teams:
        team_predictions = {}
        for team2 in teams:
            if team1 != team2:
                # Get historical data
                team1_data = params_df[params_df['Team'] == team1].sort_values('Season')
                team2_data = params_df[params_df['Team'] == team2].sort_values('Season')
                
                # Get previous seasons
                current_season = target_season.split('-')[0]
                prev_seasons = [
                    f"{int(current_season)-3}-{int(current_season)-2}",
                    f"{int(current_season)-2}-{int(current_season)-1}",
                    f"{int(current_season)-1}-{int(current_season)}"
                ]
                
                # Check if we have enough historical data
                if not all(season in team1_data['Season'].values for season in prev_seasons) or \
                   not all(season in team2_data['Season'].values for season in prev_seasons):
                    continue
                
                # Build feature vector
                X = np.array([[
                    team1_data[team1_data['Season'] == prev_seasons[0]]['Attack'].values[0],
                    team1_data[team1_data['Season'] == prev_seasons[1]]['Attack'].values[0],
                    team1_data[team1_data['Season'] == prev_seasons[2]]['Attack'].values[0],
                    team2_data[team2_data['Season'] == prev_seasons[0]]['Defend'].values[0],
                    team2_data[team2_data['Season'] == prev_seasons[1]]['Defend'].values[0],
                    team2_data[team2_data['Season'] == prev_seasons[2]]['Defend'].values[0],
                    team1_data[team1_data['Season'] == prev_seasons[0]]['HomeAdvantage'].values[0],
                    team1_data[team1_data['Season'] == prev_seasons[1]]['HomeAdvantage'].values[0],
                    team1_data[team1_data['Season'] == prev_seasons[2]]['HomeAdvantage'].values[0]
                ]])
                
                # Make prediction
                pred = calculate_predictions(X, coefficients)[0]
                print(f"{team1:10} vs {team2:10} -> {pred:.4f}")
                team_predictions[team2] = float(pred)
        
        if team_predictions:  # Only add teams that have predictions
            predictions[team1] = team_predictions
    
    return predictions, X

def calculate_attack_defense_homeadvantage(predictions, X):
    """
    Calculate Attack, Defense, and HomeAdvantage for each team using predictions.
    """
    teams = list(predictions.keys())
    n_teams = len(teams)
    
    # Create matrices for the linear system
    A = []  # Coefficient matrix
    b = []  # Result vector

    for team1, opponents in predictions.items():
        for team2, prediction in opponents.items():
            row = [0] * (2 * n_teams + 1)  # [attack_1, ..., attack_n, defense_1, ..., defense_n, homeadvantage]
            row[teams.index(team1)] = 1  # attack_1
            row[n_teams + teams.index(team2)] = -1  # -defense_2
            # row[-1] = 1  # homeadvantage
            A.append(row)
            b.append(prediction)
    
    # Solve the linear system Ax = b
    A = np.array(A)
    b = np.array(b)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # Least squares solution
    x = np.append(x, 0.5 * (X[0, 7] + X[0, 8]))

    # Extract results
    attack = x[:n_teams]
    defense = x[n_teams:2 * n_teams]
    # homeadvantage = x[-1] - x[-1] + 0.5 * (X[7].item() + X[8].item())

    # Create a DataFrame for results
    results = pd.DataFrame({
        "Team": teams,
        "Attack": attack,
        "Defense": defense
    })
    results["HomeAdvantage"] = x[-1]
    # results["HomeAdvantage"] = homeadvantage  # Add homeadvantage as a constant column

    return results

def save_results_to_csv(results, output_path):
    """
    Save the calculated Attack, Defense, and HomeAdvantage to a CSV file.
    """
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":

    params_df = load_parameters_data()
    coefficients = train_global_model(params_df)
    
    print("\n========== Optimization Results ==========")
    print("\nAttack Parameters (β):")
    print(f"β1 (t-3 year attack effect): {coefficients[1]:.4f}")
    print(f"β2 (t-2 year attack effect): {coefficients[2]:.4f}")
    print(f"β3 (t-1 year attack effect): {coefficients[3]:.4f}")
    
    # print("\nDefense Parameters (γ):")
    # print(f"γ1 (t-3 year defense effect): {coefficients[4]:.4f}")
    # print(f"γ2 (t-2 year defense effect): {coefficients[5]:.4f}")
    # print(f"γ3 (t-1 year defense effect): {coefficients[6]:.4f}")

    # print("\nHome Advantage Parameters (θ):")
    # print(f"θ1 (t-3 year home advantage): {coefficients[7]:.4f}")
    # print(f"θ2 (t-2 year home advantage): {coefficients[8]:.4f}")
    # print(f"θ3 (t-1 year home advantage): {coefficients[9]:.4f}")

    print(f"Intercept: {coefficients[0]:.4f}")
    
    # Save model parameters
    model_params = {
        'beta1': coefficients[1],
        'beta2': coefficients[2],
        'beta3': coefficients[3],
        'intercept': coefficients[0]
    }
    
    # Create output directory and save parameters
    output_dir = os.path.join('..', 'results','linear_regression', 'mle-L-BFGS-B-lambdas')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'model_parameters_new_no_home_advantage.json')
    with open(output_path, 'w') as f:
        json.dump(model_params, f, indent=4)
    
    # Make and save predictions for next season
    target_season = "2024-2025"
    predictions, X = predict_next_season(params_df, coefficients, target_season)
    
    # Save predictions
    predictions_path = os.path.join(output_dir, f'predictions_{target_season.replace("-", "_")}_no_home_advantage.json')
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    print(f"\nModel parameters and predictions have been saved to: {output_dir}")

    # Calculate Attack, Defense, and HomeAdvantage
    results = calculate_attack_defense_homeadvantage(predictions, X)
    # Save results to CSV
    output_path = os.path.join(output_dir, 'team_data_2024_2025_no_home_advantage.csv')
    save_results_to_csv(results, output_path)
    