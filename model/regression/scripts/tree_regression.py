import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json

def load_parameters_data():
    model_dir = Path(__file__).parent.parent.parent
    params_path = os.path.join(model_dir,'poisson', 'lambdas', 'mle-L-BFGS-B-lambdas')
    params_files = [f for f in os.listdir(params_path) if f.endswith('-parameters.csv')]
    all_params = []
    
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
        
        if not all(window1['Season'].values == window2['Season'].values):
            continue
        
        X = np.array([
            window1.iloc[0]['Attack'],
            window1.iloc[1]['Attack'],
            window1.iloc[2]['Attack'],
            window2.iloc[0]['Defend'],
            window2.iloc[1]['Defend'],
            window2.iloc[2]['Defend'],
            window1.iloc[0]['HomeAdvantage'],
            window1.iloc[1]['HomeAdvantage'],
            window1.iloc[2]['HomeAdvantage']
        ])
        
        y = window1.iloc[3]['Attack'] - window2.iloc[3]['Defend'] + window1.iloc[3]['HomeAdvantage']
        
        all_X.append(X)
        all_y.append(y)
    
    if not all_X:
        return None, None
        
    return np.array(all_X), np.array(all_y)

def train_tree_models(params_df):
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham']
    
    # Collect all training data
    X_all = []
    y_all = []
    
    for i in range(len(teams)):
        for j in range(len(teams)):
            if i != j:
                team1, team2 = teams[i], teams[j]
                X, y = prepare_training_data(params_df, team1, team2)
                if X is not None and y is not None:
                    X_all.extend(X)
                    y_all.extend(y)
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # Train Decision Tree
    dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_model.fit(X_all, y_all)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_all, y_all)
    
    # Calculate performance metrics
    dt_pred = dt_model.predict(X_all)
    rf_pred = rf_model.predict(X_all)
    
    metrics = {
        'decision_tree': {
            'mse': mean_squared_error(y_all, dt_pred),
            'r2': r2_score(y_all, dt_pred)
        },
        'random_forest': {
            'mse': mean_squared_error(y_all, rf_pred),
            'r2': r2_score(y_all, rf_pred)
        }
    }
    
    return dt_model, rf_model, metrics

def predict_next_season(params_df, model, team1, team2, target_season):
    team1_data = params_df[params_df['Team'] == team1].sort_values('Season')
    team2_data = params_df[params_df['Team'] == team2].sort_values('Season')
    
    current_season = target_season.split('-')[0]
    prev_seasons = [
        f"{int(current_season)-3}-{int(current_season)-2}",
        f"{int(current_season)-2}-{int(current_season)-1}",
        f"{int(current_season)-1}-{int(current_season)}"
    ]
    
    if not all(season in team1_data['Season'].values for season in prev_seasons) or \
       not all(season in team2_data['Season'].values for season in prev_seasons):
        return None
    
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
    
    return model.predict(X)[0]

def calculate_attack_defense_homeadvantage(predictions):
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
            row[-1] = 1  # homeadvantage
            A.append(row)
            b.append(prediction)
    
    # Solve the linear system Ax = b
    A = np.array(A)
    b = np.array(b)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # Least squares solution

    # Extract results
    attack = x[:n_teams]
    defense = x[n_teams:2 * n_teams]
    homeadvantage = x[-1]

    # Create a DataFrame for results
    results = pd.DataFrame({
        "Team": teams,
        "Attack": attack,
        "Defense": defense
    })
    results["HomeAdvantage"] = homeadvantage  # Add homeadvantage as a constant column

    return results

def save_results_to_csv(results, output_path):
    """
    Save the calculated Attack, Defense, and HomeAdvantage to a CSV file.
    """
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    params_df = load_parameters_data()
    
    dt_model, rf_model, metrics = train_tree_models(params_df)
    
    print("\n========== Model Performance ==========")
    print("\nDecision Tree Metrics:")
    print(f"MSE: {metrics['decision_tree']['mse']:.4f}")
    print(f"R2 Score: {metrics['decision_tree']['r2']:.4f}")
    
    print("\nRandom Forest Metrics:")
    print(f"MSE: {metrics['random_forest']['mse']:.4f}")
    print(f"R2 Score: {metrics['random_forest']['r2']:.4f}")
    
    # Feature importance for Random Forest
    feature_names = [
        'Attack (t-3)', 'Attack (t-2)', 'Attack (t-1)',
        'Defense (t-3)', 'Defense (t-2)', 'Defense (t-1)',
        'HomeAdvantage (t-3)', 'HomeAdvantage (t-2)', 'HomeAdvantage (t-1)'
    ]
    
    print("\n========== Feature Importance (Random Forest) ==========")
    importances = rf_model.feature_importances_
    feature_importance_dict = {name: importance for name, importance in zip(feature_names, importances)}

    for name, importance in feature_importance_dict.items():
        print(f"{name}: {importance:.4f}")
    
    # Predict next season
    target_season = "2024-2025"
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham']
    
    print(f"\n========== {target_season} Season Predictions ==========")
    print("\nRandom Forest Predictions:")
    print("Team1 (Attack) vs Team2 (Defense) -> Predicted Difference")
    print("-" * 50)
    
    predictions = {}

    for team1 in teams:
        predictions[team1] = {} 
        for team2 in teams:
            if team1 != team2:
                pred = predict_next_season(params_df, rf_model, team1, team2, target_season)
                if pred is not None:
                    print(f"{team1:10} vs {team2:10} -> {pred:.4f}")
                    predictions[team1][team2] = pred
    
    # Save models, metrics, and feature importances
    output_dir = os.path.join('..', 'results', 'random_forest', 'mle-L-BFGS-B-lambdas')
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_path = os.path.join(output_dir, 'tree_model_metrics.json')
    with open(metrics_path, 'w') as f:
        combined_data = {
            "metrics": metrics,
            "feature_importances": feature_importance_dict
        }
        json.dump(combined_data, f, indent=4)

    prediction_path = os.path.join(output_dir, f'predictions_{target_season.replace("-", "_")}.json')
    with open(prediction_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    print(f"\nModel metrics, feature importances, and predictions have been saved to: {metrics_path} and {prediction_path}")

    # Calculate Attack, Defense, and HomeAdvantage
    results = calculate_attack_defense_homeadvantage(predictions)
    
    # Save results to CSV
    output_path = os.path.join(output_dir, f'team_data_{target_season.replace("-", "_")}.csv')
    save_results_to_csv(results, output_path)
