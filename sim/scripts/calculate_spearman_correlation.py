import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import warnings

def load_ground_truth_table(year: str):
    """Load ground-truth table for a specific year"""
    script_dir = Path(__file__).resolve().parent
    ground_truth_path = script_dir.parent / "results" / "ground-truth"
    filename = f"{year}-ground-truth.csv"
    full_path = ground_truth_path / filename
    
    if not full_path.exists():
        raise FileNotFoundError(f"Cannot find ground-truth file: {full_path}")
    
    return pd.read_csv(full_path)

def load_simulated_table(year: str):
    """Load simulated table for a specific year"""
    script_dir = Path(__file__).resolve().parent
    tables_path = script_dir.parent / "results" / "tables"
    filename = f"table-{year}.csv"
    full_path = tables_path / filename
    
    if not full_path.exists():
        raise FileNotFoundError(f"Cannot find simulated table file: {full_path}")
    
    return pd.read_csv(full_path)

def calculate_spearman_correlation(df_ground_truth, df_simulated):
    """Calculate Spearman's rank correlation between ground-truth and simulated tables"""
    # Create mappings of team names to their ranks
    ground_truth_ranks = dict(zip(df_ground_truth['Team'], df_ground_truth['rank']))
    simulated_ranks = dict(zip(df_simulated['Team'], df_simulated['rank']))
    
    # Get all unique teams that appear in both tables
    all_teams = set(ground_truth_ranks.keys()) & set(simulated_ranks.keys())
    
    if len(all_teams) == 0:
        return None, None, 0
    
    # Create rank pairs for teams that exist in both tables
    ground_truth_rank_list = []
    simulated_rank_list = []
    
    for team in all_teams:
        ground_truth_rank_list.append(ground_truth_ranks[team])
        simulated_rank_list.append(simulated_ranks[team])
    
    # Calculate Spearman's rank correlation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        correlation, p_value = spearmanr(ground_truth_rank_list, simulated_rank_list)
    
    return correlation, p_value, len(all_teams)

def calculate_correlation_for_year(year: str):
    """Calculate Spearman's rank correlation for a specific year"""
    try:
        # Load both tables
        df_ground_truth = load_ground_truth_table(year)
        df_simulated = load_simulated_table(year)
        
        # Calculate correlation
        correlation, p_value, n_teams = calculate_spearman_correlation(df_ground_truth, df_simulated)
        
        return {
            'year': year,
            'correlation': correlation,
            'p_value': p_value,
            'n_teams': n_teams
        }
        
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return None
    except Exception as e:
        print(f"Error processing year {year}: {e}")
        return None

def main():
    """Calculate Spearman's rank correlation for all available years"""
    script_dir = Path(__file__).resolve().parent
    ground_truth_dir = script_dir.parent / "results" / "ground-truth"
    tables_dir = script_dir.parent / "results" / "tables"
    
    # Get all available years from ground-truth directory
    ground_truth_files = list(ground_truth_dir.glob("*-ground-truth.csv"))
    available_years = [f.stem.replace('-ground-truth', '') for f in ground_truth_files]
    
    # Filter to only years that exist in both directories
    results = []
    for year in available_years:
        ground_truth_file = ground_truth_dir / f"{year}-ground-truth.csv"
        table_file = tables_dir / f"table-{year}.csv"
        
        if ground_truth_file.exists() and table_file.exists():
            result = calculate_correlation_for_year(year)
            if result is not None:
                results.append(result)
    
    # Display results
    print("\n" + "="*80)
    print("Spearman's Rank Correlation Results")
    print("="*80)
    print(f"{'Year':<12} {'Correlation':<12} {'P-value':<12} {'Teams':<8}")
    print("-" * 50)
    
    for result in results:
        if result['correlation'] is not None:
            print(f"{result['year']:<12} {result['correlation']:<12.4f} {result['p_value']:<12.4f} {result['n_teams']:<8}")
        else:
            print(f"{result['year']:<12} {'N/A':<12} {'N/A':<12} {'N/A':<8}")
    
    # Calculate summary statistics
    valid_correlations = [r['correlation'] for r in results if r['correlation'] is not None]
    if valid_correlations:
        print("\nSummary Statistics:")
        print(f"Mean correlation: {np.mean(valid_correlations):.4f}")
        print(f"Median correlation: {np.median(valid_correlations):.4f}")
        print(f"Std deviation: {np.std(valid_correlations):.4f}")
        print(f"Min correlation: {np.min(valid_correlations):.4f}")
        print(f"Max correlation: {np.max(valid_correlations):.4f}")
        print(f"Total years analyzed: {len(valid_correlations)}")
    
    # Save results to CSV
    if results:
        df_results = pd.DataFrame(results)
        output_path = script_dir.parent / "results" / "spearman_correlations.csv"
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()