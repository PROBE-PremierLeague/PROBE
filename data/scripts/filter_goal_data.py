import pandas as pd
from pathlib import Path
import os

# Set paths
script_dir = Path(__file__).parent
root_dir = script_dir.parent
processed_dir = root_dir / "processed" / "year-splitted"
output_dir = root_dir / "processed" / "goal-filtered"

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Seasons to process
seasons = []
for year in range(1993, 2024):
    seasons.append(f"{year}-{year+1}")


def process_season_file(season):
    """Process single season file"""
    input_file = processed_dir / f"{season}.csv"
    output_file = output_dir / f"{season}.csv"

    if not input_file.exists():
        print(f"Warning: File {input_file} not found, skipping")
        return

    try:
        # Read original data
        df = pd.read_csv(input_file)

        # Check if required columns exist
        # Only require basic columns, optional columns will be included if available
        required_columns = ["HomeTeam", "AwayTeam", "FTH Goals", "FTA Goals"]
        optional_columns = [
            "H Shots",
            "A Shots",
            "H SOT",
            "A SOT",
            "H Corners",
            "A Corners",
        ]

        # Check required columns
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            print(
                f"Error: Missing required columns in {input_file}: {missing_required}"
            )
            return

        # Check which optional columns are available
        available_optional = [col for col in optional_columns if col in df.columns]

        # Build final column list and new column names
        final_columns = required_columns + available_optional
        new_column_names = ["HomeTeam", "AwayTeam", "HomeGoal", "AwayGoal"]

        # Add new names for available optional columns
        if "H Shots" in available_optional:
            new_column_names.append("HomeShots")
        if "A Shots" in available_optional:
            new_column_names.append("AwayShots")
        if "H SOT" in available_optional:
            new_column_names.append("HomeShotOnTarget")
        if "A SOT" in available_optional:
            new_column_names.append("AwayShotOnTarget")
        if "H Corners" in available_optional:
            new_column_names.append("HomeCorners")
        if "A Corners" in available_optional:
            new_column_names.append("AwayCorners")

        # Select available columns and rename
        filtered_df = df[final_columns].copy()
        filtered_df.columns = new_column_names

        # Handle missing values
        # Only drop rows where required columns (HomeTeam, AwayTeam, HomeGoal, AwayGoal) have missing values
        filtered_df = filtered_df.dropna(
            subset=["HomeTeam", "AwayTeam", "HomeGoal", "AwayGoal"]
        )

        # Ensure numeric columns are integers where appropriate
        # Convert required columns to integers (these should not have NaN after dropna)
        filtered_df["HomeGoal"] = filtered_df["HomeGoal"].astype(int)
        filtered_df["AwayGoal"] = filtered_df["AwayGoal"].astype(int)

        # Convert optional columns to integers if they exist and handle NaN values
        if "HomeShots" in filtered_df.columns:
            filtered_df["HomeShots"] = (
                pd.to_numeric(filtered_df["HomeShots"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        if "AwayShots" in filtered_df.columns:
            filtered_df["AwayShots"] = (
                pd.to_numeric(filtered_df["AwayShots"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        if "HomeCorners" in filtered_df.columns:
            filtered_df["HomeCorners"] = (
                pd.to_numeric(filtered_df["HomeCorners"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        if "AwayCorners" in filtered_df.columns:
            filtered_df["AwayCorners"] = (
                pd.to_numeric(filtered_df["AwayCorners"], errors="coerce")
                .fillna(0)
                .astype(int)
            )

        # Remove columns that are all zeros (indicating missing data in original)
        columns_to_drop = []
        for col in filtered_df.columns:
            if col not in ["HomeTeam", "AwayTeam"]:  # Don't check team name columns
                if (filtered_df[col] == 0).all():  # If all values are 0
                    columns_to_drop.append(col)

        if columns_to_drop:
            filtered_df = filtered_df.drop(columns=columns_to_drop)

        # Save processed data
        filtered_df.to_csv(output_file, index=False)

    except Exception as e:
        print(f"Error processing {season}: {str(e)}")


def main():
    """Main function"""
    print("Processing football match data...")

    # Process each season
    for season in seasons:
        process_season_file(season)

    print("Processing completed!")


if __name__ == "__main__":
    main()
