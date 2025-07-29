import pandas as pd
from pathlib import Path
from collections import defaultdict

# read path
dir_root_path = Path(__file__).parent.parent
raw_data_path = dir_root_path / "raw" / "raw_data.csv"
output_dir = dir_root_path / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# read data
data_frame = pd.read_csv(raw_data_path)

year_to_rows = defaultdict(list)

# process per row
for _, row in data_frame.iterrows():
    year_str = row.iloc[1] 
    year_to_rows[year_str].append(row)

# write new file
for year_str, rows in year_to_rows.items():
    parts = year_str.split("/")
    if len(parts) == 2:
        start_year = parts[0]
        # end_year = str(int(parts[0][:2] + parts[1]))  # eg: 2024 + 25 -> 2025
        # file_name = f"{start_year}-{end_year}.csv"    # error at "1999-2000"~
        file_name = f"{start_year}-{int(start_year)+1}.csv"
    else:
        file_name = f"{year_str}.csv" 

    # save as csv
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_dir / file_name, index=False)
