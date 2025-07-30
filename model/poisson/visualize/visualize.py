import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import matplotlib.patches as patches

# Set matplotlib backend to non-interactive
plt.switch_backend('Agg')

# Add font path and set up Times New Roman
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
try:
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = ['Times New Roman']
except:
    print("warning: fail to load Times New Roman, will use default font.")
    plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['axes.linewidth'] = 1.5

# Define colors
ATTACK_COLOR = '#E8425A'
DEFENSE_COLOR = '#55babe'

# Read and sort data
base_dir = Path(__file__).parent.parent if '__file__' in locals() else Path.cwd()
file_path = base_dir / "lambdas" / "mle-L-BFGS-B-lambdas" / "2023-2024-parameters.csv"
#file_path = base_dir / "lambdas" / "mle-LS-lambdas" / "2023-2024-parameters.csv"
data = pd.read_csv(file_path)
data = data.sort_values('Attack', ascending=True)  # Sort by attack strength

# Create figure
fig = plt.figure(figsize=(12, 6))

# Create main axis
ax_main = plt.gca()

# Plot data
teams = data['Team'].values
x = np.arange(len(teams))
attack = data['Attack'].values
defense = data['Defend'].values

# Main scatter plot
scatter_size = 400  # Increased scatter point size
ax_main.scatter(x, attack, s=scatter_size, c=ATTACK_COLOR, alpha=0.7, label='Attack Strength')
ax_main.scatter(x, defense, s=scatter_size, c=DEFENSE_COLOR, alpha=0.7, label='Defense Strength')

# Connect attack and defense points with lines
for i in range(len(teams)):
    ax_main.plot([x[i], x[i]], [attack[i], defense[i]], 
                 color='gray', alpha=0.3, linestyle='--', zorder=1)

# Styling main plot
ax_main.set_xlabel('Teams', fontsize=18, fontweight='bold', labelpad=10)
ax_main.set_ylabel('Performance Metrics', fontsize=18, fontweight='bold', labelpad=10)
ax_main.grid(True, linestyle='--', alpha=0.3)
ax_main.set_xticks(x)
ax_main.set_xticklabels(teams, rotation=45, ha='right', fontsize=14, fontweight='bold')
ax_main.tick_params(axis='y', labelsize=14)

# Add legend in the upper left
legend = ax_main.legend(loc='upper left', fontsize=12, framealpha=0.9, 
                       bbox_to_anchor=(0.01, 0.98), labelspacing=1.1,
                       borderpad=0.8, borderaxespad=0.5)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('gray')

# Adjust layout
plt.tight_layout()

xlabel_obj = ax_main.xaxis.get_label()
xlabel_pos = xlabel_obj.get_position()
text_content = f"Home Advantage = {float(data['HomeAdvantage'][0]):.5f}"
ax_main.text(
    0.7,  
    0.02,  
    text_content,
    fontsize=16,  fontweight='bold',
    ha='left',  
    va='bottom',  
    transform=fig.transFigure  
)

# Save with high quality
base_dir = Path(__file__).parent.parent if '__file__' in locals() else Path.cwd()
save_path = base_dir / "lambdas" / "L-BFGS-B_analysis.png"
#save_path = base_dir / "lambdas" / "LS_analysis.png"

plt.savefig(str(save_path),
            dpi=600,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.close()