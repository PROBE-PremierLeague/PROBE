import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import matplotlib.patches as patches

# Set matplotlib backend to non-interactive
plt.switch_backend('Agg')

# Add font path and set up Times New Roman
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 1.5

# Define colors
ATTACK_COLOR = '#E8425A'
DEFENSE_COLOR = '#55babe'

# Read and sort data
data = pd.read_csv('../results/linear_regression/mle-L-BFGS-B-lambdas/team_data_2024_2025_no_home_advantage.csv')
data = data.sort_values('Attack', ascending=True)  # Sort by attack strength

# Create figure
fig = plt.figure(figsize=(16, 8))

# Create main axis
ax_main = plt.gca()

# Plot data
teams = data['Team'].values
x = np.arange(len(teams))
attack = data['Attack'].values
defense = data['Defense'].values

# Main scatter plot
scatter_size = 300  # Increased scatter point size
ax_main.scatter(x, attack, s=scatter_size, c=ATTACK_COLOR, alpha=0.7, label='Attack Strength')
ax_main.scatter(x, defense, s=scatter_size, c=DEFENSE_COLOR, alpha=0.7, label='Defense Strength')

# Connect attack and defense points with lines
for i in range(len(teams)):
    ax_main.plot([x[i], x[i]], [attack[i], defense[i]], 
                 color='gray', alpha=0.3, linestyle='--', zorder=1)

# Styling main plot
ax_main.set_xlabel('Teams', fontsize=14, fontweight='bold', labelpad=10)
ax_main.set_ylabel('Performance Metrics', fontsize=14, fontweight='bold', labelpad=10)
ax_main.grid(True, linestyle='--', alpha=0.3)
ax_main.set_xticks(x)
ax_main.set_xticklabels(teams, rotation=45, ha='right', fontsize=12, fontweight='bold')
ax_main.tick_params(axis='y', labelsize=12)

# Add legend in the upper left
legend = ax_main.legend(loc='upper left', fontsize=12, framealpha=0.9, 
                       bbox_to_anchor=(0.02, 0.98))
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('none')

# Add title
plt.title('Premier League Team Performance Analysis', 
          fontsize=20, fontweight='bold', pad=20)

# Add explanatory text
explanation = ('Note: Higher attack values and lower defense values\n'
              'indicate better team performance')
fig.text(0.98, 0.02, explanation, 
         fontsize=12, style='italic', 
         ha='right', va='bottom')

# Add statistical summary
stats_text = (f'Mean Attack: {np.mean(attack):.2f}\n'
             f'Mean Defense: {np.mean(defense):.2f}')
fig.text(0.02, 0.02, stats_text, 
         fontsize=12, ha='left', va='bottom')

# Adjust layout
plt.tight_layout()

# Save with high quality
plt.savefig('../results/linear_regression/mle-L-BFGS-B-lambdas/team_performance_analysis.png',
            dpi=600,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.close()