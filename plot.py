import matplotlib.pyplot as plt

# Define your data
lookahead = [240, 120, 60, 30, 15, 5]
aurocs     = [0.528, 0.520, 0.488, 0.506, 0.528, 0.535]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(lookahead, aurocs, marker='o', linewidth=2)

# Red vertical line at AF onset (x=0)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='AF onset')

# Force ticks to be exactly the look‑ahead values
plt.xticks(lookahead)

# Labels, title, legend
plt.xlabel('AF onset forecast (mins)')
plt.ylabel('AUROC')
plt.title('Evolution of AUROC vs. AF Onset Forecast')
plt.legend()

# Y‑axis padding and grid
plt.ylim(min(aurocs) - 0.02, max(aurocs) + 0.02)
plt.grid(alpha=0.3)

# Layout and save
plt.tight_layout()
plt.savefig('auroc_vs_lookahead.png', dpi=300)
plt.close()
