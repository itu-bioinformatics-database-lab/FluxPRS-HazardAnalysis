import matplotlib.pyplot as plt
import numpy as np

# Data
labels = [
    "Metabolite Levels",
    "Pathway Diff Scores",
    "Reaction Diff Scores",
    "Metabolite Levels Pathway Combined",
    "Metabolite Levels Reaction Combined",
    "Transcriptomics",
    "Transcriptomics Metabolite Levels Combined",
    "Transcriptomics Pathway Diff Combined",
    "Transcriptomics Reaction Diff Combined"
]

odds_ratios = np.array([
    95.17,
    37.30,
    56.79,
    89.63,
    63.95,
    60.40,
    53.11,
    50.89,
    38.55
])

# Plot setup
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(labels))

# Plotting each odds ratio as a point
colors = plt.cm.tab10.colors
for i in range(len(labels)):
    ax.plot(odds_ratios[i], y_pos[i], 'o', color=colors[i % len(colors)])
    ax.text(odds_ratios[i] + 1, y_pos[i], f"p = {odds_ratios[i]:.2f}", va='center', color='gray')

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Odds Ratio")
ax.set_title("Forest Plot of Odds Ratios")
ax.set_xlim(30, 100)
ax.tick_params(axis='x', which='both', labelrotation=0)
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
