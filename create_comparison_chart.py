# create_comparison_chart.py
import matplotlib.pyplot as plt
import numpy as np

# Your data
metrics = ['Faithfulness', 'Answer Relevancy']
char_based = [0.55, 0.54]   # average of 4 old runs
article_based = [0.86, 0.64]  # the new run

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, char_based, width, label='Character-based chunking', color='#cccccc')
ax.bar(x + width/2, article_based, width, label='Article-based chunking', color='#4f8ef7')

ax.set_ylabel('Score')
ax.set_title('RAGAS Evaluation: Chunking Strategy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1)

# Annotate bars with values
for i, (a, b) in enumerate(zip(char_based, article_based)):
    ax.text(i - width/2, a + 0.02, f'{a:.2f}', ha='center')
    ax.text(i + width/2, b + 0.02, f'{b:.2f}', ha='center')

plt.tight_layout()
plt.savefig('chunking_comparison.png', dpi=150)
print('✅ Saved chunking_comparison.png')