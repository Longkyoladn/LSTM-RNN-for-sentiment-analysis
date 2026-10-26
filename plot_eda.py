import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_eda_charts(csv_path="data/IMDB Dataset.csv", output_path="plots/imdb_eda.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(csv_path)
    df['word_count'] = df['review'].apply(lambda x: len(x.split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Class Distribution
    sns.countplot(data=df, x='sentiment', ax=axes[0], palette='coolwarm')
    axes[0].set_title('Sentiment Distribution (50,000 Reviews)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Reviews')
    axes[0].set_xlabel('Sentiment')
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height()):,}', 
                         (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontsize=12, fontweight='bold')

    # 2. Word Count Distribution
    median_val = int(df['word_count'].median())
    p95_val = int(df['word_count'].quantile(0.95))
    sns.histplot(data=df, x='word_count', hue='sentiment', bins=50, kde=True, ax=axes[1], palette='coolwarm', binrange=(0, 1000))
    axes[1].set_title('Word Count Distribution (Clipped at 1000 words)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Words')
    axes[1].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val}')
    axes[1].axvline(p95_val, color='orange', linestyle='--', linewidth=2, label=f'95th percentile: {p95_val}')
    axes[1].axvline(500, color='red', linestyle=':', linewidth=2, label='Common LSTM MaxLen = 500')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    generate_eda_charts()
