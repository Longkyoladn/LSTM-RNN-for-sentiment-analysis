import re
import sys
import io

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi bảng mã cp1252
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from collections import Counter

def run_eda(csv_path="data/IMDB Dataset.csv"):
    print("=" * 70)
    print("           IMDB DATASET EXPLORATION & STATISTICAL ANALYSIS")
    print("=" * 70)

    # 1. Load Data
    df = pd.read_csv(csv_path)
    print(f"\n1. TỔNG QUAN DỮ LIỆU:")
    print(f"   - Số lượng mẫu (rows): {len(df):,}")
    print(f"   - Số lượng cột (columns): {df.shape[1]}")
    print(f"   - Danh sách cột: {list(df.columns)}")
    print(f"   - Bộ nhớ sử dụng: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # 2. Kiểm tra giá trị rỗng (Missing / Null)
    print(f"\n2. GIÁ TRỊ THIẾU (MISSING / NULL VALUES):")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        print(f"   - {col}: {count} ({count / len(df) * 100:.2f}%)")

    # 3. Kiểm tra dữ liệu trùng lặp (Duplicates)
    exact_duplicates = df.duplicated().sum()
    review_duplicates = df.duplicated(subset=['review']).sum()
    print(f"\n3. DỮ LIỆU TRÙNG LẶP (DUPLICATES):")
    print(f"   - Số hàng trùng lặp hoàn toàn (cả review & sentiment): {exact_duplicates:,}")
    print(f"   - Số hàng trùng nội dung review: {review_duplicates:,}")

    # 4. Phân phối nhãn (Sentiment Distribution)
    print(f"\n4. PHÂN PHỐI NHÃN (SENTIMENT TARGET DISTRIBUTION):")
    sentiment_counts = df['sentiment'].value_counts()
    for label, count in sentiment_counts.items():
        print(f"   - {label}: {count:,} ({count / len(df) * 100:.2f}%)")

    # 5. Phân tích độ dài văn bản (Length & Word Counts)
    print(f"\n5. PHÂN TÍCH ĐỘ DÀI VĂN BẢN (TEXT LENGTH ANALYSIS):")
    df['char_count'] = df['review'].apply(len)
    df['word_count'] = df['review'].apply(lambda x: len(x.split()))

    print("   [Thống kê số lượng từ / Review (Word Count)]:")
    quantiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q_vals = df['word_count'].quantile(quantiles)
    print(f"   - Min: {df['word_count'].min()}")
    print(f"   - Mean: {df['word_count'].mean():.2f}")
    print(f"   - Std: {df['word_count'].std():.2f}")
    for q, val in zip(quantiles, q_vals):
        print(f"   - Percentile {int(q*100)}%: {val:.0f} từ")
    print(f"   - Max: {df['word_count'].max()} từ")

    # So sánh độ dài giữa positive và negative
    pos_words = df[df['sentiment'] == 'positive']['word_count']
    neg_words = df[df['sentiment'] == 'negative']['word_count']
    print(f"   - Độ dài trung bình review TÍCH CỰC: {pos_words.mean():.2f} từ (median: {pos_words.median():.0f})")
    print(f"   - Độ dài trung bình review TIÊU CỰC: {neg_words.mean():.2f} từ (median: {neg_words.median():.0f})")

    # 6. Kiểm tra các đặc trưng nhiễu văn bản (Noise Pattern Analysis)
    print(f"\n6. PHÂN TÍCH CÁC YẾU TỐ NHIỄU (NOISE PATTERNS):")
    html_pattern = re.compile(r'<[^>]+>')
    has_html = df['review'].apply(lambda x: bool(html_pattern.search(x)))
    print(f"   - Số review chứa thẻ HTML (như <br />): {has_html.sum():,} ({has_html.mean() * 100:.2f}%)")

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    has_url = df['review'].apply(lambda x: bool(url_pattern.search(x)))
    print(f"   - Số review chứa URL / Link web: {has_url.sum():,} ({has_url.mean() * 100:.2f}%)")

    non_ascii_pattern = re.compile(r'[^\x00-\x7F]')
    has_non_ascii = df['review'].apply(lambda x: bool(non_ascii_pattern.search(x)))
    print(f"   - Số review chứa ký tự Non-ASCII / Emojis: {has_non_ascii.sum():,} ({has_non_ascii.mean() * 100:.2f}%)")

    contractions_pattern = re.compile(r"\b(can't|won't|n't|'ve|'re|'d|'ll|'m|it's|that's)\b", re.IGNORECASE)
    has_contractions = df['review'].apply(lambda x: bool(contractions_pattern.search(x)))
    print(f"   - Số review chứa từ viết tắt tiếng Anh (don't, can't...): {has_contractions.sum():,} ({has_contractions.mean() * 100:.2f}%)")

    # 7. Thống kê từ vựng (Vocabulary & Most Frequent Words)
    print(f"\n7. THỐNG KÊ TỪ VỰNG BAN ĐẦU (INITIAL VOCABULARY):")
    all_words = []
    # Lấy mẫu 10,000 review để tính nhanh tần suất từ
    sample_df = df.sample(n=10000, random_state=42)
    for text in sample_df['review']:
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        all_words.extend(tokens)
    
    vocab = Counter(all_words)
    print(f"   - Tổng số từ đơn trong mẫu 10k: {len(all_words):,}")
    print(f"   - Số lượng từ vựng độc nhất (Unique Vocab) trong 10k mẫu: {len(vocab):,}")
    print(f"   - Top 15 từ xuất hiện nhiều nhất:")
    for word, cnt in vocab.most_common(15):
        print(f"     * '{word}': {cnt:,} lần")

    # 8. Mẫu văn bản gốc
    print(f"\n8. MẪU DỮ LIỆU GỐC (RAW REVIEW SAMPLES):")
    print("   --- [Mẫu 1 (Positive)]: ---")
    pos_sample = df[df['sentiment'] == 'positive'].iloc[0]['review']
    print(pos_sample[:300] + "..." if len(pos_sample) > 300 else pos_sample)
    
    print("\n   --- [Mẫu 2 (Negative)]: ---")
    neg_sample = df[df['sentiment'] == 'negative'].iloc[0]['review']
    print(neg_sample[:300] + "..." if len(neg_sample) > 300 else neg_sample)

    print("\n" + "=" * 70)

if __name__ == "__main__":
    run_eda()
