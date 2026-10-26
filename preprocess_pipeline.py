"""
IMDB Dataset Preprocessing Pipeline
Tailored for LSTM / RNN Sentiment Analysis.

Features:
1. Label conversion: negative -> 0, positive -> 1
2. Deduplication: removes all duplicate reviews (418 samples)
3. Noise Cleaning:
   - Removes HTML tags (replacing <br /> with whitespace)
   - Removes URLs / web links
   - Normalizes curly apostrophes and expands contractions (e.g., don't -> do not)
   - PRESERVES punctuation, special characters, and emojis for sarcasm detection
   - Normalizes whitespace
4. 95th Percentile Sequence Thresholding:
   - Reviews <= 95th percentile length (<= 585 words) are split into Train (70%), Val (20%), Test (10%)
   - Reviews > 95th percentile length (top 5%, > 585 words) are EXCLUDED from Train
   - Long reviews are split 50/50 between Val and Test
   - Long reviews are chunked into sub-segments < 600 words (chunk_size = 500 words)
   - Each chunk is tagged with review_id, chunk_id, and total_chunks to allow average probability aggregation during evaluation.
5. Saves clean datasets:
   - data/train.csv
   - data/val.csv
   - data/test.csv
"""

import sys
import io

# Đảm bảo UTF-8 trên Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import re
import json
import html
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Từ điển mở rộng viết tắt tiếng Anh (Contractions Dictionary)
CONTRACTIONS_DICT = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
    "mustn't": "must not", "needn't": "need not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what're": "what are", "what's": "what is", "what've": "what have", "where's": "where is",
    "who'll": "who will", "who's": "who is", "who've": "who have", "won't": "will not",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are"
}

CONTRACTIONS_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS_DICT.keys()) + r')\b',
    re.IGNORECASE
)

def expand_contractions(text: str) -> str:
    """Mở rộng từ viết tắt tiếng Anh, chuẩn hóa dấu nháy cong sang thẳng."""
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    def replace(match):
        w = match.group(0).lower()
        expanded = CONTRACTIONS_DICT.get(w, w)
        # Giữ nguyên chữ hoa đầu từ nếu từ gốc viết hoa
        if match.group(0)[0].isupper():
            return expanded.capitalize()
        return expanded
    return CONTRACTIONS_RE.sub(replace, text)

def clean_review_text(text: str) -> str:
    """
    Làm sạch văn bản theo yêu cầu:
    - Xóa thẻ HTML (chuyển <br /> thành khoảng trắng để tránh dính từ)
    - Xóa URLs và web links
    - Mở rộng từ viết tắt (don't -> do not)
    - GIỮ NGUYÊN các ký tự đặc biệt, dấu câu (!, ?, ...) và emojis để hỗ trợ nhận diện mỉa mai (sarcasm)
    - Chuẩn hóa khoảng trắng thừa
    """
    # 1. Unescape HTML entities
    text = html.unescape(text)

    # 2. Xử lý thẻ HTML
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Xóa URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 4. Mở rộng từ viết tắt
    text = expand_contractions(text)

    # 5. Chuẩn hóa khoảng trắng (giữ nguyên emojis và dấu câu)
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    text = re.sub(r'\n+', '\n', text).strip()

    return text

def chunk_long_text(text: str, max_chunk_words: int = 500) -> List[str]:
    """Chia văn bản dài thành các đoạn nhỏ dưới 600 từ (mặc định 500 từ)."""
    words = text.split()
    if len(words) <= max_chunk_words:
        return [text]
    
    chunks = []
    for i in range(0, len(words), max_chunk_words):
        chunk_words = words[i:i + max_chunk_words]
        chunks.append(" ".join(chunk_words))
    return chunks

def format_short_dataframe(df_subset: pd.DataFrame) -> pd.DataFrame:
    """Định dạng nhóm câu ngắn (<= 95%), không cần chunking vì đã < 600 từ."""
    records = []
    for _, row in df_subset.iterrows():
        records.append({
            'review_id': row['review_id'],
            'chunk_id': 0,
            'total_chunks': 1,
            'is_chunk': False,
            'word_count': row['word_count'],
            'review': row['cleaned_review'],
            'sentiment': row['sentiment']
        })
    return pd.DataFrame(records)

def chunk_long_dataframe(df_subset: pd.DataFrame, max_chunk_words: int = 500) -> pd.DataFrame:
    """Chia nhỏ nhóm 5% câu dài thành các phần < 600 từ (mặc định 500 từ)."""
    records = []
    for _, row in df_subset.iterrows():
        rev_id = row['review_id']
        sentiment = row['sentiment']
        raw_text = row['cleaned_review']
        
        chunks = chunk_long_text(raw_text, max_chunk_words=max_chunk_words)
        total_chunks = len(chunks)
        for idx, ch in enumerate(chunks):
            records.append({
                'review_id': rev_id,
                'chunk_id': idx,
                'total_chunks': total_chunks,
                'is_chunk': True,
                'word_count': len(ch.split()),
                'review': ch,
                'sentiment': sentiment
            })
    return pd.DataFrame(records)

def run_preprocessing_pipeline(
    input_path: str = "data/IMDB Dataset.csv",
    output_dir: str = "data",
    seed: int = 42,
    max_chunk_words: int = 500
):
    print("=" * 75)
    print("        TIẾN HÀNH TIỀN XỬ LÝ VÀ PHÂN CHIA DỮ LIỆU IMDB (LSTM / RNN)")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Tải dữ liệu gốc
    print("\n[Bước 1] Tải dữ liệu gốc...")
    df = pd.read_csv(input_path)
    print(f"-> Số lượng mẫu ban đầu: {len(df):,}")

    # 2. Xử lý trùng lặp
    print("\n[Bước 2] Xử lý trùng lặp (Deduplication)...")
    init_count = len(df)
    df = df.drop_duplicates(subset=['review']).reset_index(drop=True)
    dup_removed = init_count - len(df)
    print(f"-> Đã loại bỏ {dup_removed:,} mẫu trùng lặp. Còn lại {len(df):,} mẫu sạch.")

    # 3. Mã hóa nhãn (positive: 1, negative: 0)
    print("\n[Bước 3] Chuyển đổi nhãn Sentiment (negative -> 0, positive -> 1)...")
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    print(f"-> Phân bố nhãn: 1 (Positive) = {(df['sentiment'] == 1).sum():,}, 0 (Negative) = {(df['sentiment'] == 0).sum():,}")

    # 4. Làm sạch văn bản (Giữ nguyên dấu câu và emojis)
    print("\n[Bước 4] Làm sạch văn bản (Xóa HTML, URL, mở rộng viết tắt, giữ emojis & dấu câu)...")
    df['cleaned_review'] = df['review'].apply(clean_review_text)
    df['word_count'] = df['cleaned_review'].apply(lambda x: len(x.split()))
    # Gán ID duy nhất cho từng review
    df['review_id'] = [f"rev_{i:06d}" for i in range(len(df))]

    # 5. Xác định ngưỡng phân vị 95%
    p95_threshold = df['word_count'].quantile(0.95)
    print(f"\n[Bước 5] Xác định ngưỡng độ dài 95% (95th Percentile): {p95_threshold:.1f} từ")
    
    df_short = df[df['word_count'] <= p95_threshold].reset_index(drop=True)
    df_long = df[df['word_count'] > p95_threshold].reset_index(drop=True)

    print(f"-> Nhóm câu <= 95% (Dùng cho Train, Val, Test): {len(df_short):,} review ({len(df_short)/len(df)*100:.2f}%)")
    print(f"-> Nhóm câu dài > 95% (CHỈ dùng cho Val và Test): {len(df_long):,} review ({len(df_long)/len(df)*100:.2f}%)")

    # 6. Phân chia tập dữ liệu
    # Tỷ lệ người dùng chọn: 70% Train, 20% Val, 10% Test trên tập <= 95%
    print("\n[Bước 6] Phân chia tập dữ liệu...")
    # Tách Train (70%) và Phần còn lại (30%)
    train_short, remain_short = train_test_split(
        df_short, test_size=0.30, random_state=seed, stratify=df_short['sentiment']
    )
    # Tách 30% còn lại thành Val (20/30 = 2/3) và Test (10/30 = 1/3)
    val_short, test_short = train_test_split(
        remain_short, test_size=(1/3), random_state=seed, stratify=remain_short['sentiment']
    )

    print(f"-> Phân chia nhóm ngắn: Train={len(train_short):,} (70%), Val={len(val_short):,} (20%), Test={len(test_short):,} (10%)")

    # Chia đều nhóm câu dài 50/50 cho Val và Test
    val_long, test_long = train_test_split(
        df_long, test_size=0.50, random_state=seed, stratify=df_long['sentiment']
    )
    print(f"-> Phân chia nhóm dài (50/50): Val nhận {len(val_long):,} câu dài, Test nhận {len(test_long):,} câu dài.")

    # 7. Xử lý chunking cho 5% câu dài trong Val và Test (< 600 từ mỗi chunk)
    print(f"\n[Bước 7] Cắt nhỏ các câu dài (> 95%) thành các phần < 600 từ (chunk_size={max_chunk_words})...")
    
    # Train chỉ bao gồm câu ngắn (không dùng câu dài, 100% không chunk)
    train_df = format_short_dataframe(train_short)
    
    # Val = câu ngắn (không chunk) + câu dài đã cắt nhỏ thành các chunks
    val_short_df = format_short_dataframe(val_short)
    val_long_df = chunk_long_dataframe(val_long, max_chunk_words=max_chunk_words)
    val_df = pd.concat([val_short_df, val_long_df], ignore_index=True)

    # Test = câu ngắn (không chunk) + câu dài đã cắt nhỏ thành các chunks
    test_short_df = format_short_dataframe(test_short)
    test_long_df = chunk_long_dataframe(test_long, max_chunk_words=max_chunk_words)
    test_df = pd.concat([test_short_df, test_long_df], ignore_index=True)

    print("\n[Bước 8] Thống kê tập dữ liệu cuối cùng:")
    print(f"   * Tập TRAIN: {len(train_df):,} mẫu (100% câu đơn <= {int(p95_threshold)} từ, không chứa câu dài)")
    print(f"     - Số review gốc: {train_df['review_id'].nunique():,}")
    print(f"     - Positive: {(train_df['sentiment']==1).sum():,}, Negative: {(train_df['sentiment']==0).sum():,}")

    val_chunks_count = (val_df['is_chunk'] == True).sum()
    print(f"   * Tập VAL: {len(val_df):,} mẫu (gồm {len(val_short):,} câu ngắn + {val_chunks_count:,} chunks từ {len(val_long):,} câu dài)")
    print(f"     - Số review gốc đại diện: {val_df['review_id'].nunique():,}")
    print(f"     - Positive: {(val_df['sentiment']==1).sum():,}, Negative: {(val_df['sentiment']==0).sum():,}")

    test_chunks_count = (test_df['is_chunk'] == True).sum()
    print(f"   * Tập TEST: {len(test_df):,} mẫu (gồm {len(test_short):,} câu ngắn + {test_chunks_count:,} chunks từ {len(test_long):,} câu dài)")
    print(f"     - Số review gốc đại diện: {test_df['review_id'].nunique():,}")
    print(f"     - Positive: {(test_df['sentiment']==1).sum():,}, Negative: {(test_df['sentiment']==0).sum():,}")

    # 9. Lưu các file CSV
    print("\n[Bước 9] Lưu file dữ liệu...")
    cols = ['review_id', 'chunk_id', 'total_chunks', 'is_chunk', 'word_count', 'review', 'sentiment']
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df[cols].to_csv(train_path, index=False, encoding='utf-8')
    val_df[cols].to_csv(val_path, index=False, encoding='utf-8')
    test_df[cols].to_csv(test_path, index=False, encoding='utf-8')

    # Lưu metadata tóm tắt
    summary = {
        "total_initial_reviews": init_count,
        "duplicates_removed": dup_removed,
        "clean_reviews_count": len(df),
        "p95_threshold_words": float(p95_threshold),
        "max_chunk_words": max_chunk_words,
        "train": {
            "samples": len(train_df),
            "unique_reviews": int(train_df['review_id'].nunique()),
            "positive_count": int((train_df['sentiment']==1).sum()),
            "negative_count": int((train_df['sentiment']==0).sum())
        },
        "val": {
            "samples": len(val_df),
            "unique_reviews": int(val_df['review_id'].nunique()),
            "short_reviews": len(val_short),
            "long_reviews": len(val_long),
            "chunk_samples": int(val_chunks_count),
            "positive_count": int((val_df['sentiment']==1).sum()),
            "negative_count": int((val_df['sentiment']==0).sum())
        },
        "test": {
            "samples": len(test_df),
            "unique_reviews": int(test_df['review_id'].nunique()),
            "short_reviews": len(test_short),
            "long_reviews": len(test_long),
            "chunk_samples": int(test_chunks_count),
            "positive_count": int((test_df['sentiment']==1).sum()),
            "negative_count": int((test_df['sentiment']==0).sum())
        }
    }
    with open(os.path.join(output_dir, "preprocessing_summary.json"), "w", encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"-> Đã lưu thành công:")
    print(f"   + Train: {train_path}")
    print(f"   + Val:   {val_path}")
    print(f"   + Test:  {test_path}")
    print(f"   + Summary: {os.path.join(output_dir, 'preprocessing_summary.json')}")
    print("=" * 75)

if __name__ == "__main__":
    run_preprocessing_pipeline()
