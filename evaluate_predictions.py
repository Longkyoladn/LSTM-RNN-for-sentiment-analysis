"""
Evaluation & Inference Utility for Aggregating Chunk Probabilities
When evaluating models on validation or test sets where long reviews are split into chunks (< 600 words),
this module calculates the final probability for each original review as the average of its chunk probabilities:
    P(Review) = mean(P(Chunk_1), P(Chunk_2), ..., P(Chunk_K))
"""

import sys
import io

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from typing import Tuple, Dict

def aggregate_chunk_predictions(
    df_eval: pd.DataFrame,
    prob_col: str = "pred_prob",
    label_col: str = "sentiment",
    review_id_col: str = "review_id",
    threshold: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Gộp xác suất của các đoạn chunk về cấp độ từng câu review gốc:
    - Câu đơn lẻ (không bị cắt): giữ nguyên xác suất dự đoán.
    - Câu dài (bị chia thành K chunks): lấy trung bình cộng xác suất của K chunks.

    Tham số:
    - df_eval: DataFrame chứa kết quả dự đoán của từng dòng (chunk hoặc full review).
    - prob_col: Tên cột chứa xác suất dự đoán lớp 1 (Positive) [0.0 -> 1.0].
    - label_col: Tên cột chứa nhãn thực tế (0 hoặc 1).
    - review_id_col: Cột định danh câu review gốc ('review_id').
    - threshold: Ngưỡng quyết định nhãn (mặc định 0.5).

    Trả về:
    - review_level_df: DataFrame ở cấp độ từng review duy nhất kèm xác suất gộp.
    - metrics: Dictionary chứa Accuracy, Precision, Recall, F1 ở cấp độ review.
    """
    # Gom nhóm theo review_id và tính trung bình xác suất
    review_level = df_eval.groupby(review_id_col).agg(
        final_prob=(prob_col, 'mean'),
        true_label=(label_col, 'first'),
        total_chunks=('chunk_id', 'count')
    ).reset_index()

    # Phân loại nhãn dự đoán dựa trên ngưỡng
    review_level['pred_label'] = (review_level['final_prob'] >= threshold).astype(int)

    # Đánh giá các chỉ số
    y_true = review_level['true_label'].values
    y_pred = review_level['pred_label'].values

    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "total_reviews": int(len(review_level)),
        "single_chunk_reviews": int((review_level['total_chunks'] == 1).sum()),
        "multi_chunk_reviews": int((review_level['total_chunks'] > 1).sum())
    }

    return review_level, metrics

if __name__ == "__main__":
    # Ví dụ kiểm tra hoạt động của module tính xác suất gộp
    mock_data = pd.DataFrame([
        # Review ngắn (1 chunk)
        {"review_id": "rev_0001", "chunk_id": 0, "sentiment": 1, "pred_prob": 0.85},
        {"review_id": "rev_0002", "chunk_id": 0, "sentiment": 0, "pred_prob": 0.15},
        # Review dài bị chia thành 3 chunks (chunk 0, 1, 2)
        # Giả sử mô hình dự đoán xác suất từng chunk là: 0.60, 0.70, 0.80 -> Trung bình: 0.70 (Positive)
        {"review_id": "rev_0003", "chunk_id": 0, "sentiment": 1, "pred_prob": 0.60},
        {"review_id": "rev_0003", "chunk_id": 1, "sentiment": 1, "pred_prob": 0.70},
        {"review_id": "rev_0003", "chunk_id": 2, "sentiment": 1, "pred_prob": 0.80},
    ])

    print("--- DỮ LIỆU ĐẦU VÀO MOCK (CẤP ĐỘ CHUNK): ---")
    print(mock_data)

    rev_df, metrics = aggregate_chunk_predictions(mock_data)
    print("\n--- KẾT QUẢ GỘP XÁC SUẤT (CẤP ĐỘ REVIEW): ---")
    print(rev_df)
    print("\n--- CHỈ SỐ ĐÁNH GIÁ: ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
