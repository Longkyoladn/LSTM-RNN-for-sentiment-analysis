# Báo Cáo Hiệu Năng Mô Hình RNN (Sentiment Analysis)

## 1. Siêu Tham Số (Hyperparameters)
- **Mô hình**: Simple RNN (Unidirectional, 2 layers)
- **Embedding Dim**: 128
- **Hidden Dim**: 128
- **Dropout**: 0.3
- **Max Sequence Length**: 300
- **Batch Size**: 64
- **Epochs**: 15
- **Thiết bị**: cuda

## 2. Kết Quả Trên Tập Test (Đã gộp xác suất trung bình các chunk)
- **Accuracy**: **51.58%**
- **Precision**: 51.43%
- **Recall**: 86.19%
- **F1-Score**: **0.6442**
- **Loss**: 0.6925
- **Tổng số review gốc được đánh giá**: 5,950

