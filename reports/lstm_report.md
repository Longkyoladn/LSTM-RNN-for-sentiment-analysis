# Báo Cáo Hiệu Năng Mô Hình LSTM (Sentiment Analysis)



## 1. Siêu Tham Số (Hyperparameters)
- **Mô hình**: Bidirectional LSTM (Bi-LSTM) (2 layers)
- **Embedding Dim**: 128
- **Hidden Dim**: 128
- **Dropout**: 0.3
- **Max Sequence Length**: 300
- **Batch Size**: 64
- **Epochs**: 15
- **Thiết bị**: cuda

## 2. Kết Quả Trên Tập Test (Đã gộp xác suất trung bình các chunk)
- **Accuracy**: **87.82%**
- **Precision**: 87.71%
- **Recall**: 88.43%
- **F1-Score**: **0.8807**
- **Loss**: 0.5098
- **Tổng số review gốc được đánh giá**: 5,950

