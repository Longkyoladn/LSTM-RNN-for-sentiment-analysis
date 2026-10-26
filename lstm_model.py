"""
LSTM Model & Training Pipeline for IMDB Sentiment Analysis
Supports both standalone training on Kaggle (GPU/CPU) and modular evaluation.
Results are saved to the `reports/` directory and will overwrite previous runs.
"""

import sys
import io

# Đảm bảo UTF-8 khi chạy trên console Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import re
import json
import argparse
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. TOKENIZER & VOCABULARY
# ==============================================================================

def simple_tokenizer(text: str) -> List[str]:
    """Tách từ giữ nguyên dấu câu và emojis để hỗ trợ nhận diện mỉa mai."""
    if not isinstance(text, str):
        return []
    return re.findall(r'\w+|[^\w\s]', text.lower())

class Vocabulary:
    """Bảng từ vựng Token <-> Index."""
    def __init__(self, pad_token="<PAD>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = 0
        self.unk_idx = 1
        self.word2idx = {pad_token: self.pad_idx, unk_token: self.unk_idx}
        self.idx2word = {self.pad_idx: pad_token, self.unk_idx: unk_token}
        self.word_counts = Counter()

    def build_vocab(self, texts: List[str], max_vocab_size: int = 25000, min_freq: int = 2):
        for text in texts:
            tokens = simple_tokenizer(text)
            self.word_counts.update(tokens)

        sorted_words = [w for w, c in self.word_counts.most_common() if c >= min_freq]
        for w in sorted_words:
            if len(self.word2idx) >= max_vocab_size:
                break
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w

    def encode(self, text: str, max_seq_len: int = 300) -> List[int]:
        tokens = simple_tokenizer(text)
        indices = [self.word2idx.get(t, self.unk_idx) for t in tokens[:max_seq_len]]
        # Padding
        if len(indices) < max_seq_len:
            indices += [self.pad_idx] * (max_seq_len - len(indices))
        return indices

# ==============================================================================
# 2. PYTORCH DATASET
# ==============================================================================

class IMDBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_seq_len: int = 300):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens_idx = self.vocab.encode(row['review'], max_seq_len=self.max_seq_len)
        return {
            'input_ids': torch.tensor(tokens_idx, dtype=torch.long),
            'label': torch.tensor(row['sentiment'], dtype=torch.float32),
            'review_id': row['review_id'],
            'chunk_id': int(row.get('chunk_id', 0)),
            'total_chunks': int(row.get('total_chunks', 1))
        }

# ==============================================================================
# 3. LSTM CLASSIFIER ARCHITECTURE
# ==============================================================================

class LSTMClassifier(nn.Module):
    """
    Mô hình LSTM / Bi-LSTM nhiều tầng cho Sentiment Analysis.
    Sử dụng nn.LSTM với Cell State & Hidden State, Dropout và Dense Classification Head.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        fc_in = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids) # (batch_size, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(embedded)

        # Trích xuất đặc trưng từ hidden state của tầng cuối cùng
        if self.bidirectional:
            # Ghép forward hidden state (h_n[-2]) và backward hidden state (h_n[-1])
            feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            feat = h_n[-1]

        logits = self.fc(feat).squeeze(-1) # (batch_size,)
        return logits

# ==============================================================================
# 4. EVALUATION & CHUNK PROBABILITY AGGREGATION
# ==============================================================================

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Đánh giá mô hình trên tập dữ liệu.
    Tự động gộp xác suất các chunks thuộc cùng một câu review gốc theo trung bình cộng:
    P(review) = mean(P(chunk_1), ..., P(chunk_K)).
    """
    model.eval()
    total_loss = 0.0
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()
            rev_ids = batch['review_id']
            chunk_ids = batch['chunk_id'].numpy()
            total_chunks = batch['total_chunks'].numpy()

            for r_id, c_id, t_c, prob, lbl in zip(rev_ids, chunk_ids, total_chunks, probs, labels_np):
                predictions.append({
                    'review_id': r_id,
                    'chunk_id': c_id,
                    'total_chunks': t_c,
                    'pred_prob': float(prob),
                    'label': int(lbl)
                })

    avg_loss = total_loss / len(dataloader.dataset)
    pred_df = pd.DataFrame(predictions)

    # Gộp xác suất trung bình theo từng review gốc
    review_level = pred_df.groupby('review_id').agg(
        final_prob=('pred_prob', 'mean'),
        true_label=('label', 'first'),
        total_chunks=('total_chunks', 'first')
    ).reset_index()

    review_level['pred_label'] = (review_level['final_prob'] >= 0.5).astype(int)

    y_true = review_level['true_label'].values
    y_pred = review_level['pred_label'].values

    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        'loss': float(avg_loss),
        'accuracy': float(acc),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_reviews': int(len(review_level)),
        'total_chunks': int(len(pred_df))
    }
    return avg_loss, metrics, review_level

# ==============================================================================
# 5. TRAINING PIPELINE & REPORT GENERATION
# ==============================================================================

def train_lstm(
    train_csv: str = "data/train.csv",
    val_csv: str = "data/val.csv",
    test_csv: str = "data/test.csv",
    reports_dir: str = "reports",
    checkpoints_dir: str = "checkpoints",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    max_vocab_size: int = 25000,
    max_seq_len: int = 300,
    bidirectional: bool = True,
    device: Optional[str] = None
):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"=== BẮT ĐẦU HUẤN LUYỆN LSTM MODEL TRÊN THIẾT BỊ: {dev} ===")

    # 1. Đọc dữ liệu
    print("-> Đang tải dữ liệu...")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # 2. Xây dựng bảng từ vựng từ tập Train
    print("-> Đang xây dựng Vocabulary từ tập Train...")
    vocab = Vocabulary()
    vocab.build_vocab(df_train['review'].tolist(), max_vocab_size=max_vocab_size)
    vocab_size = len(vocab.word2idx)
    print(f"-> Kích thước từ vựng: {vocab_size:,} tokens")

    # 3. Tạo DataLoaders
    train_dataset = IMDBDataset(df_train, vocab, max_seq_len=max_seq_len)
    val_dataset = IMDBDataset(df_val, vocab, max_seq_len=max_seq_len)
    test_dataset = IMDBDataset(df_test, vocab, max_seq_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Khởi tạo mô hình LSTM
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    ).to(dev)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    best_val_f1 = 0.0
    best_checkpoint_path = os.path.join(checkpoints_dir, "lstm_best.pt")

    # 5. Vòng lặp huấn luyện
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(dev)
            labels = batch['label'].to(dev)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train_samples += len(labels)

        train_loss = total_train_loss / total_train_samples
        train_acc = correct_train / total_train_samples

        # Đánh giá trên tập Validation
        val_loss, val_metrics, _ = evaluate_model(model, val_loader, criterion, dev)
        scheduler.step(val_metrics['f1_score'])

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_score'])

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% "
              f"| Val Loss: {val_loss:.4f} - Acc: {val_metrics['accuracy']*100:.2f}% - F1: {val_metrics['f1_score']:.4f}")

        # Lưu checkpoint tốt nhất theo Val F1
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> Lưu mô hình tốt nhất (Val F1: {best_val_f1:.4f}) vào {best_checkpoint_path}")

    # 6. Đánh giá cuối cùng trên tập Test bằng checkpoint tốt nhất
    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=dev))
        print(f"\n-> Đã tải lại checkpoint tốt nhất từ {best_checkpoint_path} để kiểm thử trên tập Test.")

    test_loss, test_metrics, _ = evaluate_model(model, test_loader, criterion, dev)
    print(f"\n=== KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST (LSTM) ===")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Test Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Test Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"Test F1-Score:  {test_metrics['f1_score']:.4f}")

    # 7. Ghi đè kết quả vào thư mục reports/ (JSON, Markdown, PNG)
    report_json_path = os.path.join(reports_dir, "lstm_report.json")
    report_md_path = os.path.join(reports_dir, "lstm_report.md")
    report_plot_path = os.path.join(reports_dir, "lstm_curves.png")

    final_report = {
        "model_type": "Bi-LSTM" if bidirectional else "LSTM",
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "max_seq_len": max_seq_len,
            "max_vocab_size": max_vocab_size,
            "device": str(dev)
        },
        "test_metrics": test_metrics,
        "history": history
    }

    # Ghi đè JSON report
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    # Ghi đè Markdown report
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write("# Báo Cáo Hiệu Năng Mô Hình LSTM (Sentiment Analysis)\n\n")
        f.write("Báo cáo này được tự động tạo và ghi đè sau mỗi lần huấn luyện.\n\n")
        f.write("## 1. Siêu Tham Số (Hyperparameters)\n")
        f.write(f"- **Mô hình**: {'Bidirectional LSTM (Bi-LSTM)' if bidirectional else 'Standard LSTM'} ({num_layers} layers)\n")
        f.write(f"- **Embedding Dim**: {embedding_dim}\n")
        f.write(f"- **Hidden Dim**: {hidden_dim}\n")
        f.write(f"- **Dropout**: {dropout}\n")
        f.write(f"- **Max Sequence Length**: {max_seq_len}\n")
        f.write(f"- **Batch Size**: {batch_size}\n")
        f.write(f"- **Epochs**: {epochs}\n")
        f.write(f"- **Thiết bị**: {dev}\n\n")
        f.write("## 2. Kết Quả Trên Tập Test (Đã gộp xác suất trung bình các chunk)\n")
        f.write(f"- **Accuracy**: **{test_metrics['accuracy']*100:.2f}%**\n")
        f.write(f"- **Precision**: {test_metrics['precision']*100:.2f}%\n")
        f.write(f"- **Recall**: {test_metrics['recall']*100:.2f}%\n")
        f.write(f"- **F1-Score**: **{test_metrics['f1_score']:.4f}**\n")
        f.write(f"- **Loss**: {test_loss:.4f}\n")
        f.write(f"- **Tổng số review gốc được đánh giá**: {test_metrics['total_reviews']:,}\n\n")

    # Vẽ biểu đồ loss và accuracy, ghi đè ảnh PNG
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_title('LSTM Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['epoch'], history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['epoch'], history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_title('LSTM Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(report_plot_path, dpi=150)
    plt.close()

    print(f"\n-> Báo cáo kết quả đã được ghi đè thành công tại:")
    print(f"   + JSON: {report_json_path}")
    print(f"   + Markdown: {report_md_path}")
    print(f"   + Biểu đồ: {report_plot_path}")
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình LSTM cho IMDB Sentiment Analysis")
    parser.add_argument("--train", action="store_true", help="Kích hoạt tiến trình huấn luyện mô hình")
    parser.add_argument("--epochs", type=int, default=15, help="Số lượng epoch (mặc định: 15)")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch (mặc định: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (mặc định: 0.001)")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Kích thước embedding (mặc định: 128)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Kích thước hidden state (mặc định: 128)")
    parser.add_argument("--num_layers", type=int, default=2, help="Số tầng LSTM (mặc định: 2)")
    parser.add_argument("--unidirectional", action="store_true", help="Sử dụng LSTM 1 chiều thay vì 2 chiều (Bi-LSTM)")
    parser.add_argument("--max_seq_len", type=int, default=300, help="Độ dài sequence tối đa (mặc định: 300)")
    parser.add_argument("--device", type=str, default=None, help="cuda hoặc cpu")

    args = parser.parse_args()

    if args.train:
        train_lstm(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            bidirectional=not args.unidirectional,
            max_seq_len=args.max_seq_len,
            device=args.device
        )
