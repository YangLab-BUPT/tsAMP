import os
import torch
import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 1. 基础配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_PATH = '/data2/lhmData/AMP/testmodel/AMP-BERT-main/' 
POS_FASTA = 
NEG_FASTA = #own dir_path

# 2. 数据处理类
class amp_data_fasta():
    def __init__(self, seqs, labels, tokenizer_path=MODEL_PATH, max_len=200):
        # 使用 local_files_only=True 强制从本地读取
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=False, local_files_only=True)
        self.max_len = max_len
        self.seqs = seqs
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 处理序列中的空格，确保 ProtBert 识别
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_len)
        
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])
        return sample

# 3. 读取 FASTA 函数
def read_fasta(path, label):
    sequences = []
    labels = []
    for record in SeqIO.parse(path, "fasta"):
        sequences.append(str(record.seq))
        labels.append(label)
    return sequences, labels

# 4. 指标计算函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp,
    }

def main():
    print("Loading FASTA files...")
    pos_seqs, pos_labels = read_fasta(POS_FASTA, 1) # 正样本标签为 1
    neg_seqs, neg_labels = read_fasta(NEG_FASTA, 0) # 负样本标签为 0
    
    test_seqs = pos_seqs + neg_seqs
    test_labels = pos_labels + neg_labels
    
    test_dataset = amp_data_fasta(test_seqs, test_labels, tokenizer_path=MODEL_PATH)
    print(f"Total samples: {len(test_dataset)} (Pos: {len(pos_labels)}, Neg: {len(neg_labels)})")

    test_args = TrainingArguments(
        output_dir='./temp_results',
        per_device_eval_batch_size=8,
        do_train=False,
        do_eval=True,
        fp16=True if torch.cuda.is_available() else False,
    )

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

    trainer = Trainer(
        model=model,
        args=test_args,
        compute_metrics=compute_metrics,
    )

    print("Running evaluation...")
    metrics = trainer.evaluate(test_dataset)

    print("\n" + "="*30)
    print("Evaluation Results:")
    #print(f"Precision: {metrics['eval_precision']:.4f}")
    #print(f"Recall:    {metrics['eval_recall']:.4f}")
    #print(f"F1 Score:  {metrics['eval_f1']:.4f}")
    #print(f"Accuracy:  {metrics['eval_accuracy']:.4f}")
   
    print("="*30)

if __name__ == "__main__":
    main()