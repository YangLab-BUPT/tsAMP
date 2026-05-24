import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import set_seed
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
import argparse

from transformers import AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_checkpoint1 = "/data2/lhmData/AMP/testmodel/diff-amp/esm2_t12_35M_UR50D"




tokenizer = AutoTokenizer.from_pretrained(model_checkpoint1)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint1)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert1 = AutoModelForSequenceClassification.from_pretrained(model_checkpoint1, num_labels=3000).to(device)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(3000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        with torch.no_grad():
            bert_output = self.bert1(input_ids=x['input_ids'].to(device),
                                   attention_mask=x['attention_mask'].to(device))
        output_feature = self.dropout(bert_output["logits"])
        output_feature = self.dropout(self.relu(self.bn1(self.fc1(output_feature))))
        output_feature = self.dropout(self.relu(self.bn2(self.fc2(output_feature))))
        output_feature = self.dropout(self.relu(self.bn3(self.fc3(output_feature))))
        output_feature = self.dropout(self.output_layer(output_feature))
        return torch.softmax(output_feature, dim=1)

def AMP(test_sequences, model):
    max_len = 18
    test_data = tokenizer(test_sequences, max_length=max_len, padding="max_length", truncation=True,
                        return_tensors='pt')
    model = model.to(device)
    model.eval()
    out_probability = []
    with torch.no_grad():
        predict = model(test_data).to(device)
        out_probability.extend(np.max(np.array(predict.cpu()), axis=1).tolist())
        test_argmax = np.argmax(predict.cpu(), axis=1).tolist()
    id2str = {0: "non-AMP", 1: "AMP"}
    return id2str[test_argmax[0]], out_probability[0]

def read_fasta(file_path):
    """Read sequences from a FASTA file"""
    sequences = []
    with open(file_path, 'r') as f:
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append(''.join(current_seq))
    return sequences

def main():
    parser = argparse.ArgumentParser(description='AMP Sequence Classifier from FASTA file')
    parser.add_argument('--input_fasta', default='/data2/lhmData/AMP/test/Non-AMPs.fasta',help='Input FASTA file containing protein sequences')
    parser.add_argument('--output', default='Non-AMPs.txt', help='Output file for predictions')
    parser.add_argument('--amp_output', default='amp_sequences.txt', help='Output file for AMP sequences only')
    args = parser.parse_args()

    # Load model
    model = MyModel()
    model.load_state_dict(torch.load("weight/best_model.pth", map_location=torch.device(device)))
    model.eval()

    # Read sequences from FASTA file
    sequences = read_fasta(args.input_fasta)
    print(f"Found {len(sequences)} sequences in {args.input_fasta}")

    amp_count = 0
    non_amp_count = 0

    print('\nStarting classification...')
    with open(args.output, 'w') as outfile, open(args.amp_output, 'w') as ampfile:
        for seq in tqdm(sequences, desc="Processing sequences"):
            seq = seq.strip().upper()
            # Skip sequences with invalid amino acids
            valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
            if not all(aa in valid_amino_acids for aa in seq) or len(seq) < 3:
                outfile.write(f"Invalid sequence: {seq}\n")
                non_amp_count += 1
                continue

            result, probability = AMP(seq, model)
            outfile.write(f"{seq}\t{result}\t{probability:.4f}\n")

            if result == "AMP":
                amp_count += 1
                ampfile.write(f"{seq}\t{probability:.4f}\n")
            else:
                non_amp_count += 1

    print("\nClassification finished")
    print(f"AMP sequences: {amp_count}")
    print(f"Non-AMP sequences: {non_amp_count}")
    print(f"Results saved to {args.output}")
    print(f"AMP sequences saved to {args.amp_output}")

if __name__ == "__main__":
    main()