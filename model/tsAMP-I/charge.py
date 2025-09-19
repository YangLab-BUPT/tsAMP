from Bio import SeqIO
from Bio.SeqUtils import ProtParam
from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA
import numpy as np
import pandas as pd

# 氨基酸的疏水性值（Kyte-Doolittle量表）
hydrophobicity_values = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5,
    'F': 2.8,  'G': -0.4, 'H': -3.2, 'I': 4.5,
    'K': -3.9, 'L': 3.8,  'M': 1.9,  'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
    'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3,
}

def calculate_hydrophobicity(peptide_sequence, scale=hydrophobicity_values):
    total_hydrophobicity = 0.0
    for aa in peptide_sequence:
        if aa in scale:
            total_hydrophobicity += scale[aa]
    return -(total_hydrophobicity) / len(peptide_sequence) if peptide_sequence else 0.0

# Define hydrophobic and hydrophilic amino acids
hydrophobic_aa = {'A', 'C', 'F', 'I', 'L', 'M', 'V'}
hydrophilic_aa = {'D', 'E', 'G', 'H', 'K', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'Y'}

def calculate_amphipathicity(sequence):
    hydro_count = sum(1 for aa in sequence if aa in hydrophobic_aa)
    hydrophilic_count = sum(1 for aa in sequence if aa in hydrophilic_aa)
    total_count = hydro_count + hydrophilic_count
    if total_count == 0:
        return 0
    amphipathicity_score = abs(hydro_count - hydrophilic_count) / total_count
    return amphipathicity_score

def read_sequences_and_calculate_amphipathicity(input_file, output_csv):
    results = []
    
    for record in SeqIO.parse(input_file, "fasta"):
        sequence = record.seq
        normalized_hydrophobicity = calculate_hydrophobicity(str(sequence))
        
        protein = PA(str(sequence))
        net_charge = round(protein.charge_at_pH(7.4))
        isoelectric_point = protein.isoelectric_point()
        amphipathicity = calculate_amphipathicity(str(sequence))

        results.append({
            "序列ID": record.id,
            "Sequence": str(sequence),
            "Normalized Hydrophobicity": normalized_hydrophobicity,
            "Net Charge": net_charge,
            "Isoelectric Point": isoelectric_point,
            "Amphipathicity": amphipathicity
        })

    # Save results to CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

# Usage example
input_file = '/data2/lhmData/AMP/dataprocess/AMPgh.fasta'  # Input FASTA file
output_csv = '/data2/lhmData/AMP/dataprocess/charge/AMPgh.csv'  # Output CSV file
read_sequences_and_calculate_amphipathicity(input_file, output_csv)