import filetype
import os
import time
from Database import Database
from ORF import ORF
from Blast import Blast
from Diamond import Diamond
from settings import *
from Bio import SeqIO
from Base import*


import os
import sys
from Bio import SeqIO

class AMP(AMPBase):
    def __init__(self, input_type='contig', input_sequence=None, threads=32, output_file=None, data='na', aligner='blast'):
        o_f_path, o_f_name = os.path.split(os.path.abspath(output_file))

        self.input_type = input_type.lower()
        self.input_sequence = os.path.abspath(input_sequence)
        self.threads = threads
        self.output_file = os.path.abspath(output_file)
        self.data = data
        self.aligner = aligner.lower()

        self.db = path
        self.dp = data_path

        self.working_directory = o_f_path

        super(AMPBase, self).__init__()

    def validate_inputs(self):
        if not os.path.exists(self.input_sequence):
            print(f"ERROR: Input file does not exist: {self.input_sequence}", file=sys.stderr)
            sys.exit(1)

        if self.output_file == self.input_sequence and self.clean:
            print("ERROR: Output path same as input. Must specify a different path when cleaning.", file=sys.stderr)
            sys.exit(1)

        file_kind = filetype.guess(self.input_sequence)
        print(f"File type detected: {file_kind}")

        if file_kind is None:
            if not self.is_fasta():
                print("ERROR: Invalid FASTA format.", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"ERROR: Unsupported file format (extension: {file_kind.extension}, MIME: {file_kind.mime})", file=sys.stderr)
            sys.exit(1)

        if self.threads > os.cpu_count():
            print(f"ERROR: Invalid thread count (max {os.cpu_count()}), given: {self.threads}", file=sys.stderr)
            sys.exit(1)

    def is_fasta(self):
        """Checks for valid FASTA format."""
        with open(self.input_sequence, "r") as handle:
            fasta = SeqIO.parse(handle, "fasta")
            for record in fasta:
                if not record.id or not record.seq:
                    return False
                if self.input_type == "contig":
                    return self.is_dna(record.seq)
                if self.input_type == "protein":
                    return self.is_protein(record.seq)
            return True

    @staticmethod
    def is_dna(sequence):
        nucleotide_dict = {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0, 'U': 0,
                          'W': 0, 'S': 0, 'M': 0, 'K': 0, 'R': 0, 'Y': 0,
                          'B': 0, 'D': 0, 'H': 0, 'V': 0}
        for base in sequence:
            try:
                nucleotide_dict[base.upper()] += 1
            except KeyError as e:
                print(f"ERROR: Invalid nucleotide in FASTA: {e}", file=sys.stderr)
                return False
        print(f"Valid nucleotide FASTA: {nucleotide_dict}")
        return True

    @staticmethod
    def is_protein(sequence):
        amino_acids_dict = {
            'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0, 'U': 0,
            'R': 0, 'D': 0, 'Q': 0, 'E': 0, 'H': 0, 'I': 0,
            'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0,
            'W': 0, 'Y': 0, 'V': 0, 'X': 0, 'Z': 0, 'J': 0, 'B': 0
        }
        count = 0
        for amino_acid in sequence:
            try:
                amino_acids_dict[amino_acid.upper()] += 1
            except KeyError as e:
                print(f"ERROR: Invalid amino acid in FASTA: {e}", file=sys.stderr)
                return False

        for a in amino_acids_dict:
            if a not in 'ATGCNU':
                count += amino_acids_dict[a]

        if count == 0:
            print(f"ERROR: Invalid protein FASTA (no valid amino acids): {amino_acids_dict}", file=sys.stderr)
            return False

        print(f"Valid protein FASTA: {amino_acids_dict}")
        return True

    def run(self):
        self.validate_inputs()
        self.run_blast()

    def run_blast(self):
        if self.input_type == "protein":
            self.process_protein()
        elif self.input_type == "contig":
            self.process_contig()
        else:
            sys.exit(1)

    def process_protein(self):
        file_name = os.path.basename(self.input_sequence)
        output = self.output_file
        if self.aligner == "diamond":
            diamond_obj = Diamond(self.input_sequence, output_file=output, num_threads=self.threads)
            diamond_obj.run()
        else:
            blast_obj = Blast(input_file=file_name, output_file=output, num_threads=self.threads)
            blast_obj.run()

    def process_contig(self):
        file_name = os.path.basename(self.input_sequence)
        output = self.output_file
        orf_obj = ORF(input_file=self.input_sequence)
        orf_obj.contig_to_orf()
        contig_fsa_file = os.path.join(self.working_directory, f"{file_name}.temp.contig.fsa")
        try:
            if os.stat(contig_fsa_file).st_size > 0:
                if self.aligner == "diamond":
                    diamond_obj = Diamond(input_file=contig_fsa_file, output_file=output, num_threads=self.threads)
                    diamond_obj.run()
                else:
                    blast_obj = Blast(input_file=contig_fsa_file, output_file=output, num_threads=self.threads)
                    blast_obj.run()
            else:
                self.write_stub_output_file()
        except Exception as e:
            print(f"ERROR: Failed to write ORF file: {e}", file=sys.stderr)

