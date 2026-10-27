#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tsAMP - standalone ESM-1v feature extraction.

This single file replaces `scripts/extract.py` from the facebookresearch/esm
repository. It is self-contained: nothing from the ESM repo is needed, only the
pip package `fair-esm` (which provides the `esm` module) and PyTorch.

Installation
------------
    conda create -n tsAMP python=3.11
    conda activate tsAMP
    pip install torch fair-esm numpy

Model weights (~2.5 GB for esm1v_t33_650M_UR90S_1) are downloaded automatically
on first run into the torch hub cache (~/.cache/torch/hub/checkpoints) and
reused afterwards. You can change that location with the TORCH_HOME env var.

Usage
-----
# one FASTA file -> one directory of per-sequence .pt files (same layout as the
# original ESM extract.py, so any downstream tsAMP code keeps working)
python extract_features.py peptides.fasta out/peptides

# one FASTA file -> a single feature matrix (easiest for a classifier)
python extract_features.py peptides.fasta out/peptides --csv out/peptides.csv

# a whole directory of .fasta files, one output sub-directory per file
python extract_features.py /path/to/fasta_dir/ /path/to/output_dir/

# CPU only
python extract_features.py peptides.fasta out/ --nogpu

Defaults reproduce the features used in the tsAMP paper:
    model          esm1v_t33_650M_UR90S_1
    layer          33
    representation mean over residues  ->  1280-dim vector per peptide
"""

import argparse
import os
import pathlib
import sys
import time

import torch

try:
    import esm  # noqa: F401  (pip install fair-esm)
    from esm import pretrained
except ImportError:  # pragma: no cover
    sys.exit(
        "The `esm` module was not found.\n"
        "Install it with:  pip install fair-esm\n"
        "(the PyPI package is called fair-esm, the import name is esm)"
    )


# --------------------------------------------------------------------------- #
# FASTA reading / cleaning
# --------------------------------------------------------------------------- #
# Residues understood by the ESM alphabet. Anything else has to be handled.
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    """Yield (header, sequence) pairs. Robust to blank lines and CRLF."""
    header, chunks = None, []
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:].strip(), []
            else:
                if header is None:
                    raise ValueError(
                        f"{path}: sequence data before the first '>' header line"
                    )
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def clean_sequence(seq, mode):
    """Normalise a peptide sequence.

    mode='drop' : remove '*' and every non-standard residue (B, J, O, U, X, Z).
                  This is what the tsAMP paper used.
    mode='x'    : remove '*' only, map non-standard residues to 'X'
                  (ESM has a dedicated token for X; keeps the length intact).
    """
    seq = seq.upper().replace("*", "").replace("-", "").replace(" ", "")
    if mode == "x":
        return "".join(c if c in STANDARD_AA else "X" for c in seq)
    return "".join(c for c in seq if c in STANDARD_AA)


def load_sequences(path, clean_mode):
    labels, seqs, skipped = [], [], []
    seen = set()
    for header, seq in read_fasta(path):
        label = header.split()[0] if header.split() else header
        cleaned = clean_sequence(seq, clean_mode)
        if not cleaned:
            skipped.append(label)
            continue
        if label in seen:
            # duplicate IDs would overwrite each other's .pt file
            suffix = 2
            while f"{label}_{suffix}" in seen:
                suffix += 1
            label = f"{label}_{suffix}"
        seen.add(label)
        labels.append(label)
        seqs.append(cleaned)
    return labels, seqs, skipped


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def make_batches(seqs, toks_per_batch):
    """Group sequence indices into batches of at most `toks_per_batch` tokens.

    Sequences are sorted by length so each batch is padded as little as
    possible. +2 accounts for the BOS and EOS tokens.
    """
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    batches, current, longest = [], [], 0
    for i in order:
        new_longest = max(longest, len(seqs[i]) + 2)
        if current and new_longest * (len(current) + 1) > toks_per_batch:
            batches.append(current)
            current, longest = [i], len(seqs[i]) + 2
        else:
            current.append(i)
            longest = new_longest
    if current:
        batches.append(current)
    return batches


# --------------------------------------------------------------------------- #
# Core extraction
# --------------------------------------------------------------------------- #
def extract_one_file(fasta_path, out_dir, model, alphabet, args, device):
    labels, seqs, skipped = load_sequences(fasta_path, args.clean)
    if not labels:
        print(f"  [skip] {fasta_path}: no usable sequences")
        return None
    if skipped:
        print(f"  [warn] {len(skipped)} empty sequence(s) skipped, e.g. {skipped[:3]}")

    out_dir = pathlib.Path(out_dir)
    if args.save_pt:
        out_dir.mkdir(parents=True, exist_ok=True)

    batch_converter = alphabet.get_batch_converter(args.truncation_seq_length)
    batches = make_batches(seqs, args.toks_per_batch)
    return_contacts = "contacts" in args.include

    collected = {}  # label -> 1-D mean tensor, used for the --csv / --npy output
    n_done = 0

    with torch.no_grad():
        for b_i, idxs in enumerate(batches, 1):
            data = [(labels[i], seqs[i]) for i in idxs]
            batch_labels, batch_strs, toks = batch_converter(data)
            toks = toks.to(device=device, non_blocking=True)

            out = model(
                toks,
                repr_layers=args.repr_layers,
                return_contacts=return_contacts,
            )
            reps = {l: t.to("cpu") for l, t in out["representations"].items()}
            contacts = out["contacts"].to("cpu") if return_contacts else None

            for j, label in enumerate(batch_labels):
                truncate_len = min(args.truncation_seq_length, len(batch_strs[j]))
                result = {"label": label}

                if "per_tok" in args.include:
                    result["representations"] = {
                        l: t[j, 1 : truncate_len + 1].clone() for l, t in reps.items()
                    }
                if "mean" in args.include:
                    means = {
                        l: t[j, 1 : truncate_len + 1].mean(0).clone()
                        for l, t in reps.items()
                    }
                    result["mean_representations"] = means
                    collected[label] = means[args.repr_layers[-1]]
                if "bos" in args.include:
                    result["bos_representations"] = {
                        l: t[j, 0].clone() for l, t in reps.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[j, :truncate_len, :truncate_len].clone()

                if args.save_pt:
                    torch.save(result, out_dir / f"{label}.pt")
                n_done += 1

            print(
                f"  batch {b_i}/{len(batches)}  "
                f"({len(idxs)} seqs, {n_done}/{len(labels)} done)",
                flush=True,
            )

    return labels, collected


def write_matrix(labels, collected, csv_path=None, npy_path=None):
    """Write all mean embeddings as one table: id + 1280 feature columns."""
    labels = [l for l in labels if l in collected]
    if not labels:
        return
    mat = torch.stack([collected[l] for l in labels]).numpy()

    if csv_path:
        csv_path = pathlib.Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = "id," + ",".join(f"f{i}" for i in range(mat.shape[1]))
        with open(csv_path, "w") as fh:
            fh.write(header + "\n")
            for label, row in zip(labels, mat):
                fh.write(label + "," + ",".join(f"{v:.6f}" for v in row) + "\n")
        print(f"  wrote {csv_path}  ({mat.shape[0]} x {mat.shape[1]})")

    if npy_path:
        import numpy as np

        npy_path = pathlib.Path(npy_path)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, mat)
        with open(npy_path.with_suffix(".ids.txt"), "w") as fh:
            fh.write("\n".join(labels) + "\n")
        print(f"  wrote {npy_path}  ({mat.shape[0]} x {mat.shape[1]})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Extract ESM-1v representations for peptides in FASTA "
        "format (standalone; no ESM repository checkout required).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input",
        type=pathlib.Path,
        help="a FASTA file, or a directory containing .fasta/.fa/.faa files",
    )
    p.add_argument(
        "output",
        type=pathlib.Path,
        help="output directory for the per-sequence .pt files",
    )
    p.add_argument(
        "--model",
        default="esm1v_t33_650M_UR90S_1",
        help="pretrained model name or path to a local .pt checkpoint. "
        "Note the trailing _1: the five ESM-1v models are named "
        "esm1v_t33_650M_UR90S_1 ... _5",
    )
    p.add_argument(
        "--repr_layers",
        type=int,
        nargs="+",
        default=[33],
        help="layer indices to extract (33 = last layer of the 650M model)",
    )
    p.add_argument(
        "--include",
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        default=["mean"],
        help="which representations to keep",
    )
    p.add_argument("--toks_per_batch", type=int, default=4096, help="max tokens per batch")
    p.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than this",
    )
    p.add_argument(
        "--clean",
        choices=["drop", "x"],
        default="drop",
        help="'drop' removes '*' and non-standard residues (B/J/O/U/X/Z) as in "
        "the tsAMP paper; 'x' maps them to the X token instead",
    )
    p.add_argument(
        "--csv",
        type=pathlib.Path,
        default=None,
        help="also write every mean embedding into one CSV (id + 1280 columns)",
    )
    p.add_argument(
        "--npy", type=pathlib.Path, default=None, help="also write one .npy matrix"
    )
    p.add_argument(
        "--no-pt",
        dest="save_pt",
        action="store_false",
        help="do not write the per-sequence .pt files (use with --csv/--npy)",
    )
    p.add_argument("--overwrite", action="store_true", help="redo files already present")
    p.add_argument("--nogpu", action="store_true", help="force CPU even if CUDA is available")
    return p


def main():
    args = build_parser().parse_args()
    t0 = time.time()

    if args.input.is_dir():
        fasta_files = sorted(
            f
            for f in args.input.iterdir()
            if f.suffix.lower() in {".fasta", ".fa", ".faa", ".fas"}
        )
        if not fasta_files:
            sys.exit(f"No FASTA files found in {args.input}")
    else:
        fasta_files = [args.input]

    print(f"Loading {args.model} ...", flush=True)
    model, alphabet = pretrained.load_model_and_alphabet(args.model)
    model.eval()

    device = "cpu"
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (this is slow for the 650M model)")

    n_layers = model.num_layers
    if not all(-(n_layers + 1) <= i <= n_layers for i in args.repr_layers):
        sys.exit(f"--repr_layers out of range for a {n_layers}-layer model")
    args.repr_layers = [(i + n_layers + 1) % (n_layers + 1) for i in args.repr_layers]

    all_labels, all_means = [], {}

    for fasta in fasta_files:
        out_dir = (
            args.output / fasta.stem if len(fasta_files) > 1 else args.output
        )
        if args.save_pt and out_dir.exists() and not args.overwrite and any(out_dir.glob("*.pt")):
            print(f"[skip] {fasta.name}: {out_dir} already contains .pt files "
                  f"(use --overwrite to redo)")
            continue

        print(f"[run ] {fasta}")
        res = extract_one_file(fasta, out_dir, model, alphabet, args, device)
        if res:
            labels, means = res
            all_labels.extend(labels)
            all_means.update(means)

    if (args.csv or args.npy) and all_means:
        write_matrix(all_labels, all_means, args.csv, args.npy)

    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
