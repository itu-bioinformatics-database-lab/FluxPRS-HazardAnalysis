#!/usr/bin/env python3
"""Convert DiCAD long-format SNP-array CSV to PLINK1 SNP-major .bed/.bim/.fam.

Kaan's prs_pipeline.py can then ingest the .bed with --geno-glob '*.bed'.

The Synapse dictionary calls Position "Plate position". Illumina GenomeStudio
Final Report "Position" is genomic, so this script inspects the values and
uses them as bp if they look genomic; otherwise BIM pos=0 and matching falls
back to rsIDs in SNP Name (Kaan positions those from PGS Catalog coords).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, OrderedDict

ACGT = set("ACGT")
MISSING_ALLELES = {"", "-", "0", ".", "n", "N", "NA", "NaN", "I", "D", "NC", "--"}
WELL_RE = re.compile(r"^[A-Ha-h]\d{1,2}$")
RS_RE = re.compile(r"^rs\d+", re.I)


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def pick_col(fields, *aliases):
    idx = {norm_key(f): f for f in fields if f}
    for alias in aliases:
        if norm_key(alias) in idx:
            return idx[norm_key(alias)]
    return None


def open_text(path):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return open(path, "r", encoding="utf-8", errors="replace", newline="")


def sniff_header(path, max_skip=40):
    with open_text(path) as fh:
        skipped = []
        for i, line in enumerate(fh):
            if i >= max_skip:
                break
            raw = line.rstrip("\r\n")
            if not raw or raw.startswith("[") or raw.startswith("#"):
                skipped.append(raw[:200])
                continue
            cols = next(csv.reader([raw]))
            keys = [norm_key(c) for c in cols]
            if any(k in keys for k in ("patientid", "snpname", "iddicad", "dicadid")):
                return i, cols, skipped
            skipped.append(raw[:200])
    raise SystemExit("could not find Patient ID / SNP Name header in %s" % path)


def norm_chr(raw) -> str:
    s = str(raw or "").strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    s = s.upper()
    if s in {"X", "Y", "XY", "MT", "M"}:
        return "23" if s == "X" else "24" if s == "Y" else "25" if s == "XY" else "26"
    if re.fullmatch(r"\d+", s):
        n = int(s)
        if 1 <= n <= 26:
            return str(n)
    return "0"


def parse_pos(raw) -> int:
    s = str(raw or "").strip().replace(",", "")
    if WELL_RE.match(s):
        return 0
    if re.fullmatch(r"\d+", s):
        return int(s)
    return 0


def gt_code(a, b, a1, a2) -> int:
    """PLINK SNP-major 2-bit: 0=hom A1, 1=missing, 2=het, 3=hom A2."""
    if a in MISSING_ALLELES or b in MISSING_ALLELES:
        return 1
    a, b = a.upper(), b.upper()
    if a not in ACGT or b not in ACGT:
        return 1
    copies_a1 = (a == a1) + (b == a1)
    copies_a2 = (a == a2) + (b == a2)
    if copies_a1 + copies_a2 != 2:
        return 1
    if copies_a1 == 2:
        return 0
    if copies_a1 == 1:
        return 2
    return 3


def pack_bed_row(codes, n_samples: int) -> bytes:
    nbytes = (n_samples + 3) // 4
    out = bytearray(nbytes)
    for i, g in enumerate(codes):
        out[i // 4] |= (g & 3) << (2 * (i % 4))
    return bytes(out)


def choose_alleles(counter: Counter):
    items = [(al, n) for al, n in counter.items() if al in ACGT]
    items.sort(key=lambda x: (-x[1], x[0]))
    if not items:
        return None, None
    a1 = items[0][0]
    a2 = items[1][0] if len(items) > 1 else "0"
    return a1, a2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="DiCAD_GWAS_JUNE2020.csv")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--min-gc", type=float, default=0.0)
    ap.add_argument("--autosomes-only", action="store_true", default=True)
    ap.add_argument("--keep-allosomes", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0, help="debug: stop after N data rows")
    args = ap.parse_args()
    autosomes_only = not args.keep_allosomes
    out_prefix = args.out_prefix
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)

    header_line, cols, skipped = sniff_header(args.input)
    c_pid = pick_col(cols, "id_dicad", "Patient ID", "Sample ID", "SampleID", "dicad_id")
    c_snp = pick_col(cols, "SNP Name", "SNP_name", "SNP")
    c_chr = pick_col(cols, "Chr", "Chromosome")
    c_pos = pick_col(cols, "position", "Position", "MapInfo", "Position(hg19)", "Position(GRCh37)")
    c_a1 = pick_col(cols, "Allele1_forward", "Allele 1 Forward", "Allele1 - Forward", "Allele1-Forward", "Allele1")
    c_a2 = pick_col(cols, "Allele2_forward", "Allele 2 Forward", "Allele2 - Forward", "Allele2-Forward", "Allele2")
    c_gc = pick_col(cols, "GC_score", "GC Score", "GCScore", "GenCall")
    missing = [name for name, col in [("Patient ID", c_pid), ("SNP Name", c_snp),
                                      ("Allele 1", c_a1), ("Allele 2", c_a2)] if col is None]
    if missing:
        raise SystemExit("missing columns %s; have %s" % (missing, cols))

    print("header at line", header_line, "columns:", cols, flush=True)
    print("mapped:", dict(patient=c_pid, snp=c_snp, chr=c_chr, pos=c_pos,
                          a1=c_a1, a2=c_a2, gc=c_gc), flush=True)

    samples = OrderedDict()
    snps = OrderedDict()
    pos_examples = []
    n_rows = 0
    n_low_gc = 0
    n_missing_gt = 0

    print("PASS 1: samples, SNPs, alleles", flush=True)
    with open_text(args.input) as fh:
        for _ in range(header_line):
            next(fh)
        rdr = csv.DictReader(fh, fieldnames=cols)
        next(rdr, None)
        for rec in rdr:
            n_rows += 1
            if args.max_rows and n_rows > args.max_rows:
                break
            pid = (rec.get(c_pid) or "").strip()
            snp = (rec.get(c_snp) or "").strip()
            if not pid or not snp:
                continue
            if pid not in samples:
                samples[pid] = len(samples)
            a = (rec.get(c_a1) or "").strip().upper()
            b = (rec.get(c_a2) or "").strip().upper()
            gc_raw = (rec.get(c_gc) or "").strip() if c_gc else ""
            try:
                gc = float(gc_raw) if gc_raw not in ("", "NA") else 1.0
            except ValueError:
                gc = 1.0
            if gc < args.min_gc:
                n_low_gc += 1
                continue
            chrom = norm_chr(rec.get(c_chr) if c_chr else "")
            pos = parse_pos(rec.get(c_pos) if c_pos else "")
            if len(pos_examples) < 8 and (rec.get(c_pos) or ""):
                pos_examples.append(rec.get(c_pos))
            if snp not in snps:
                snps[snp] = {"chr": chrom, "pos": pos, "alleles": Counter(), "n": 0}
            meta = snps[snp]
            if meta["chr"] == "0" and chrom != "0":
                meta["chr"] = chrom
            if meta["pos"] == 0 and pos > 0:
                meta["pos"] = pos
            if a not in MISSING_ALLELES and a in ACGT:
                meta["alleles"][a] += 1
            if b not in MISSING_ALLELES and b in ACGT:
                meta["alleles"][b] += 1
            meta["n"] += 1
            if n_rows % 5_000_000 == 0:
                print("  pass1 rows=%d samples=%d snps=%d" % (n_rows, len(samples), len(snps)),
                      flush=True)

    n_samples = len(samples)
    keep_snps = []
    n_non_auto = 0
    n_no_allele = 0
    genomic_pos = 0
    for snp, meta in snps.items():
        chrom_i = int(meta["chr"])
        if autosomes_only and not (1 <= chrom_i <= 22):
            n_non_auto += 1
            continue
        a1, a2 = choose_alleles(meta["alleles"])
        if a1 is None:
            n_no_allele += 1
            continue
        meta["a1"], meta["a2"] = a1, a2
        if meta["pos"] >= 1000:
            genomic_pos += 1
        keep_snps.append(snp)
    n_snps = len(keep_snps)
    snp_index = {s: i for i, s in enumerate(keep_snps)}
    print("PASS 1 done: rows=%d samples=%d snps_kept=%d genomic_pos=%d examples_pos=%s" % (
        n_rows, n_samples, n_snps, genomic_pos, pos_examples), flush=True)

    codes = bytearray(n_snps * n_samples)
    for i in range(n_snps * n_samples):
        codes[i] = 1

    print("PASS 2: genotypes", flush=True)
    n_rows2 = 0
    with open_text(args.input) as fh:
        for _ in range(header_line):
            next(fh)
        rdr = csv.DictReader(fh, fieldnames=cols)
        next(rdr, None)
        for rec in rdr:
            n_rows2 += 1
            if args.max_rows and n_rows2 > args.max_rows:
                break
            pid = (rec.get(c_pid) or "").strip()
            snp = (rec.get(c_snp) or "").strip()
            si = snp_index.get(snp)
            pi = samples.get(pid)
            if si is None or pi is None:
                continue
            gc_raw = (rec.get(c_gc) or "").strip() if c_gc else ""
            try:
                gc = float(gc_raw) if gc_raw not in ("", "NA") else 1.0
            except ValueError:
                gc = 1.0
            if gc < args.min_gc:
                continue
            a = (rec.get(c_a1) or "").strip().upper()
            b = (rec.get(c_a2) or "").strip().upper()
            meta = snps[snp]
            g = gt_code(a, b, meta["a1"], meta["a2"])
            if g == 1:
                n_missing_gt += 1
            codes[si * n_samples + pi] = g
            if n_rows2 % 5_000_000 == 0:
                print("  pass2 rows=%d" % n_rows2, flush=True)

    fam_path = out_prefix + ".fam"
    bim_path = out_prefix + ".bim"
    bed_path = out_prefix + ".bed"
    with open(fam_path, "w", encoding="utf-8") as fh:
        for pid in samples:
            fh.write("%s %s 0 0 0 -9\n" % (pid, pid))
    with open(bim_path, "w", encoding="utf-8") as fh:
        for snp in keep_snps:
            m = snps[snp]
            fh.write("%s %s 0 %d %s %s\n" % (m["chr"], snp, m["pos"], m["a1"], m["a2"]))
    with open(bed_path, "wb") as fh:
        fh.write(bytes([0x6C, 0x1B, 0x01]))
        for si in range(n_snps):
            start = si * n_samples
            row = codes[start:start + n_samples]
            fh.write(pack_bed_row(row, n_samples))

    n_rs = sum(1 for s in keep_snps if RS_RE.match(s))
    summary = {
        "input": os.path.abspath(args.input),
        "out_prefix": os.path.abspath(out_prefix),
        "n_rows": n_rows,
        "n_samples": n_samples,
        "n_snps_kept": n_snps,
        "n_snps_rsid": n_rs,
        "n_snps_genomic_pos_ge_1000": genomic_pos,
        "n_snps_non_autosomal_dropped": n_non_auto,
        "n_snps_no_acgt_allele": n_no_allele,
        "n_low_gc_dropped": n_low_gc,
        "n_missing_gt_codes": n_missing_gt,
        "position_examples": pos_examples,
        "sample_ids_head": list(samples)[:8],
        "autosomes_only": autosomes_only,
        "min_gc": args.min_gc,
        "fam": fam_path,
        "bim": bim_path,
        "bed": bed_path,
        "bed_bytes": os.path.getsize(bed_path),
    }
    sum_path = out_prefix + ".conversion_summary.json"
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", fam_path, bim_path, bed_path, sum_path, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
