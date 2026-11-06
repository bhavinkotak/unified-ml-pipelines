"""Build the Zenodo deposit: metadata.json (legacy deposit schema), PDF copy, and the code/data zip."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "zenodo"
SLUG = "unified-ml-pipelines"
AFFILIATION = "IBM, Armonk, NY"
GITHUB = "https://github.com/bhavinkotak/unified-ml-pipelines"  # returned HTTP 200 when checked
CHANGES = [
    "Removed every organisational outcome figure of version 1 (deployment time, incident counts, case-study "
    "tables). No releasable data supported them.",
    "Added a preregistered experiment on three public OpenML tables (adult, bank-marketing, credit-g) with five "
    "seeds. It compares a single feature definition with a dual implementation that has five documented divergence "
    "sources, and includes a per-source ablation and breakpoint scenarios (store timeouts, heavier missingness, "
    "covariate shift).",
    "Added a rerun study of content-hash lineage records, and a simulated CI release gate.",
    "Wrote out formal definitions of the system model, the skew and disagreement measures, the lineage record, and "
    "the fail-closed gate, and stated how each decision point behaves under missing data, unknown results, and "
    "timeouts.",
    "Replaced blog posts and references that could not be verified with sources checked against a DOI or arXiv "
    "record. Added a Threats to Validity section.",
]
ZIP_INCLUDE = ["preregistration.json", "REVIEW.md", "code", "tests", "results", "data/SOURCES.md", "data/raw",
               "paper/main.tex", "paper/references.bib", "paper/IEEEtran.bst", "paper/figures", "paper/main.pdf"]
ZIP_EXCLUDE = re.compile(r"(__pycache__|\.pyc$|\.venv|\.DS_Store|\.aux$|\.log$|\.blg$|\.bbl$|\.out$)")


def env(tex: str, name: str) -> str:
    return re.search(rf"\\begin\{{{name}\}}(.*?)\\end\{{{name}\}}", tex, re.DOTALL).group(1).strip()


def detex(s: str) -> str:
    s = s.replace("\\%", "%").replace("~", " ").replace("--", "-")
    s = re.sub(r"\\(?:emph|textbf)\{([^{}]*)\}", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def bib_entries(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", text, re.DOTALL):
        f = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", m.group(3) + "\n")}
        f["_key"] = m.group(2)
        out.append(f)
    return out


def clean(s: str) -> str:
    s = re.sub(r"\\[`'\"^~=.uvHc]\{?\\?([A-Za-z])\}?", r"\1", s)
    s = s.replace("{\\o}", "o").replace("\\o", "o")
    return re.sub(r"[{}]", "", s)


def fmt_ref(e: dict) -> str:
    authors = clean(e.get("author", "")).replace(" and others", " et al.").replace(" and ", ", ")
    venue = e.get("journal") or e.get("booktitle") or ("" if e.get("eprint") else e.get("howpublished", ""))
    parts = [f"{authors} ({e.get('year', 'n.d.')}). {clean(e['title'])}." + (f" {clean(venue)}." if venue else "")]
    if e.get("doi"):
        parts.append(f"https://doi.org/{e['doi']}")
    elif e.get("eprint"):
        parts.append(f"arXiv:{e['eprint']} https://arxiv.org/abs/{e['eprint']}")
    elif e.get("url"):
        parts.append(e["url"])
    return " ".join(parts)


def cited_keys(tex: str) -> list[str]:
    seen: dict[str, None] = {}
    for g in re.findall(r"\\cite\{([^{}]+)\}", tex):
        for k in g.split(","):
            seen.setdefault(k.strip(), None)
    return list(seen)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tex = (PAPER / "main.tex").read_text()
    title = re.search(r"\\title\{(.*?)\}\s*\n", tex, re.DOTALL).group(1)
    abstract = detex(env(tex, "abstract"))
    keywords = [k.strip() for k in env(tex, "IEEEkeywords").split(",") if k.strip()]
    entries = {e["_key"]: e for e in bib_entries((PAPER / "references.bib").read_text())}
    refs = [fmt_ref(entries[k]) for k in cited_keys(tex) if k in entries]
    desc = (f"<p>{abstract}</p><p><strong>Changes in version 2.0</strong></p><ul>"
            + "".join(f"<li>{c}</li>" for c in CHANGES) + "</ul>"
            "<p>The deposit contains the paper, the experiment code, cached public data with sources and checksums, "
            "all result files, tests, and the preregistration. Reproduce with <em>python code/run_all.py</em>.</p>")
    meta = {
        "title": title, "upload_type": "publication", "publication_type": "preprint", "description": desc,
        "creators": [{"name": "Kotak, Bhavin", "affiliation": AFFILIATION, "orcid": "0009-0006-4793-7046"}],
        "keywords": keywords, "license": "cc-by-4.0", "access_right": "open", "version": "2.0", "language": "eng",
        "references": refs,
        "notes": "Reproduction: create a Python 3.12 environment with numpy, scipy, pandas, scikit-learn, matplotlib, "
                 "threadpoolctl, and pytest; then run 'python code/run_all.py' (downloads or reuses data/raw, "
                 "writes results/, regenerates paper/figures and the tables in paper/main.tex) and "
                 "'python -m pytest tests'. Build the PDF in paper/ with pdflatex and bibtex.",
        "related_identifiers": [{"relation": "isSupplementedBy", "identifier": GITHUB, "resource_type": "software",
                                 "scheme": "url"}],
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    shutil.copyfile(PAPER / "main.pdf", OUT / f"{SLUG}-v2.pdf")
    zpath = OUT / f"{SLUG}-v2-code-data.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for item in ZIP_INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
            for q in files:
                rel = q.relative_to(ROOT).as_posix()
                if not ZIP_EXCLUDE.search(rel):
                    z.write(q, f"{SLUG}-v2/{rel}")
    print(f"wrote {OUT / 'metadata.json'}, {zpath.name} ({zpath.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
