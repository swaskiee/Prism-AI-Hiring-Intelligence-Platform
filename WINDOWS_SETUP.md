# Running Prism on Windows

If you hit `ModuleNotFoundError: No module named 'numpy'` even after `pip install` says
"Requirement already satisfied," it's almost always because Windows has **more than one
Python installed**, and `pip` and `python`/`python3` are pointing at different ones.

## Check which Pythons you have

```cmd
py -0
```

This lists every Python version Windows knows about (e.g. `-3.13`, `-3.14`). Pick one
and use it consistently for every command below — never mix `pip install` under one
version with `python3 rank.py` under another.

## Setup (replace 3.13 with whichever version you choose)

```cmd
cd Prism-AI-Hiring-Intelligence-Platform
py -3.13 -m pip install pandas numpy scikit-learn scipy
```

No virtual environment is required to get this running. If you do want one (optional,
keeps dependencies isolated):

```cmd
py -3.13 -m venv venv
venv\Scripts\activate.bat
pip install pandas numpy scikit-learn scipy
```

(`source venv/bin/activate` is a Mac/Linux command and will not work on Windows —
use `venv\Scripts\activate.bat` instead, or `venv\Scripts\Activate.ps1` in PowerShell.)

## Running the pipeline

```cmd
py -3.13 rank.py --candidates C:\path\to\your\candidates.jsonl --out hyperion_submission.csv
```

Replace `C:\path\to\your\candidates.jsonl` with wherever you actually saved the real
dataset file from the hackathon's Google Drive link — `path/to/candidates.jsonl` in
the README is a placeholder, not a real path.

## Validating your submission

`validate_submission.py` is included directly in this repo root (no separate copy
needed):

```cmd
py -3.13 validate_submission.py hyperion_submission.csv
```

Expected output: `Submission is valid.`

## Quick troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `numpy`/`pandas`/`sklearn` not found, but pip says installed | Two different Pythons in use | Use the same `py -X.XX` for both install and run |
| `'source' is not recognized` | Mac/Linux command on Windows | Use `venv\Scripts\activate.bat` instead, or skip the venv entirely |
| `validate_submission.py` not found | Script wasn't copied into the project folder | Already fixed — it now ships in this repo's root |
| `rank.py` runs but errors on the candidates path | Using the literal placeholder path from the README | Point `--candidates` at your real, downloaded `candidates.jsonl` |
