# Setup Guide — IDS-LLM

Step-by-step instructions to get this project running from scratch after cloning the repository.

---

## Prerequisites

- **Python 3.11** — [download here](https://www.python.org/downloads/release/python-3117/)
- **Git**
- An **OpenAI API key** with access to `gpt-3.5-turbo-16k`

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/Dongming1010/IDS-LLM.git
cd IDS-LLM
```

---

## Step 2 — Create a Virtual Environment

Create and activate a Python virtual environment inside the project folder:

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

> You should see `(venv)` at the start of your terminal prompt once activated.

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all pinned packages: scikit-learn, pandas, numpy, matplotlib, aiohttp, tiktoken, typer, python-dotenv, jupyter, and nbconvert.

---

## Step 4 — Set Up Your OpenAI API Key

Create a file called `.env` inside the `Code_Base/` folder:

```
Code_Base/.env
```

Add this single line to it (replace with your actual key):

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Important:** This file is listed in `.gitignore` and will never be committed. Do not paste your key anywhere else in the code.

---

## Step 5 — Run the Full Pipeline

There are two experiment scenarios. Run them from the directories specified below — the scripts use relative paths and will fail if run from the wrong location.

---

### Scenario A — All Features Present

This reproduces the main experiment: 5,000 test samples, both trees, LLM arbitration for disagreements.

#### Part 1 — Extract Decision Tree Paths

Navigate to the Part 1 directory:

```powershell
cd "Code_Base\Unmissing_features\8_categories(decision tree + LLM)\Part1(Decision trees training + Path Extraction+Serialization)"
```

Run both path extraction scripts:

```powershell
python script\path_search_revised_1.py
python script\path_search_revised_2.py
```

**Output:** `processed\1_5000.csv` and `processed\2_5000.csv`
Each file contains the serialized decision path and prediction for all 5,000 test rows.

---

#### Part 2 — Build LLM Prompts

Navigate to the Part 2 directory:

```powershell
cd "..\Part2(Prompt formation + GPT_Response + Evaluation)\dt_compare_gpt-3.5-turbo-16k"
```

Run the prompt-building scripts in order:

```powershell
python dt_compare\1.0-file_check.py
python dt_compare\1.1-input_file_integration.py
python dt_compare\2.0-file_prep.py
python dt_compare\2.1-file_prep.py
python dt_compare\2.2-file_prep.py
```

**Output:**
- `dt_compare\analysis_input_with_prob.jsonl` — 584 prompts including confidence scores
- `dt_compare\analysis_input_without_prob.jsonl` — 584 prompts without confidence scores

> Only the ~584 samples where Tree 1 and Tree 2 **disagree** are sent to the LLM.
> The remaining ~4,416 where both trees agree are classified directly.

---

#### Part 3 — Run LLM Inference

Navigate back to the `Code_Base` directory:

```powershell
cd "..\..\..\..\..\..\Code_Base"
```

Define paths as variables for convenience:

```powershell
$PART2 = "Unmissing_features\8_categories(decision tree + LLM)\Part2(Prompt formation + GPT_Response + Evaluation)\dt_compare_gpt-3.5-turbo-16k\dt_compare"
```

Run inference for the **with confidence scores** variant (~$2.15):

```powershell
python parallel_request.py `
  "$PART2\analysis_input_with_prob.jsonl" `
  "$PART2\analysis_input_with_prob_results.jsonl" `
  --model gpt-3.5-turbo-16k
```

Run inference for the **without confidence scores** variant (~$2.15):

```powershell
python parallel_request.py `
  "$PART2\analysis_input_without_prob.jsonl" `
  "$PART2\analysis_input_without_prob_results.jsonl" `
  --model gpt-3.5-turbo-16k
```

Both commands print progress as batches complete. When done you will see `Task completed`.

---

#### Part 4 — Evaluate Results

Navigate back to the Part 2 directory:

```powershell
cd "$PART2"
```

Run the evaluation scripts. The default evaluates the **without confidence** variant:

```powershell
python dt_compare\2.3-file_integration.py
python dt_compare\2.4-file_integration.py
python dt_compare\3.0-evaluation.py
```

To evaluate the **with confidence** variant, open `2.3-file_integration.py`, `2.4-file_integration.py`, and `3.0-evaluation.py` and change every occurrence of `without_prob` to `with_prob` in the file path strings, then re-run.

**Expected results:**

| Variant | Accuracy | F1 (macro) |
|---|---|---|
| With confidence scores | ~78.6% | ~78.6% |
| Without confidence scores | ~76.7% | ~76.8% |

---

### Scenario B — Missing Features (1 Feature Randomly Dropped)

This tests robustness: one random feature is removed before the decision trees run.

#### Part 1 — Extract Paths with Missing Feature

Navigate to the Missing Features Part 1 directory:

```powershell
cd "Code_Base\Missing_features\Part1(Decision trees training + Path Extraction+Serialization)\script"
```

Run both notebooks:

```powershell
jupyter nbconvert --to notebook --execute find_paths_1.ipynb --output find_paths_1.ipynb
jupyter nbconvert --to notebook --execute find_paths_2.ipynb --output find_paths_2.ipynb
```

**Output:** `missing_feature\1_missing\dt_1_1_missing.json` and `dt_2_1_missing.json`

> To change the number of missing features, edit `selected_features = random.sample(all_features, 1)` in both notebooks and change `1` to the desired count before running.

---

#### Part 2 — Build LLM Prompts

Navigate to the Missing Features Part 2 directory:

```powershell
cd "..\..\Part2(Prompt formation + GPT_Response + Evaluation)"
```

Run the prompt-building scripts in order:

```powershell
python dt_compare_missing_features_miss1\0.0-file_integration.py
python dt_compare_missing_features_miss1\1.1-input_file_integration.py
python dt_compare_missing_features_miss1\2.0-file_prep.py
```

**Output:** `dt_compare_missing_features_miss1\output\analysis_input.jsonl` (~1,152 prompts)

> Note: `1.2-rep.py` and `1.3-rep.py` are listed in the original README but are optional retry
> scripts for failed API calls. Skip them on a fresh run — they require results to already exist.

---

#### Part 3 — Run LLM Inference

Navigate back to `Code_Base`:

```powershell
cd "..\..\..\Code_Base"
```

```powershell
$MISS1 = "Missing_features\Part2(Prompt formation + GPT_Response + Evaluation)\dt_compare_missing_features_miss1"

python parallel_request.py `
  "$MISS1\output\analysis_input.jsonl" `
  "$MISS1\output\analysis_input_results.jsonl" `
  --model gpt-3.5-turbo-16k
```

Estimated cost: ~$3.30

---

#### Part 4 — Evaluate Results

```powershell
cd "$MISS1"

python 2.1-file_integration.py
python 2.2-file_integration.py
python 3.0-evaluation.py
```

**Expected results:** Accuracy ~76.0%, F1 ~76.0% (only ~2.6% drop from Scenario A)

---

## Step 6 — Verify the Setup

Run the single-sample end-to-end test to confirm everything works before running the full pipeline:

```powershell
cd Code_Base
python test_single_sample.py
```

Expected output:

```
Loading training data and fitting decision trees...
  Trees trained.
  Sending to gpt-3.5-turbo-16k...

============================================================
  SINGLE SAMPLE TEST
============================================================
  Row index in combined_test.csv: 912

  Sample features (top 10):
    Rate                 2.7150
    ...

  Tree 1 says:  Brute_Force  (confidence: 100.0%)
  Tree 2 says:  Brute_Force  (confidence: 100.0%)

  GPT Final Answer: Brute_Force [CORRECT]
  ...
  Result: CORRECT [CORRECT]
============================================================
```

If this runs without errors, your environment, API key, and data files are all working correctly.

---

## Step 7 — View a Summary of All Results

From the `Code_Base` directory:

```powershell
python view_results.py
```

This prints a comparison table of all scenarios, per-class F1 breakdown, sample predictions, and worst-case errors — all from the results files already on disk.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: source/combined_test.csv` | You are running a script from the wrong directory. Check the directory instructions for each part above. |
| `OPENAI_API_KEY not found` | Make sure `Code_Base/.env` exists and contains `OPENAI_API_KEY=sk-...` |
| `JSONDecodeError` on results file | The results JSONL was created with a BOM. Always create JSONL files using Python, never PowerShell `Set-Content`. |
| `UnicodeEncodeError` running view_results.py | Run from PowerShell (not cmd.exe). If it still fails, your terminal encoding is not UTF-8. |
| LLM inference produces no output | Check your API key is valid and has quota. Run `test_single_sample.py` first to isolate the issue. |
| `1.2-rep.py` crashes immediately | This is expected on a fresh run — it requires a results file to exist. Skip it. |

---

## Cost Summary

| Step | Approx. Cost |
|---|---|
| Scenario A — with confidence scores (584 queries) | ~$2.15 |
| Scenario A — without confidence scores (584 queries) | ~$2.15 |
| Scenario B — 1 missing feature (1,152 queries) | ~$3.30 |
| **Total (all three runs)** | **~$7.60** |

All estimates are for `gpt-3.5-turbo-16k` at standard pricing.

---

## Project Structure (Quick Reference)

```
IDS-LLM/
├── Code_Base/
│   ├── .env                          <-- YOU CREATE THIS (API key)
│   ├── parallel_request.py           LLM batch inference
│   ├── test_single_sample.py         End-to-end single sample demo
│   ├── view_results.py               Results summary viewer
│   ├── Unmissing_features/
│   │   └── 8_categories(decision tree + LLM)/
│   │       ├── Part1/                DT training + path extraction
│   │       │   ├── script/           path_search_revised_1.py, _2.py
│   │       │   ├── source/           combined_test.csv (test set)
│   │       │   │                     second_50000.csv  (training set)
│   │       │   └── processed/        1_5000.csv, 2_5000.csv (generated)
│   │       └── Part2/
│   │           └── dt_compare_gpt-3.5-turbo-16k/dt_compare/
│   │               ├── analysis_input_with_prob.jsonl      (generated)
│   │               ├── analysis_input_with_prob_results.jsonl (generated)
│   │               ├── analysis_input_without_prob.jsonl   (generated)
│   │               └── analysis_input_without_prob_results.jsonl (generated)
│   └── Missing_features/
│       ├── Part1/script/             find_paths_1.ipynb, find_paths_2.ipynb
│       └── Part2/dt_compare_missing_features_miss1/
│           └── output/               analysis_input.jsonl (generated)
│                                     analysis_input_results.jsonl (generated)
├── SETUP.md                          This file
├── PROJECT_OVERVIEW.md               Full project explanation
├── README.md                         Original paper README
├── requirements.txt                  Python dependencies
└── flowchart.png                     Pipeline diagram from paper
```

Files marked `(generated)` do not exist in the fresh repo and are created by running the pipeline.
