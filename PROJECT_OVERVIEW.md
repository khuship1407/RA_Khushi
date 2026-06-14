# IDS-LLM: Transforming Network Intrusion Detection Using Large Language Models

## What This Project Does

This project builds a **Network Intrusion Detection System (IDS)** that combines two technologies:

1. **Decision Trees** — fast, interpretable classifiers trained on network traffic features
2. **Large Language Models (LLMs)** — GPT models that read the decision tree's reasoning and make a final, explainable classification

The core idea: instead of using a black-box neural network, the system generates a *human-readable explanation* of why a packet was classified as an attack — making the IDS both accurate and transparent.

---

## Real-World Example: From Wireshark to Verdict

To understand what this system does in practice, imagine a network administrator monitoring live traffic with a tool like **Wireshark** or **Zeek**. Here is what the end-to-end flow looks like for a single suspicious connection.

### Step 1 — Capture a Network Flow

A packet capture tool observes a connection and computes flow-level statistics:

```
Source IP:       192.168.1.45
Destination IP:  10.0.0.1
Protocol:        TCP
Duration:        0.003 s
Packets sent:    312
SYN flags:       312        <-- every packet has a SYN flag
ACK flags:       0
Avg packet size: 58 bytes
Packet rate:     104,000 / sec
Std deviation:   0.0        <-- all packets are identical size
IAT (inter-arrival time): 0.000009 s
... (39 more features)
```

### Step 2 — Two Decision Trees Evaluate the Flow

Both pre-trained decision trees independently walk their if/else logic using these 46 feature values:

```
Tree 1 path taken:
  Std <= 9.8           (std is zero -- all packets same size)
    IAT <= 83033340    (inter-arrival time is extremely low)
      rst_count <= 0.5 (no RST flags)
        syn_count > 0.29 (high SYN ratio)
          IAT <= 0.01  (packets arriving almost simultaneously)
  --> Prediction: DDoS  (confidence: 98.7%)

Tree 2 path taken:
  Std <= 9.8
    IAT <= 83033340
      rst_count <= 0.5
        syn_count > 0.29
          Covariance <= 90.4
  --> Prediction: Spoofing  (confidence: 61.2%)
```

The trees **disagree** — Tree 1 says DDoS, Tree 2 says Spoofing. This is the trigger to send the case to the LLM.

### Step 3 — Paths Are Serialized to Natural Language

The raw threshold conditions are converted into human-readable descriptions before being sent to the LLM:

```
Tree 1 path (natural language):
  "Standard deviation of packet length is zero.
   The time difference between two consecutive packets is extremely low.
   Number of packets with an RST flag is zero.
   On a scale from Level 1 to Level 12, Number of packets with a SYN
   flag is level 9. The time difference between two consecutive packets
   is extremely low. The syn(Synchronize) flag exists in this traffic."
  Prediction: DDoS  |  Confidence: 98.7%

Tree 2 path (natural language):
  "Standard deviation of packet length is zero.
   The time difference between two consecutive packets is extremely low.
   Number of packets with an RST flag is zero.
   On a scale from Level 1 to Level 12, Number of packets with a SYN
   flag is level 9. Covariance is level 2."
  Prediction: Spoofing  |  Confidence: 61.2%
```

### Step 4 — LLM Makes the Final Call

The two natural-language paths are sent to GPT-3.5-turbo-16k with a structured prompt:

```
SYSTEM:
  Your task is to evaluate 2 decision paths from two different decision
  trees analyzing network traffic data. Begin your response with
  'Most likely type of traffic: <label>' choosing from
  [BenignTraffic, DDoS, Brute_Force, Spoofing, DoS, Recon,
   Web-Based, Mirai]. Then explain which tree's path is more plausible.

  [Definitions of all 8 categories...]

USER:
  Path from first_tree:
  Path_Description: Standard deviation of packet length is zero.
  The time difference between two consecutive packets is extremely low...
  Prediction result: DDoS
  Prediction confidence rate: 98.7

  Path from second_tree:
  Path_Description: Standard deviation of packet length is zero...
  Prediction result: Spoofing
  Prediction confidence rate: 61.2
```

### Step 5 — GPT Responds with a Verdict and Explanation

```
Most likely type of traffic: DDoS

The most plausible path is from the first tree.

Rationale:
The first tree's path strongly indicates a SYN Flood DDoS attack.
The combination of an extremely low inter-arrival time (packets
arriving near-simultaneously), zero standard deviation (all packets
are identical, typical of automated flood tools), and a very high
SYN flag ratio with zero ACK flags is a textbook signature of a
TCP SYN flood — where an attacker sends a massive volume of SYN
packets without completing the handshake to exhaust server resources.

The second tree's Spoofing prediction is less convincing at only
61.2% confidence, and Spoofing typically requires mismatched IP/MAC
patterns which are not prominent in this path description.

Final verdict: DDoS attack — recommend blocking source IP immediately.
```

### Complete Flow Diagram

```
  [Live Network]
       |
       | raw packets
       v
  [Wireshark / Zeek / Custom Sensor]
       |
       | computes 46 flow-level features per connection
       v
  [Feature Vector]
  { Rate: 104000, SYN_flags: 312, Std: 0.0, IAT: 0.000009, ... }
       |
       |-----------------------------+
       |                             |
       v                             v
  [Decision Tree 1]           [Decision Tree 2]
  random_state=17              random_state=2
  max_depth=9                  max_depth=9
       |                             |
       v                             v
  DDoS (98.7%)              Spoofing (61.2%)
       |                             |
       +-----------> DISAGREE <------+
                         |
                         v
              [Path Serializer]
              converts threshold conditions
              to plain English descriptions
                         |
                         v
              [GPT-3.5-turbo-16k]
              receives both paths + definitions
                         |
                         v
              Final Answer: DDoS
              + human-readable explanation
                         |
                         v
              [Alert to Administrator]
              "DDoS attack detected from 192.168.1.45
               Block source IP. Confidence: High."
```

---

## The Dataset — CICIoT2023

- Source: [University of New Brunswick](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- **34 attack subfolders**, one CSV per attack type
- Each row = one network flow (a sequence of packets between two endpoints)
- **46 features** per row (packet sizes, flags, protocols, timing stats, and derived statistics)

### The 8-Category Mapping

The 34 raw attack types are grouped into 8 classes for classification:

| Category | Raw Folder Names |
|---|---|
| `BenignTraffic` | Benign_Final |
| `DDoS` | DDoS-ICMP_Flood, DDoS-TCP_Flood, DDoS-UDP_Flood, DDoS-SYN_Flood, DDoS-HTTP_Flood, DDoS-PSHACK_FLOOD, DDoS-RSTFINFLOOD, DDoS-SlowLoris, DDoS-SynonymousIP_Flood, DDoS-ACK_Fragmentation, DDoS-ICMP_Fragmentation, DDoS-UDP_Fragmentation |
| `DoS` | DoS-HTTP_Flood, DoS-SYN_Flood, DoS-TCP_Flood, DoS-UDP_Flood |
| `Mirai` | Mirai-greeth_flood, Mirai-greip_flood, Mirai-udpplain |
| `Recon` | Recon-HostDiscovery, Recon-OSScan, Recon-PingSweep, Recon-PortScan, VulnerabilityScan |
| `Spoofing` | DNS_Spoofing, MITM-ArpSpoofing, Backdoor_Malware |
| `Brute_Force` | DictionaryBruteForce |
| `Web-Based` | BrowserHijacking, CommandInjection, SqlInjection, Uploading_Attack, XSS |

---

## The Research Pipeline (Batch Evaluation)

This is how the paper's experiments were run — evaluating 5,000 pre-labeled test samples to measure accuracy:

```
source/combined_test.csv  (5,000 rows x 46 features, 625 per class)
        |
        ├─── SCENARIO A: All Features Present
        |           |
        |    PART 1 - Decision Tree Path Extraction
        |    path_search_revised_1.py  -->  processed/1_5000.csv
        |    path_search_revised_2.py  -->  processed/2_5000.csv
        |           |
        |    PART 2 - Prompt Formation
        |    1.0-file_check.py             -->  differences_output.txt
        |    1.1-input_file_integration.py -->  integrated_data.json
        |    2.0-file_prep.py              -->  matching_predictions.json (584 disagreements)
        |    2.1-file_prep.py              -->  analysis_input_with_prob.jsonl
        |    2.2-file_prep.py              -->  analysis_input_without_prob.jsonl
        |           |
        |    PART 3 - LLM Inference
        |    parallel_request.py  -->  *_results.jsonl
        |           |
        |    PART 4 - Evaluation
        |    2.3-file_integration.py  (parse LLM responses)
        |    2.4-file_integration.py  (merge with ground truth)
        |    3.0-evaluation.py        -->  accuracy / F1 / confusion matrix
        |
        └─── SCENARIO B: Missing Features (1 feature randomly dropped)
                    |
             (same structure, but DTs navigate around missing values)
```

**Key design choice:** Only the ~11.7% of samples where the two trees *disagree* (584 of 5,000) are sent to the LLM. The remaining 88.3% where both trees agree are classified directly, keeping API costs low.

---

## How the Decision Tree + LLM Combination Works

### Step 1 — Two Independent Decision Trees

Two decision trees are trained with **different random seeds** on the same 50,000-sample training set. This intentional diversity means they sometimes reach different conclusions for ambiguous traffic:

```
Same flow  -->  Tree 1:  Brute_Force (55%)
           -->  Tree 2:  BenignTraffic (100%)
```

### Step 2 — Path Serialization

The raw decision path (a series of threshold comparisons) is converted to natural language:

```
Raw:        Std <= 9.8  ->  IAT <= 83033340  ->  Number <= 9.28  ->  DDoS
Serialized: "Standard deviation of packet length is low.
             The time difference between two consecutive packets is low.
             On a scale from Level 1 to Level 9, the total number
             of packets is level 4."
```

Feature values are **binned into levels** so the LLM can reason comparatively without needing to interpret raw numbers.

### Step 3 — LLM Resolution

- **Both trees agree** → use that prediction directly, no LLM call needed
- **Trees disagree** → send both serialized paths to GPT with the system prompt, get a final label + explanation

---

## Experimental Results (gpt-3.5-turbo-16k, 5,000 test samples)

### Scenario A — All Features Present

| Variant | Accuracy | F1 (macro) |
|---|---|---|
| Decision Tree 1 alone | 76.9% | — |
| Decision Tree 2 alone | 77.8% | — |
| DT + LLM without confidence scores | 76.7% | 76.8% |
| **DT + LLM with confidence scores** | **78.6%** | **78.6%** |

### Scenario B — 1 Feature Randomly Missing

| Variant | Accuracy | F1 (macro) |
|---|---|---|
| DT + LLM, 1 missing feature | 76.0% | 76.0% |

Only a **2.6% drop** with one missing feature — the LLM handles incomplete paths gracefully.

### Per-Class F1 (Scenario A, with confidence scores)

| Class | F1 | Notes |
|---|---|---|
| Mirai | **0.996** | Near-perfect — very distinctive IoT botnet pattern |
| DoS | **0.993** | Near-perfect — high-volume single-source flooding |
| DDoS | **0.992** | Near-perfect — multi-source flooding |
| BenignTraffic | 0.753 | Some confusion with Brute_Force |
| Spoofing | 0.743 | Leaks into BenignTraffic / Web-Based |
| Recon | 0.664 | Confused with Brute_Force |
| Brute_Force | 0.627 | Confused with Web-Based and BenignTraffic |
| Web-Based | 0.523 | Hardest class — shares features with Brute_Force |

**Key insight:** Volume-based attacks (DDoS, DoS, Mirai) are nearly perfect because their traffic signatures are highly distinctive. Behaviorally similar attacks (Web-Based, Brute_Force, Recon) are hardest because they share many features.

---

## Changes Made From the Original Repository

The following changes were made when reproducing and running the paper's experiments:

### Bug Fixes

| File | Change |
|---|---|
| `Code_Base/parallel_request.py` | Added `python-dotenv` support — API key now loads from `Code_Base/.env` instead of being hardcoded. Commented out `os.environ["OPENAI_API_KEY"] = "Your KEY"`. |
| `dt_compare_gpt-3.5-turbo-16k/dt_compare/2.3-file_integration.py` | Replaced fragile split-based LLM response parser with a two-stage regex parser. Handles bold markdown (`**DDoS**`), missing blank lines, and verbose GPT phrasing like "Based on the information, the most likely type is...". |
| `dt_compare_missing_features_miss1/2.1-file_integration.py` | Same regex parser fix applied to the Scenario B variant. |
| `.gitignore` | Added `.env` entry to prevent accidental API key commits. |

### New Files Added

| File | Purpose |
|---|---|
| `Code_Base/view_results.py` | Results viewer — prints accuracy summary table, per-class F1 breakdown, sample predictions browser, and worst-case errors. |
| `Code_Base/test_single_sample.py` | End-to-end single sample tester — retrains both DTs from scratch, picks a random row from the test set, runs it through both trees, serializes the paths, calls GPT, and prints the full verdict. Useful for demonstrating the live pipeline. |
| `PROJECT_OVERVIEW.md` | This file — full documentation of the project. |
| `requirements.txt` | Pinned dependency versions from the project venv. |

### Files Deleted

| File | Reason |
|---|---|
| `Code_Base/test_5000.csv` | Built from raw CSVs (39 features) — incompatible with pipeline scripts that expect 46 features. Superseded by pre-existing `combined_test.csv`. |
| `build_test_set.py` | Script that produced the above. No longer needed. |
| `CSV/` (entire folder) | Raw CICIoT2023 dataset — not referenced by any pipeline script. Can be re-downloaded from UNB if needed. |

### Pipeline Outputs Generated

These files did not exist in the original repo and were produced by running the full experiments:

| File | Description |
|---|---|
| `analysis_input_with_prob_results.jsonl` | GPT responses for 584 Scenario A (with confidence) queries |
| `analysis_input_without_prob_results.jsonl` | GPT responses for 584 Scenario A (without confidence) queries |
| `output/analysis_input_results.jsonl` | GPT responses for 1,152 Scenario B (1 missing feature) queries |
| `updated_matching_predictions_with_prob.json` | Scenario A results merged with ground truth + GPT label |
| `updated_matching_predictions_without_prob.json` | Scenario A without-prob results merged |
| `output/analysis_input_results_without_prob.json` | Scenario B results parsed and merged |
| `processed/1_5000.csv` | DT1 decision paths for all 5,000 test samples |

---

## How to Run the Full Pipeline

### Prerequisites

```
pip install -r requirements.txt
```

Set your OpenAI API key in `Code_Base/.env`:
```
OPENAI_API_KEY=sk-...
```

### Quick Single-Sample Test

To verify the full pipeline works end-to-end on one sample:

```powershell
cd D:\KP_IDS-LLM\Code_Base
python test_single_sample.py
```

### Scenario A — All Features

```powershell
# PART 1 — from Part1/ directory
python script\path_search_revised_1.py   # -> processed/1_5000.csv
python script\path_search_revised_2.py   # -> processed/2_5000.csv

# PART 2 — from dt_compare_gpt-3.5-turbo-16k/ directory
python dt_compare\1.0-file_check.py
python dt_compare\1.1-input_file_integration.py
python dt_compare\2.0-file_prep.py
python dt_compare\2.1-file_prep.py       # -> analysis_input_with_prob.jsonl

# PART 3 — from Code_Base/
python parallel_request.py "...\analysis_input_with_prob.jsonl" \
                           "...\analysis_input_with_prob_results.jsonl" \
                           --model gpt-3.5-turbo-16k

# PART 4 — from dt_compare_gpt-3.5-turbo-16k/ directory
python dt_compare\2.3-file_integration.py
python dt_compare\2.4-file_integration.py
python dt_compare\3.0-evaluation.py
```

### Scenario B — Missing Features

```powershell
# PART 1 — run notebooks from Part1/script/
jupyter nbconvert --to notebook --execute find_paths_1.ipynb
jupyter nbconvert --to notebook --execute find_paths_2.ipynb

# PART 2 — from Part2/ directory
python dt_compare_missing_features_miss1\0.0-file_integration.py
python dt_compare_missing_features_miss1\1.1-input_file_integration.py
python dt_compare_missing_features_miss1\2.0-file_prep.py

# PART 3 — from Code_Base/
python parallel_request.py "...\output\analysis_input.jsonl" \
                           "...\output\analysis_input_results.jsonl" \
                           --model gpt-3.5-turbo-16k

# PART 4 — from Part2/ directory
python dt_compare_missing_features_miss1\2.1-file_integration.py
python dt_compare_missing_features_miss1\2.2-file_integration.py
python dt_compare_missing_features_miss1\3.0-evaluation.py
```

> **Note on 1.2-rep.py and 1.3-rep.py:** Optional retry scripts for failed API requests (HTTP 429/500). Only needed if the batch run had failures.

---

## Cost Estimate (gpt-3.5-turbo-16k)

| Scenario | Queries | Avg tokens/query | Estimated cost |
|---|---|---|---|
| Scenario A — 584 queries | 584 | ~1,145 | ~$2.15 |
| Scenario B — 1 missing feature | 1,152 | ~1,145 | ~$3.30 |

---

## Repository Structure

```
D:\KP_IDS-LLM\
├── Code_Base\
│   ├── .env                         OpenAI API key (never commit this)
│   ├── parallel_request.py          Async LLM batch inference
│   ├── view_results.py              Results summary viewer
│   ├── test_single_sample.py        End-to-end single sample demo
│   ├── Unmissing_features\
│   │   └── 8_categories(decision tree + LLM)\
│   │       ├── Part1\
│   │       │   ├── script\          dt_17.py, dt_2.py, path_search_revised_*.py
│   │       │   ├── source\          combined_test.csv, second_50000.csv
│   │       │   └── processed\       1_5000.csv, 2_5000.csv
│   │       └── Part2\
│   │           └── dt_compare_gpt-3.5-turbo-16k\dt_compare\
│   │               ├── analysis_input_with_prob.jsonl
│   │               ├── analysis_input_with_prob_results.jsonl
│   │               ├── analysis_input_without_prob.jsonl
│   │               ├── analysis_input_without_prob_results.jsonl
│   │               └── output\      data_with_prob.json, data_without_prob.json
│   └── Missing_features\
│       ├── Part1\
│       │   ├── script\              find_paths_1.ipynb, find_paths_2.ipynb
│       │   ├── source\              combined_test.csv, full_dataset.csv
│       │   └── missing_feature\     dt_1_1_missing.json, dt_2_1_missing.json
│       └── Part2\
│           └── dt_compare_missing_features_miss1\
│               ├── input\           dt_1_1_missing.json, dt_2_1_missing.json
│               └── output\          analysis_input.jsonl, *_results.jsonl
├── PROJECT_OVERVIEW.md              This file
└── requirements.txt                 Pinned Python dependencies
```

---

## Authors

Dongming Wu, Zhiyuan Peng, Yuchen Liu (NCSU)

**Paper:** *Transforming Network Intrusion Detection Using Large Language Models*
IEEE Consumer Communications & Networking Conference (CCNC), 2025
