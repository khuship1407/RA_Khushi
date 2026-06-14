"""
test_single_sample.py
End-to-end single-sample test: train DTs -> pick random row -> get paths
-> build prompt -> call GPT -> print verdict.
"""
import os
import sys
import random
import textwrap
import json

import asyncio
import warnings
import aiohttp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
TRAIN_CSV    = (
    r"C:\Users\khush\sem_2\RA-2026\para_mi_amor-main\Code_Base\Unmissing_features"
    r"\8_categories(decision tree + LLM)"
    r"\Part1(Decision trees training + Path Extraction+Serialization)"
    r"\source\second_50000.csv"
)
TEST_CSV     = (
    r"C:\Users\khush\sem_2\RA-2026\para_mi_amor-main\Code_Base\Unmissing_features"
    r"\8_categories(decision tree + LLM)"
    r"\Part1(Decision trees training + Path Extraction+Serialization)"
    r"\source\combined_test.csv"
)
ENV_PATH     = r"C:\Users\khush\sem_2\RA-2026\para_mi_amor-main\Code_Base\.env"
MODEL        = "gpt-3.5-turbo-16k"
LABELS       = ("BenignTraffic", "Brute_Force", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-Based")
SEP          = "=" * 60

SYSTEM_PROMPT = (
    "Your task is to evaluate 2 decision paths from two different decision trees analyzing network traffic data, "
    "each supported by a decision tree path and a stated accuracy rate. "
    "Begin your response by stating the most likely type of traffic start by 'Most likely type of traffic: ' "
    "and choose exactly from ['BenignTraffic', 'DDoS', 'Brute_Force', 'Spoofing', 'DoS', 'Recon', 'Web-Based', 'Mirai']. "
    "Then, specify 'The most plausible path is from the first tree' or 'The most plausible path is from the second tree', "
    "depending on which path from which tree you believe provides the most accurate explanation based on the path descriptions. "
    "Provide a rationale for your choice, considering the descriptions and any other relevant data provided in the paths.\n"
    "Below are the brief definitions for each category.\n"
    "BenignTraffic: Network traffic that is legitimate and poses no threat.\n"
    "DDoS: Multiple compromised systems flood a target with traffic.\n"
    "Brute_Force: Systematically tries all possible password combinations.\n"
    "Spoofing: Masquerades as another by falsifying data.\n"
    "DoS: Makes a service unavailable by overwhelming it.\n"
    "Recon: Gathers information to identify vulnerabilities.\n"
    "Web-Based: Exploits web application vulnerabilities (SQLi, XSS, etc.).\n"
    "Mirai: Malware that infects IoT devices to form a DDoS botnet."
)

# ── Feature categorisation (mirrors path_search_revised_1.py) ─────────────────
def categorize_std(v):
    if v == 0: return 'zero'
    if v <= 0.38: return 'low'
    if v <= 9.8:  return 'medium'
    return 'high'

def categorize_iat(v):
    if v == 0:              return 'zero'
    if v <= 0.02:           return 'extremely low'
    if v <= 41462904.01:    return 'low'
    if v <= 78690612.0:     return 'midly low'
    if v <= 83469472.0:     return 'medium'
    if v <= 133481500.0:    return 'midly high'
    if v <= 167246344.0:    return 'high'
    return 'extremely high'

def categorize_variance(v):
    if v == 0:    return 'zero'
    if v <= 0.39: return 'extremely low'
    if v <= 0.64: return 'low'
    if v <= 0.83: return 'midly low'
    if v <= 0.95: return 'midly high'
    if v <= 0.97: return 'high'
    return 'extremely high'

def categorize_fin_count(v):
    if v == 0:    return 'zero'
    if v <= 0.05: return 'extremely low'
    if v <= 0.08: return 'low'
    if v <= 0.32: return 'midly low'
    if v <= 0.35: return 'midly high'
    if v <= 0.47: return 'high'
    return 'extremely high'

def categorize_ack_count(v):
    if v == 0:    return 'zero'
    if v <= 0.02: return 'low'
    if v <= 0.13: return 'medium'
    return 'high'

def _level(v, thresholds):
    if v == 0: return 'zero'
    for i, t in enumerate(thresholds, 1):
        if v <= t: return f'level {i}'
    return f'level {len(thresholds)+1}'

def categorize_rst_count(v):
    return _level(v, [0.5,1.32,4.3,5.44,39.4,89.1,115.55,178.75,240.9,284.45,287.0,732.1,1266.6,6170.85])

def categorize_magnitude(v):
    return _level(v, [10.39,10.63,10.68,11.04,11.12,11.14,53.6])

def categorize_tot_size(v):
    return _level(v, [56.38,66.45,85.75,98.2,292.18,350.8,478.82,902.01,2347.6])

def categorize_number(v):
    return _level(v, [6.0,7.01,7.5,9.28,9.5,9.56,11.5,12.75])

def categorize_avg(v):
    return _level(v, [53.88,55.42,56.22,59.13,60.58,66.04,98.91,203.92,380.76,1382.54])

def categorize_radius(v):
    return _level(v, [5.41,5.87,8.31,48.07,56.66,69.8,81.55,278.51,753.63])

def categorize_tot_sum(v):
    return _level(v, [303.1,354.75,435.1,442.75,557.26,700.96,1398.8,1616.65,3884.2,37934.4])

def categorize_syn_count(v):
    return _level(v, [0.05,0.1,0.15,0.25,0.29,0.45,0.65,0.7,1.55,1.7,1.75])

def categorize_flow_duration(v):
    return _level(v, [0.15,0.2,0.39,0.59,3.07,3.52,17.93,20.54,38.82,72.36,516.47])

def categorize_weight(v):
    return _level(v, [77.83,79.94,83.68,90.03,126.97,140.4,193.08])

def categorize_duration(v):
    return _level(v, [54.0,57.0,61.5,62.75,84.4,100.6,116.55,160.4,246.5])

def categorize_srate(v):
    return _level(v, [1.34,6.95,22.49,29.92,34.16,73.62,107.41,695.91])

def categorize_covariance(v):
    return _level(v, [1.12,90.4,97.33,4959.5,6269.43,18379.3,19281.41,81486.11,86125.98,915674.0,1051078.34,2317899.75])

def categorize_min(v):
    return _level(v, [52.0,60.5,67.7,67.95,290.12,786.0,1504.0])

def categorize_max(v):
    return _level(v, [54.2,64.31,74.7,80.8,90.5,140.5,364.0,466.88,873.25,3076.5,3093.8])

def categorize_header_length(v):
    return _level(v, [7715.2,23286.3,27577.7,82721.44,150021.5,156572.95,180764.0,220049.1,221395.0,427410.41,1000736.09])

def categorize_rate(v):
    return _level(v, [3.37,7.23,25.51,28.62,74.44,106.78,114.73,292.85,372.16,1169.08])

def categorize_urg_count(v):
    return _level(v, [12.1,14.65,36.65,64.35,86.1,188.85,340.9])

CATEGORIZERS = {
    'Std':           categorize_std,
    'IAT':           categorize_iat,
    'rst_count':     categorize_rst_count,
    'Magnitue':      categorize_magnitude,
    'Tot_size':      categorize_tot_size,
    'Number':        categorize_number,
    'AVG':           categorize_avg,
    'Radius':        categorize_radius,
    'Tot_sum':       categorize_tot_sum,
    'syn_count':     categorize_syn_count,
    'flow_duration': categorize_flow_duration,
    'Weight':        categorize_weight,
    'Duration':      categorize_duration,
    'Srate':         categorize_srate,
    'Covariance':    categorize_covariance,
    'Min':           categorize_min,
    'Max':           categorize_max,
    'Header_Length': categorize_header_length,
    'Rate':          categorize_rate,
    'urg_count':     categorize_urg_count,
    'fin_count':     categorize_fin_count,
    'ack_count':     categorize_ack_count,
    'Variance':      categorize_variance,
}

FEAT_DESC_ZERO = {
    'flow_duration': "The Duration of the packet's flow",
    'Header_Length': "Packet header length",
    'Duration': "Time-to-Live",
    'Rate': "Rate of packet transmission",
    'Srate': "Rate of outbound packets transmission",
    'fin_flag_number': "Count of FIN flags",
    'syn_flag_number': "Count of SYN flags",
    'rst_flag_number': "Count of RST flags",
    'psh_flag_number': "Count of PSH flags",
    'ack_flag_number': "Count of ACK flags",
    'ece_flag_number': "Count of ECE flags",
    'cwr_flag_number': "Count of CWR flags",
    'ack_count': "Number of packets with an ACK flag",
    'syn_count': "Number of packets with a SYN flag",
    'fin_count': "Number of packets with a FIN flag",
    'urg_count': "Number of packets with an URG flag",
    'rst_count': "Number of packets with an RST flag",
    'Tot_sum': "Total sum of packets lengths",
    'Min': "Minimum packet length in the flow",
    'Max': "Maximum packet length in the flow",
    'AVG': "Average packet length in the flow",
    'Std': "Standard deviation of packet length",
    'Tot_size': "Packet's length",
    'IAT': "The time difference between two consecutive packets",
    'Number': "The total number of packets",
    'Magnitue': "Magnitude",
    'Radius': "Radius",
    'Covariance': "Covariance",
    'Variance': "Variance",
    'Weight': "Weight",
}

FEAT_DESC = {
    'flow_duration': "On a scale from Level 1 to Level 12, the Duration of the packet's flow",
    'Header_Length': "On a scale from Level 1 to Level 12, Packet header length",
    'Duration': "On a scale from Level 1 to Level 10, Time-to-Live",
    'Rate': "On a scale from Level 1 to Level 11, Rate of packet transmission",
    'Srate': "On a scale from Level 1 to Level 9, Rate of outbound packets transmission",
    'ack_count': "Number of packets with an ACK flag",
    'syn_count': "On a scale from Level 1 to Level 12, Number of packets with a SYN flag",
    'fin_count': "Number of packets with a FIN flag",
    'urg_count': "On a scale from Level 1 to Level 8, Number of packets with an URG flag",
    'rst_count': "On a scale from Level 1 to Level 15, number of packets with an RST flag",
    'Tot_sum': "On a scale from Level 1 to Level 11, Total sum of packets lengths",
    'Min': "On a scale from Level 1 to Level 8, Minimum packet length in the flow",
    'Max': "On a scale from Level 1 to Level 12, Maximum packet length in the flow",
    'AVG': "On a scale from Level 1 to Level 11, average packet length in the flow",
    'Std': "Standard deviation of packet length",
    'Tot_size': "On a scale from Level 1 to Level 10, Packet's length",
    'IAT': "The time difference between two consecutive packets",
    'Number': "On a scale from Level 1 to Level 9, the total number of packets",
    'Magnitue': "On a scale from Level 1 to Level 8, Magnitude",
    'Radius': "On a scale from Level 1 to Level 10, Radius",
    'Covariance': "On a scale from Level 1 to Level 13, Covariance",
    'Variance': "Variance",
    'Weight': "On a scale from Level 1 to Level 8, Weight",
    'HTTP': "Application layer protocol is HTTP",
    'HTTPS': "Application layer protocol is HTTPS",
    'DNS': "Application layer protocol is DNS",
    'Telnet': "Application layer protocol is Telnet",
    'SMTP': "Application layer protocol is SMTP",
    'SSH': "Application layer protocol is SSH",
    'IRC': "Application layer protocol is IRC",
    'TCP': "Transport layer protocol is TCP",
    'UDP': "Transport layer protocol is UDP",
    'DHCP': "Application layer protocol is DHCP",
    'ARP': "Link layer protocol is ARP",
    'ICMP': "Network layer protocol is ICMP",
    'IPv': "Network layer protocol is IP",
    'LLC': "Link layer protocol is LLC",
    'fin_flag_number': "Count of FIN flags",
    'syn_flag_number': "Count of SYN flags",
    'rst_flag_number': "Count of RST flags",
    'psh_flag_number': "Count of PSH flags",
    'ack_flag_number': "Count of ACK flags",
    'ece_flag_number': "Count of ECE flags",
    'cwr_flag_number': "Count of CWR flags",
}

NUMERICAL_FEATURES = [
    'flow_duration','Header_Length','Duration','Rate','Srate','Drate',
    'ack_count','syn_count','fin_count','urg_count','rst_count',
    'Tot_sum','Min','Max','AVG','Std','Tot_size','IAT','Number',
    'Magnitue','Radius','Covariance','Variance','Weight',
    'fin_flag_number','syn_flag_number','rst_flag_number',
    'psh_flag_number','ack_flag_number','ece_flag_number','cwr_flag_number',
]
FLAG_FEATURES = ['HTTP','HTTPS','DNS','Telnet','SMTP','SSH','IRC','TCP','UDP','DHCP','ARP','ICMP','IPv','LLC']


def create_description(row_dict):
    parts = []
    for feat in NUMERICAL_FEATURES:
        val = row_dict.get(feat)
        if val is not None and pd.notnull(val):
            desc = FEAT_DESC_ZERO.get(feat, feat) if val == 'zero' else FEAT_DESC.get(feat, feat)
            parts.append(f"{desc} is {val}.")
    for feat in FLAG_FEATURES:
        if row_dict.get(feat) == 1:
            parts.append(f"{FEAT_DESC.get(feat, feat)}.")
    if row_dict.get('psh_flag_number') == 1:
        parts.append("The psh(push) flag exists in this traffic")
    if row_dict.get('rst_flag_number') == 1 and row_dict.get('rst_count') != 'zero':
        parts.append("The rst(reset) flag exists in this traffic")
    if row_dict.get('ack_flag_number') == 1 and row_dict.get('ack_count') != 'zero':
        parts.append("The ack(Acknowledgment) flag exists in this traffic")
    if row_dict.get('syn_flag_number') == 1 and row_dict.get('syn_count') != 'zero':
        parts.append("The syn(Synchronize) flag exists in this traffic")
    return ' '.join(parts)


# ── DT path extraction ────────────────────────────────────────────────────────
def get_path_description_and_confidence(clf, sample_array, feature_names):
    """
    Use sklearn's decision_path to reconstruct the feature dict visited,
    categorise the values, then build the natural-language description.
    Returns (description_str, predicted_label, confidence_pct).
    """
    tree = clf.tree_
    node_indicator = clf.decision_path(sample_array.reshape(1, -1))
    node_ids = node_indicator.indices

    feat_dict = {}
    for node_id in node_ids[:-1]:  # skip leaf
        feat_idx  = tree.feature[node_id]
        feat_name = feature_names[feat_idx]
        feat_val  = sample_array[feat_idx]
        feat_dict[feat_name] = float(feat_val)

    # categorise numeric features that appear in the path
    for feat, categorizer in CATEGORIZERS.items():
        if feat in feat_dict:
            feat_dict[feat] = categorizer(feat_dict[feat])

    # Remove Protocol_Type (not used in descriptions)
    feat_dict.pop('Protocol_Type', None)

    # For flag features in path: keep raw 0/1 value
    leaf_id      = node_ids[-1]
    counts       = tree.value[leaf_id][0]
    class_probs  = counts / counts.sum()
    pred_idx     = int(class_probs.argmax())
    confidence   = round(float(class_probs[pred_idx]) * 100, 2)
    pred_label   = LABELS[pred_idx]

    description = create_description(feat_dict)
    return description, pred_label, confidence


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    load_dotenv(ENV_PATH)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not found in .env")

    # ── 1. Load + rename training data ────────────────────────────────────────
    print("Loading training data and fitting decision trees...")
    train_df = pd.read_csv(TRAIN_CSV)
    train_df.rename(columns={'Tot sum': 'Tot_sum', 'Tot size': 'Tot_size',
                              'Protocol Type': 'Protocol_Type'}, inplace=True)

    X = train_df.drop(columns=['label'])
    y = train_df['label']
    feature_names = list(X.columns)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=4000, random_state=2)

    clf1 = DecisionTreeClassifier(max_depth=9, max_features='sqrt', random_state=17)
    clf1.fit(X_train, y_train)

    clf2 = DecisionTreeClassifier(max_depth=9, max_features='sqrt', random_state=2)
    clf2.fit(X_train, y_train)
    print("  Trees trained.")

    # ── 2. Load test CSV and pick one random row ───────────────────────────────
    test_df = pd.read_csv(TEST_CSV)
    test_df.rename(columns={'Tot sum': 'Tot_sum', 'Tot size': 'Tot_size',
                             'Protocol Type': 'Protocol_Type'}, inplace=True)

    random.seed(RANDOM_SEED)
    row_idx = random.randint(0, len(test_df) - 1)
    row = test_df.iloc[row_idx]
    actual_label = row['label']
    sample = row.drop('label').values.astype(float)

    # ── 3. Run both trees ─────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path1, pred1, conf1 = get_path_description_and_confidence(clf1, sample, feature_names)
        path2, pred2, conf2 = get_path_description_and_confidence(clf2, sample, feature_names)

    # ── 4. Build prompt ───────────────────────────────────────────────────────
    user_content = (
        f"Please evaluate the following two different paths, each generated by a different decision tree, "
        f"and choose the single path that makes the most sense based on the descriptions and predictions."
        f"Provide reasons for your choice:\n"
        f"Path from first_tree:\n"
        f"Path_Description: {path1}.\n"
        f"Prediction result for above path description:{pred1}\n"
        f"Prediction confidence rate for above path description:{conf1}\n"
        f"Path from second_tree:\n"
        f"Path_Description: {path2}.\n"
        f"Prediction result for above path description:{pred2}\n"
        f"Prediction confidence rate for above path description:{conf2}"
    )

    # ── 5. Call GPT via aiohttp (avoids openai/httpx version conflict) ────────
    print(f"  Sending to {MODEL}...\n")

    async def call_gpt():
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()

    gpt_response = asyncio.run(call_gpt())

    # Parse GPT answer
    import re
    m = re.search(r'Most likely type of traffic:\s*\*{0,2}\s*([\w-]+)', gpt_response)
    gpt_label = m.group(1).strip() if m else None
    if gpt_label not in LABELS:
        fb = re.search(r'\b(BenignTraffic|DDoS|Brute_Force|Spoofing|DoS|Recon|Web-Based|Mirai)\b', gpt_response)
        gpt_label = fb.group(1) if fb else "UNKNOWN"

    # Everything after the first line is the explanation
    first_nl = gpt_response.find('\n')
    explanation = gpt_response[first_nl:].strip() if first_nl != -1 else gpt_response

    # ── 6. Print result ───────────────────────────────────────────────────────
    correct = (gpt_label == actual_label)
    tick    = "[CORRECT]" if correct else "[WRONG]"

    # Top-10 feature values for display
    top_features = [
        ('Rate',            row.get('Rate',       'N/A')),
        ('TCP',             row.get('TCP',         'N/A')),
        ('syn_flag_number', row.get('syn_flag_number','N/A')),
        ('syn_count',       row.get('syn_count',   'N/A')),
        ('rst_count',       row.get('rst_count',   'N/A')),
        ('Std',             row.get('Std',         'N/A')),
        ('IAT',             row.get('IAT',         'N/A')),
        ('Variance',        row.get('Variance',    'N/A')),
        ('Header_Length',   row.get('Header_Length','N/A')),
        ('Number',          row.get('Number',      'N/A')),
    ]

    def wrap60(text):
        lines = text.strip().splitlines()
        out = []
        for line in lines:
            if len(line) <= 56:
                out.append(line)
            else:
                out.extend(textwrap.wrap(line, 56))
        return ("\n    " + " " * 14).join(out[:8])

    print(SEP)
    print("  SINGLE SAMPLE TEST")
    print(SEP)
    print(f"  Row index in combined_test.csv: {row_idx}")
    print()
    print("  Sample features (top 10):")
    for name, val in top_features:
        if isinstance(val, float):
            print(f"    {name:<20} {val:.4f}")
        else:
            print(f"    {name:<20} {val}")
    print()
    print(f"  Tree 1 says:  {pred1}  (confidence: {conf1:.1f}%)")
    print(f"  Tree 2 says:  {pred2}  (confidence: {conf2:.1f}%)")
    print()

    if pred1 == pred2:
        print("  Both trees agreed -- LLM arbitration not normally needed.")
        print("  (Running LLM anyway for demonstration.)")
    else:
        print("  Trees disagreed -- LLM arbitration needed.")
    print()

    print(f"  GPT Final Answer: {gpt_label} {tick}")
    print()
    print("  GPT Explanation:")
    print(f"    {wrap60(explanation)}")
    print()
    print(f"  Actual Label:  {actual_label}")
    verdict = "CORRECT" if correct else "WRONG"
    print(f"  Result: {verdict} {tick}")
    print(SEP)


if __name__ == "__main__":
    main()
