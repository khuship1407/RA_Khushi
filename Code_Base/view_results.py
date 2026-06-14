import json
import os
import textwrap
from sklearn.metrics import classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_UNMISS = (
    r"C:\Users\khush\sem_2\RA-2026\para_mi_amor-main\Code_Base\Unmissing_features"
    
    r"\8_categories(decision tree + LLM)"
    r"\Part2(Prompt formation + GPT_Response + Evaluation)"
    r"\dt_compare_gpt-3.5-turbo-16k\dt_compare"
)
WITH_PROB_JSON    = os.path.join(BASE_UNMISS, "updated_matching_predictions_with_prob.json")
WITHOUT_PROB_JSON = os.path.join(BASE_UNMISS, "updated_matching_predictions_without_prob.json")
MISS1_JSON = (
    r"C:\Users\khush\sem_2\RA-2026\para_mi_amor-main\Code_Base\Missing_features"
    r"\Part2(Prompt formation + GPT_Response + Evaluation)"
    r"\dt_compare_missing_features_miss1\output"
    r"\analysis_input_results_without_prob.json"
)

CLASSES = ["BenignTraffic", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Brute_Force", "Web-Based"]
SEP  = "-" * 60
SEP2 = "=" * 60


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wrap_text(text, width=54, indent="               "):
    lines = text.strip().splitlines()
    result = []
    for line in lines:
        if len(line) <= width:
            result.append(line)
        else:
            result.extend(textwrap.wrap(line, width))
    return ("\n" + indent).join(result[:6])


# ── Section 1: Overall Summary Table ──────────────────────────────────────────
def print_summary():
    print("\n" + SEP)
    print("  OVERALL ACCURACY SUMMARY")
    print(SEP)
    rows = [
        ("Decision Tree 1 alone",                  76.9),
        ("Decision Tree 2 alone",                  77.8),
        ("DT + LLM  (without confidence scores)",  76.7),
        ("DT + LLM  (with confidence scores)",     78.6),
        ("DT + LLM  (1 missing feature)",          76.0),
    ]
    best = max(r[1] for r in rows)
    for label, acc in rows:
        marker = " << best" if acc == best else ""
        bar_len = int((acc - 74) * 5)
        bar = "#" * bar_len
        print(f"  {label:<42}  {acc:5.1f}%  {bar}{marker}")
    print(SEP)


# ── Section 2: Per-Class F1 Breakdown ─────────────────────────────────────────
def print_per_class(data):
    y_true = [d["Label"]      for d in data]
    y_pred = [d["gpt_choice"] for d in data]
    report = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )

    print("\n" + SEP)
    print("  PER-CLASS F1  (DT + LLM with confidence scores, n=584)")
    print(SEP)
    print(f"  {'Class':<16} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Support':>8}")
    print(f"  {'-'*16} {'-'*9} {'-'*7} {'-'*7} {'-'*8}")
    for cls in CLASSES:
        r   = report.get(cls, {})
        p   = r.get("precision", 0)
        rc  = r.get("recall",    0)
        f1  = r.get("f1-score",  0)
        sup = int(r.get("support", 0))
        bar = "#" * int(f1 * 10)
        print(f"  {cls:<16} {p:>9.3f} {rc:>7.3f} {f1:>7.3f} {sup:>8}  {bar}")
    macro = report["macro avg"]
    print(f"  {'-'*16} {'-'*9} {'-'*7} {'-'*7} {'-'*8}")
    print(
        f"  {'macro avg':<16} {macro['precision']:>9.3f}"
        f" {macro['recall']:>7.3f} {macro['f1-score']:>7.3f}"
        f" {int(macro['support']):>8}"
    )
    print(SEP)


# ── Section 3: Sample Predictions Browser (first 10) ──────────────────────────
def print_samples(data, n=10):
    print("\n" + SEP)
    print(f"  SAMPLE PREDICTIONS  (first {n} -- with confidence scores)")
    print(SEP)
    for i, d in enumerate(data[:n], 1):
        actual    = d["Label"]
        predicted = d["gpt_choice"]
        reason    = d.get("gpt_reason", "").strip()
        tick = "[OK]" if predicted == actual else "[WRONG]"
        print(f"\n  Sample {i}")
        print(f"    Actual:    {actual}")
        print(f"    Predicted: {predicted} {tick}")
        if reason:
            short = wrap_text(reason)
            print(f"    GPT said:  \"{short}\"")
    print("\n" + SEP)


# ── Section 4: Worst Cases (3 wrong predictions) ──────────────────────────────
def print_worst_cases(data, n=3):
    wrong = [d for d in data if d["Label"] != d["gpt_choice"]]
    print("\n" + SEP)
    print(f"  WORST CASES  (first {n} incorrect predictions)")
    print(SEP)
    for i, d in enumerate(wrong[:n], 1):
        actual    = d["Label"]
        predicted = d["gpt_choice"]
        reason    = d.get("gpt_reason", "").strip()
        short = wrap_text(reason)
        print(f"\n  Case {i}")
        print(f"    Actual:    {actual}")
        print(f"    Predicted: {predicted} [WRONG]")
        print(f"    GPT said:  \"{short}\"")
    total_wrong = len(wrong)
    total = len(data)
    print(f"\n  ({total_wrong} wrong out of {total} LLM-evaluated samples)")
    print("\n" + SEP)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + SEP2)
    print("  IDS-LLM RESULTS VIEWER")
    print(SEP2)

    with_prob = load_json(WITH_PROB_JSON)

    print_summary()
    print_per_class(with_prob)
    print_samples(with_prob, n=10)
    print_worst_cases(with_prob, n=3)

    print(SEP2 + "\n")
