"""Shared code for all methods: loading data, parsing answers, scoring and run logging.

Everything that must be identical for the regex baseline, MedGemma and Gemma 3 lives here,
so the comparison stays fair: the same parser, the same metrics and the same logging.

Every method produces a raw output string per report. For the LLMs this is the generated
text; for the regex it is a JSON string in the same schema. `parse_answer` turns it into
primary / secondary / score and a status, `score_run` computes the metrics, and `log_run`
saves the run to `results/<subset>/`.

Answer schema (what the prompt asks for):
    {"primary": 3, "secondary": 4}        Gleason score found
    {"primary": null, "secondary": null}  no Gleason score in the report
An optional "score" key is accepted and checked against primary + secondary.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "prostate_reports.csv"
RESULTS_DIR = ROOT / "results"

# Gleason patterns are graded 1-5. In practice only 3-5 are reported, but one TCGA label is 2+4,
# so any pattern in 1-5 is accepted as valid and anything else is invalid.
VALID_PATTERNS = {1, 2, 3, 4, 5}

# MedGemma 1.5 writes its reasoning before this marker and the answer after it
ANSWER_MARKER = "<unused95>"

# Columns written to v<N>.csv (the report text is left out; it is in data/prostate_reports.csv)
RUN_COLUMNS = [
    "patient_id", "method", "model_id", "raw_output",
    "pred_primary", "pred_secondary", "pred_score", "status", "parse_error",
    "pair_in_text", "possible_hallucination",
    "label_primary", "label_secondary", "label_score", "correct_score", "correct_pair",
]

# Config keys that differ on every run and are not "changes" to the setup
IGNORED_IN_DIFF = {"date", "run", "note", "log_run"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(split, n=None):
    """Load the prostate reports with their labels.

    Args:
        split: "tuning", "eval" or "all".
        n: if given, keep only the first n reports (for quick test runs).

    Returns:
        DataFrame with patient_id, patient_filename, text, label_primary, label_secondary,
        label_score and split.
    """
    data = pd.read_csv(DATA_CSV)
    if split != "all":
        data = data[data["split"] == split]
    if n is not None:
        data = data.head(n)
    return data.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def answer_text(raw):
    """Return the part of a model output that holds the answer.

    MedGemma puts its reasoning before `<unused95>` and the answer after it. If the marker
    is missing (Gemma 3, regex, or MedGemma with reasoning blocked) the whole text is returned.

    Args:
        raw: the raw output string.

    Returns:
        The text after the last `<unused95>`, or the whole text.
    """
    return raw.split(ANSWER_MARKER)[-1]


def _to_pattern(value):
    """Convert a JSON value to a pattern number.

    Args:
        value: a value from the parsed JSON (int, digit string, None, ...).

    Returns:
        (pattern, ok): pattern is an int or None; ok is False if the value cannot be a pattern.
    """
    if value is None:
        return None, True
    # bool is a subclass of int in Python, so exclude it explicitly
    if isinstance(value, int) and not isinstance(value, bool):
        return value, True
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip()), True
    return None, False


def parse_answer(raw):
    """Parse a raw output into primary / secondary / score and a status.

    The first JSON object in the answer text that contains both "primary" and "secondary"
    is used. Markdown code fences around it are fine.

    Status values:
        valid:      both patterns are numbers in VALID_PATTERNS (and match "score", if given)
        not_found:  both patterns are null (the method says the report has no score)
        invalid:    anything else; `parse_error` says why

    Args:
        raw: the raw output string (model output, or the regex's JSON string).

    Returns:
        dict with pred_primary, pred_secondary, pred_score (ints or None), status and
        parse_error (None unless status is "invalid").
    """
    result = {"pred_primary": None, "pred_secondary": None, "pred_score": None,
              "status": "invalid", "parse_error": None}
    if not isinstance(raw, str):
        result["parse_error"] = "no_output"
        return result

    # Find the first {...} block that parses as JSON and has the expected keys.
    # The schema is flat, so a non-nested {...} match is enough.
    answer = None
    for block in re.findall(r"\{[^{}]*\}", answer_text(raw)):
        try:
            candidate = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "primary" in candidate and "secondary" in candidate:
            answer = candidate
            break
    if answer is None:
        result["parse_error"] = "no_json"
        return result

    primary, ok_p = _to_pattern(answer["primary"])
    secondary, ok_s = _to_pattern(answer["secondary"])
    if not (ok_p and ok_s):
        result["parse_error"] = "not_a_number"
        return result

    if primary is None and secondary is None:
        result["status"] = "not_found"
        return result
    if primary is None or secondary is None:
        result["parse_error"] = "one_pattern_missing"
        return result

    # Keep the parsed numbers even if invalid, so the error analysis can see them
    result["pred_primary"], result["pred_secondary"] = primary, secondary
    result["pred_score"] = primary + secondary
    if primary not in VALID_PATTERNS or secondary not in VALID_PATTERNS:
        result["parse_error"] = "pattern_out_of_range"
        return result

    # If the answer also states a score, it must agree with the patterns
    if answer.get("score") is not None:
        score, ok = _to_pattern(answer["score"])
        if not ok or score != primary + secondary:
            result["parse_error"] = "score_inconsistent"
            return result

    result["status"] = "valid"
    return result


# ---------------------------------------------------------------------------
# Hallucination check
# ---------------------------------------------------------------------------

def pair_in_text(text, primary, secondary):
    """Check whether a pattern pair is written somewhere in the report.

    This is a heuristic used to flag *possible* hallucinations for manual review. It looks for
    the common ways a pair is written:
        "3+4", "3 + 4", "3 and 4", "3 plus 4", "3 & 4"
        "primary ... 3 ... secondary ... 4" (within a short window)
    A pair that is only written as a sum ("Gleason score 7") or garbled by OCR is not found,
    so a flag is not proof of a hallucination.

    Args:
        text: the report text.
        primary, secondary: the predicted patterns (ints).

    Returns:
        True if the pair is found in the text.
    """
    text = text.lower()
    p, s = str(primary), str(secondary)
    # p followed by a joining word or symbol and then s, not part of a longer number
    joined = rf"(?<!\d){p}\s*(?:\+|&|and|plus)\s*{s}(?!\d)"
    # "primary <up to 40 chars> p ... secondary <up to 40 chars> s"
    labelled = rf"primary[^0-9]{{0,40}}{p}(?!\d).{{0,120}}?secondary[^0-9]{{0,40}}{s}(?!\d)"
    return bool(re.search(joined, text) or re.search(labelled, text, flags=re.DOTALL))


# ---------------------------------------------------------------------------
# Building the run table and scoring
# ---------------------------------------------------------------------------

def build_run_table(data, raw_outputs, config, extra=None):
    """Combine the reports, the raw outputs and the parsed answers into one table.

    Args:
        data: DataFrame from load_data (must contain text and the label columns).
        raw_outputs: list of raw output strings, one per row of `data`, in the same order.
        config: the run's config dict (method and model_id are copied into the table).
        extra: optional dict {column name: list of values, one per report} with method-specific
            columns to keep, e.g. token counts for the LLMs. They are added after RUN_COLUMNS.

    Returns:
        DataFrame with one row per report and the columns in RUN_COLUMNS (plus any extra columns).
    """
    assert len(raw_outputs) == len(data), "One raw output per report is needed"
    table = data.copy()
    table["method"] = config["method"]
    table["model_id"] = config.get("model_id")
    table["raw_output"] = raw_outputs

    parsed = pd.DataFrame([parse_answer(raw) for raw in raw_outputs], index=table.index)
    table = pd.concat([table, parsed], axis=1)
    # Nullable integers, so missing predictions show as empty instead of turning the column into floats
    for col in ["pred_primary", "pred_secondary", "pred_score"]:
        table[col] = table[col].astype("Int64")

    # Flag valid answers whose pair cannot be found in the report text
    table["pair_in_text"] = [
        pair_in_text(row.text, row.pred_primary, row.pred_secondary) if row.status == "valid" else None
        for row in table.itertuples()
    ]
    table["possible_hallucination"] = (table["status"] == "valid") & (table["pair_in_text"] == False)  # noqa: E712

    valid = table["status"] == "valid"
    table["correct_score"] = valid & (table["pred_score"] == table["label_score"])
    table["correct_pair"] = valid & (table["pred_primary"] == table["label_primary"]) \
        & (table["pred_secondary"] == table["label_secondary"])
    # pandas comparisons with <NA> give <NA>; count those as wrong
    table[["correct_score", "correct_pair"]] = table[["correct_score", "correct_pair"]].fillna(False).astype(bool)

    extra = extra or {}
    for name, values in extra.items():
        assert len(values) == len(table), f"Extra column {name} needs one value per report"
        table[name] = list(values)
    return table[RUN_COLUMNS + list(extra)]


def score_run(table):
    """Compute all metrics for one run.

    Not-found and invalid answers count as wrong for accuracy, macro-F1 and pair exact match.
    Macro-F1 averages the per-class F1 over the Gleason scores present in the labels, so a
    class that is missing from a subset (e.g. score 10 in the tuning set) does not count as 0.

    Args:
        table: DataFrame from build_run_table.

    Returns:
        dict with n, accuracy, macro_f1, f1_per_score, pair_exact_match, accuracy_when_valid,
        the rates of valid / not_found / invalid / possible_hallucination answers, counts of
        parse errors, and the confusion matrix ({true score: {predicted: count}}).
    """
    n = len(table)
    valid = table["status"] == "valid"

    # Predicted class per report: the score as a string, or "not_found" / "invalid"
    y_true = table["label_score"].astype(str)
    y_pred = table["pred_score"].astype(str).where(valid, table["status"])
    classes = sorted(table["label_score"].unique())
    class_names = [str(c) for c in classes]

    f1_values = f1_score(y_true, y_pred, labels=class_names, average=None, zero_division=0)
    confusion = pd.crosstab(y_true, y_pred)

    return {
        "n": n,
        "accuracy": round(float(table["correct_score"].mean()), 4),
        "macro_f1": round(float(f1_values.mean()), 4),
        "f1_per_score": {name: round(float(v), 4) for name, v in zip(class_names, f1_values)},
        "pair_exact_match": round(float(table["correct_pair"].mean()), 4),
        "accuracy_when_valid": round(float(table.loc[valid, "correct_score"].mean()), 4) if valid.any() else None,
        "valid_rate": round(float(valid.mean()), 4),
        "not_found_rate": round(float((table["status"] == "not_found").mean()), 4),
        "invalid_rate": round(float((table["status"] == "invalid").mean()), 4),
        "possible_hallucination_rate": round(float(table["possible_hallucination"].mean()), 4),
        "parse_errors": {k: int(v) for k, v in table["parse_error"].value_counts().items()},
        "confusion_matrix": {
            true: {pred: int(count) for pred, count in row.items() if count > 0}
            for true, row in confusion.iterrows()
        },
    }


# ---------------------------------------------------------------------------
# Run logging
# ---------------------------------------------------------------------------

def config_diff(old, new):
    """List the settings that differ between two configs.

    Long values (e.g. the prompt) are shown as "changed" instead of printed in full.

    Args:
        old, new: config dicts (old may be None for the first run).

    Returns:
        List of human-readable strings, one per changed setting.
    """
    if old is None:
        return ["first run"]
    changes = []
    for key in sorted(set(old) | set(new)):
        if key in IGNORED_IN_DIFF or old.get(key) == new.get(key):
            continue
        a, b = old.get(key, "<missing>"), new.get(key, "<missing>")
        if len(repr(a)) > 60 or len(repr(b)) > 60:
            changes.append(f"{key}: changed")
        else:
            changes.append(f"{key}: {a!r} -> {b!r}")
    return changes


def _run_numbers(subset_dir):
    """Return the sorted run numbers N that have a v<N>_config.json in a folder."""
    numbers = [int(m.group(1)) for f in subset_dir.glob("v*_config.json")
               if (m := re.fullmatch(r"v(\d+)_config\.json", f.name))]
    return sorted(numbers)


def rebuild_runs_csv(subset_dir):
    """Rebuild runs.csv in a subset folder from the v<N>_config.json and v<N>_scores.json files.

    Each row is one run: its number, date, method, model, note, what changed in the config
    since the previous run, and its scalar scores.

    Args:
        subset_dir: Path to results/<subset>/.

    Returns:
        The runs table as a DataFrame (also written to runs.csv).
    """
    rows, previous = [], None
    for n in _run_numbers(subset_dir):
        config = json.loads((subset_dir / f"v{n}_config.json").read_text())
        scores = json.loads((subset_dir / f"v{n}_scores.json").read_text())
        row = {
            "run": f"v{n}",
            "date": config.get("date"),
            "method": config.get("method"),
            "model_id": config.get("model_id"),
            "note": config.get("note"),
            "changes": "; ".join(config_diff(previous, config)),
        }
        # Keep only the single-number scores; the nested ones are in the JSON files
        row.update({k: v for k, v in scores.items() if not isinstance(v, dict)})
        rows.append(row)
        previous = config
    runs = pd.DataFrame(rows)
    runs.to_csv(subset_dir / "runs.csv", index=False)
    return runs


def log_run(config, table, scores, results_dir=RESULTS_DIR):
    """Save a run to results/<subset>/ as the next free v<N>, unless config["log_run"] is False.

    Writes v<N>.csv (the run table), v<N>_config.json (the config plus date and run number) and
    v<N>_scores.json, prints what changed since the previous run, and rebuilds runs.csv.
    Existing runs are never overwritten.

    Args:
        config: the run's config dict; must contain "subset" and "log_run".
        table: DataFrame from build_run_table.
        scores: dict from score_run.
        results_dir: root folder for results (only changed for testing).

    Returns:
        The run name (e.g. "v3"), or None for a test run.
    """
    if not config["log_run"]:
        print("Test run (log_run=False): nothing saved.")
        return None

    subset_dir = Path(results_dir) / config["subset"]
    subset_dir.mkdir(parents=True, exist_ok=True)
    numbers = _run_numbers(subset_dir)
    # The next number is above every v<N> file in the folder, even from an interrupted save
    used = [int(m.group(1)) for f in subset_dir.iterdir() if (m := re.match(r"v(\d+)[._]", f.name))]
    n = max(used, default=0) + 1
    run = f"v{n}"

    previous = json.loads((subset_dir / f"v{numbers[-1]}_config.json").read_text()) if numbers else None
    saved_config = {**config, "run": run, "date": datetime.now().isoformat(timespec="seconds")}

    # Write the scores and table before the config: the config file is what marks a run as
    # existing, so an interrupted save does not leave a run without scores
    table.to_csv(subset_dir / f"{run}.csv", index=False)
    (subset_dir / f"{run}_scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    (subset_dir / f"{run}_config.json").write_text(json.dumps(saved_config, indent=2, default=str) + "\n")
    rebuild_runs_csv(subset_dir)

    print(f"Saved {subset_dir.relative_to(ROOT) if subset_dir.is_relative_to(ROOT) else subset_dir}/{run}")
    print("Changes since the previous run:")
    for change in config_diff(previous, saved_config):
        print(f"  - {change}")
    return run
