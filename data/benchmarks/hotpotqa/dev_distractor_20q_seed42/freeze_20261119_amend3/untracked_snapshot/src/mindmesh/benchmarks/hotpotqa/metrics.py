"""
metrics.py -- HotpotQA metrics.

Answer EM / F1, supporting-fact EM / P / R / F1 and joint EM / F1 follow the
official evaluation script (hotpot_evaluate_v1.py, HotpotQA authors):
  * normalize_answer: lowercase, strip punctuation, articles, extra spaces
  * F1 is 0 when either side is yes/no/noanswer and they differ
  * SP metrics are computed over sets of (title, sent_id) pairs
  * joint P = ans_P * sp_P, joint R = ans_R * sp_R, joint EM = ans_EM * sp_EM

Retrieval Recall@k / MRR are MindMesh additions (not part of the official
script): gold = the chunk IDs of the paragraphs that hold supporting facts.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO = (0.0, 0.0, 0.0)
    special = ["yes", "no", "noanswer"]
    if normalized_prediction in special and normalized_prediction != normalized_ground_truth:
        return ZERO
    if normalized_ground_truth in special and normalized_prediction != normalized_ground_truth:
        return ZERO
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def answer_metrics(prediction: str, gold: str) -> dict:
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return {"em": em, "f1": f1, "prec": prec, "recall": recall}


def sp_metrics(prediction: list, gold: list) -> dict:
    cur_sp_pred = {tuple(x) for x in prediction}
    gold_sp_pred = {tuple(x) for x in gold}
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return {"em": em, "f1": f1, "prec": prec, "recall": recall}


def joint_metrics(ans: dict, sp: dict) -> dict:
    joint_prec = ans["prec"] * sp["prec"]
    joint_recall = ans["recall"] * sp["recall"]
    if joint_prec + joint_recall > 0:
        joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
    else:
        joint_f1 = 0.0
    return {"em": ans["em"] * sp["em"], "f1": joint_f1, "prec": joint_prec, "recall": joint_recall}


def retrieval_metrics(retrieved: list[str], gold: list[str], ks=(1, 3, 5, 10)) -> dict:
    """Recall@k = fraction of gold chunks in the top-k; MRR = 1/rank of first gold."""
    gold_set = set(gold)
    out = {}
    for k in ks:
        top = retrieved[:k]
        out[f"recall@{k}"] = (sum(1 for g in gold_set if g in top) / len(gold_set)) if gold_set else 0.0
    mrr = 0.0
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold_set:
            mrr = 1.0 / rank
            break
    out["mrr"] = mrr
    out["all_gold_in_top10"] = float(gold_set.issubset(set(retrieved[:10]))) if gold_set else 0.0
    return out
