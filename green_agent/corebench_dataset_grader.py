
from __future__ import annotations
import math, re
from typing import Dict, Any

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()

def _is_number(x: str) -> bool:
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False

def grade_by_gold(qid2pred: Dict[str, str], qid2gold: Dict[str, str]) -> Dict[str, Any]:
    """
    Compare predictions to gold with small numeric tolerance; return detailed stats.
    """
    per_q = {}
    correct = 0
    total = 0

    for qid, gold in qid2gold.items():
        pred = qid2pred.get(qid, "")
        ok = False
        if _is_number(gold) and _is_number(pred):
            g = float(gold); p = float(pred)
            ok = math.isclose(g, p, rel_tol=1e-3, abs_tol=1e-3)
        else:
            ok = _norm(gold) == _norm(pred)

        per_q[qid] = {"gold": str(gold), "pred": str(pred), "correct": bool(ok)}
        total += 1
        if ok:
            correct += 1

    score = correct / total if total else 0.0
    return {"passed": bool(score >= 0.5), "score": score, "per_question": per_q}
