
from __future__ import annotations

import os, json, uuid, time, random
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from shared.schemas import TaskSpec, Report, Grade, DispatchRequest, DispatchResponse, Question
from green_agent.corebench_dataset_grader import grade_by_gold

load_dotenv()

DATA_DIR = Path(os.getenv("GREEN_DATA_DIR", Path(__file__).parent / "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
GREEN_PUBLIC_BASE_URL = os.getenv("GREEN_PUBLIC_BASE_URL", "http://localhost:8000")
GREEN_CALLBACK_TOKEN = os.getenv("GREEN_CALLBACK_TOKEN", "change-me-token")

DEFAULT_LOCAL_DATASET = Path(os.getenv("GREEN_LOCAL_DATASET", ROOT / "core_train.sample.json")).resolve()

app = FastAPI(title="CoreBench Green Agent (AB-style)", version="0.3.0")
RUNS: Dict[str, Dict[str, Any]] = {}

# ------------------------ utils ------------------------
def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _load_local_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Local dataset not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Local dataset appears empty or invalid")
    return rows

def _gold_map_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # Dataset stores results as a list of dicts (possibly duplicates); we choose the first occurrence per question.
    qid2gold = {}
    qtexts = []
    # Build ordered list of unique questions
    seen = set()
    for block in row.get("results", []):
        for q, val in block.items():
            if q not in seen:
                qtexts.append(q)
                seen.add(q)
    # assign ids q1..qn
    for i, q in enumerate(qtexts, start=1):
        qid2gold[f"q{i}"] = val = row.get("results", [{}])[0].get(q, "")
        # fallback: if first block doesn't contain, search others
        if val == "":
            for blk in row.get("results", [])[1:]:
                if q in blk:
                    val = blk[q]; break
        qid2gold[f"q{i}"] = val
    return qid2gold, qtexts

def _sample_or_select(rows: List[Dict[str, Any]], capsule_id: Optional[str]) -> Dict[str, Any]:
    if capsule_id:
        for r in rows:
            if r.get("capsule_id") == capsule_id:
                return r
        raise HTTPException(404, f"capsule_id not found in dataset: {capsule_id}")
    return random.choice(rows)

# ------------------------ endpoints ------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/.well-known/agent-card.json")
def agent_card():
    card = {
        "name": "CoreBench Green Agent (AB-style)",
        "version": "0.3.0",
        "capabilities": ["referee", "grading", "dispatch"],
        "endpoints": {
            "dispatch": "/admin/dispatch",
            "submit_results": "/submit_results",
            "health": "/health"
        }
    }
    return JSONResponse(card)

@app.get("/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Unknown run_id")
    return RUNS[run_id]

def _build_task_spec(row: Dict[str, Any], run_id: str, notify_url: str) -> TaskSpec:
    qid2gold, qtexts = _gold_map_from_row(row)
    questions = [Question(id=f"q{i+1}", text=qt) for i, qt in enumerate(qtexts)]
    prompt = row.get("task_prompt") or ""
    capsule_url = row.get("capsule_doi") or None
    task_id = row.get("capsule_id") or "unknown-task"

    submit_url = f"{GREEN_PUBLIC_BASE_URL.rstrip('/')}/submit_results?run_id={run_id}"

    return TaskSpec(
        task_id=task_id,
        task_name=row.get("capsule_title",""),
        prompt=prompt,
        questions=questions,
        capsule_url=capsule_url,
        time_limit_sec=1800,
        env={},  # pass-through from dispatch if needed
        callback_submit_url=submit_url,
        callback_auth_token=GREEN_CALLBACK_TOKEN,
        callback_results_url=notify_url
    )

@app.post("/admin/dispatch", response_model=DispatchResponse)
def dispatch(req: DispatchRequest):
    # Load dataset
    local_path = Path(req.parameters.get("local_corebench_train") or DEFAULT_LOCAL_DATASET)
    rows = _load_local_dataset(local_path)
    # Choose capsule
    row = _sample_or_select(rows, req.parameters.get("capsule_id") or req.task_id)

    run_id = str(uuid.uuid4())
    notify_url = req.purple_results_url or f"{req.purple_base_url.rstrip('/')}/evaluation"
    task_spec = _build_task_spec(row, run_id, notify_url)
    # attach env overrides if provided
    task_spec.env = req.parameters.get("env", {})

    RUNS[run_id] = {
        "status": "dispatched",
        "purple": req.purple_base_url,
        "task_id": task_spec.task_id,
        "qid2text": {q.id: q.text for q in task_spec.questions},
        "local_train_json": str(local_path),
        "started_at": time.time(),
        "notify_url": notify_url,
    }

    try:
        r = requests.post(f"{req.purple_base_url.rstrip('/')}/start_task", json=task_spec.model_dump(), timeout=30)
        r.raise_for_status()
    except Exception as e:
        RUNS[run_id]["status"] = "dispatch_error"
        RUNS[run_id]["error"] = str(e)
        raise HTTPException(502, f"Failed to call Purple /start_task: {e}")

    RUNS[run_id]["status"] = "running"
    return DispatchResponse(run_id=run_id, status="running", message="Task dispatched to Purple.")

@app.post("/submit_results")
async def submit_results(request: Request):
    # token check (optional/lenient for local dev)
    auth = request.headers.get("authorization", "")
    token = auth.replace("Bearer", "").strip() if auth else None
    if token and GREEN_CALLBACK_TOKEN and token != GREEN_CALLBACK_TOKEN:
        raise HTTPException(401, "Bad token")

    run_id = request.query_params.get("run_id")
    if not run_id or run_id not in RUNS:
        raise HTTPException(400, "Invalid or missing run_id")
    payload = await request.json()
    report = Report(**payload)

    # Persist
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")

    # Grade (dataset-based fallback here)
    # Build gold map from dataset row we used earlier (store qid->text during dispatch)
    qid2text: Dict[str, str] = RUNS[run_id].get("qid2text", {})
    # reload dataset row by text -> gold
    # We stored local dataset path; find golds by matching qtext
    dataset_path = Path(RUNS[run_id]["local_train_json"])
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    # find the exact row by capsule id
    target = next((r for r in rows if r.get("capsule_id") == RUNS[run_id]["task_id"]), None)
    qid2gold = {}
    if target:
        # build mapping qtext -> gold (first occurrence)
        seen = {}
        for blk in target.get("results", []):
            for q, v in blk.items():
                if q not in seen:
                    seen[q] = v
        for qid, text in qid2text.items():
            qid2gold[qid] = seen.get(text, "")

    qid2pred = {a["id"]: a["answer"] for a in [ai.model_dump() for ai in report.answers]}
    g = grade_by_gold(qid2pred, qid2gold)
    grade = Grade(task_id=report.task_id, passed=g["passed"], score=float(g["score"]), details={"per_question": g["per_question"]})
    (run_dir / "grade.json").write_text(json.dumps(grade.model_dump(), indent=2), encoding="utf-8")

    RUNS[run_id]["status"] = "graded"
    RUNS[run_id]["grade"] = grade.model_dump()

    # Notify Purple
    try:
        cb = RUNS[run_id].get("notify_url")
        if cb:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            requests.post(cb, json=grade.model_dump(), headers=headers, timeout=30)
    except Exception as e:
        RUNS[run_id]["notify_error"] = str(e)

    return {"ok": True, "grade": grade.model_dump()}

def run():
    import uvicorn
    uvicorn.run("green_agent.main:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")), reload=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        run()
    else:
        print("Usage: python main.py run")
