from __future__ import annotations

import json, os, time, uuid
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, BackgroundTasks, HTTPException
from dotenv import load_dotenv
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.schemas import TaskSpec, Report, AnswerItem, Grade
from purple_agent.corebench_runner import RunnerConfig, run_autogpt_core

load_dotenv()

DATA_DIR = Path(os.getenv("PURPLE_DATA_DIR", Path(__file__).parent / "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Purple Agent (Solver)", version="0.4.1")
RUNS: Dict[str, Dict[str, Any]] = {}

@app.get("/health")
def health():
    return {"status": "ok", "mode": "autogpt-core"}

def _answers_from_kv_report(report_path: Path, task: TaskSpec) -> List[Dict[str, str]]:
    """
    Report is {<question text>: <answer>, ...}. Return [{id, answer}, ...] aligned to TaskSpec.
    """
    try:
        kv = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        kv = {}
    out = []
    for q in task.questions:
        qtext = q.text
        out.append({"id": q.id, "answer": str(kv.get(qtext, "UNKNOWN"))})
    return out

def _upload_report(task: TaskSpec, report: Report) -> None:
    stamp = str(int(time.time() * 1000))
    (DATA_DIR / f"{stamp}_report.json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    headers = {}
    if task.callback_auth_token:
        headers["Authorization"] = f"Bearer {task.callback_auth_token}"
    import requests
    r = requests.post(task.callback_submit_url, json=report.model_dump(), headers=headers, timeout=300)
    r.raise_for_status()

def run_task(run_id: str, task: TaskSpec) -> None:
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Require core-bench repo
    core_home = os.getenv("CORE_BENCH_HOME")
    if not core_home:
        raise HTTPException(500, "Set CORE_BENCH_HOME to your local clone of https://github.com/siegelz/core-bench")

    # Build runner config (AutoGPT-CORE in Docker by default)
    env_overrides = ((task.env or {}).get("env_overrides") or {})
    cfg = RunnerConfig(
        core_bench_home=core_home,
        agent_kind=(task.env or {}).get("agent_kind", "autogpt-core"),
        agent_script=(task.env or {}).get("agent_script", "agents/autogpt-core/coreagent_easy_gpt4o-mini.sh"),
        agent_docker_image=(task.env or {}).get("agent_docker_image", "autogpt-core-linux:py3.10"),
        benchmark_level=(task.env or {}).get("benchmark_level", "a2a"),
        no_gpu=bool((task.env or {}).get("no_gpu", True)),
        env_overrides={k: str(v) for k, v in env_overrides.items()},
        timeout_sec=int((task.env or {}).get("timeout_sec", 900)),
    )

    # Run the official wrapper in Docker and collect the report
    report_json_path = run_autogpt_core(task, run_dir, cfg)
    answers = _answers_from_kv_report(Path(report_json_path), task)

    report = Report(
        task_id=task.task_id,
        answers=[AnswerItem(**a) for a in answers],
        metadata={"mode": "autogpt-core", "report_path": str(report_json_path)},
    )

    _upload_report(task, report)
    RUNS[run_id]["submitted"] = True

@app.post("/start_task")
def start_task(task: TaskSpec, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    RUNS[run_id] = {"status": "running", "task_id": task.task_id, "created_at": time.time()}
    background_tasks.add_task(run_task, run_id, task)
    return {"run_id": run_id, "status": "accepted"}

@app.post("/evaluation")
def receive_evaluation(grade: Grade):
    (DATA_DIR / "last_grade.json").write_text(json.dumps(grade.model_dump(), indent=2), "utf-8")
    return {"ok": True, "received": grade.model_dump()}
