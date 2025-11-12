# COREBench A2A – Green (Referee) & Purple (Solver)

This repository hosts a minimal **Agent‑to‑Agent (A2A)** pipeline that you can run locally to evaluate “capsule” tasks from a COREBench‑style dataset.  
It wires two lightweight FastAPI micro‑services:

- **Green Agent (referee / grader)** – selects a capsule from `core_train.json`, builds a task, receives a **report** from Purple, then **grades** it.
- **Purple Agent (solver / runner)** – solves the task and posts a **report** back to Green. Two solver paths are supported:
  1. **LLM fallback (OpenAI)** – always available; no external repos; returns best‑effort answers.
  2. **AutoGPT‑CORE wrapper (experimental)** – shells into the **official AutoGPT‑CORE agent** (from the `siegelz/core-bench` repo) either directly or **in Docker** for Linux parity.

> Why the split? The A2A interface makes it easy to swap solvers (Purple) while keeping the grader (Green) stable and comparable.

---

## 1) Repository Layout

```
corebench_a2a_ab/
├─ core_train.json                # Local training capsules (what Green samples from)
├─ green_agent/
│  ├─ main.py                     # Green FastAPI app (dispatch, submit_results, grade)
│  └─ corebench_dataset_grader.py # Simple “exact match” grading fallback
├─ purple_agent/
│  ├─ main.py                     # Purple FastAPI app (start_task, evaluation)
│  ├─ llm_solver.py               # Minimal OpenAI-based solver
│  ├─ corebench_runner.py         # Harness that runs external agents or Docker
│  ├─ bridges/                    # Optional shims (e.g., docker / sudo helpers)
│  └─ data/                       # Logs & transient reports
├─ shared/schemas.py              # Pydantic models: TaskSpec, Report, Grade, etc.
└─ Dockerfile.autogpt-a2a         # (Optional) Reference image for AutoGPT-CORE path
```

---

## 2) What’s a “capsule”?

A capsule describes a **paper / code artifact** and a **task prompt** with **expected results**. Green uses `capsule_id` to create a task; Purple reproduces the result and returns answers.  
Example (from your `core_train.json`):

- **capsule‑8197429** “Low‑Latency Live Video Streaming over a Low‑Earth‑Orbit Satellite Network with DASH”, task prompt: `Run 'plot.sh'`. Expected results include questions such as “report the name of the model with the highest average bitrate…”, answer “L2A‑LL”, and “report the x‑axis label…”, answer “Seconds”. fileciteturn0file0

> In our pipeline, Green converts the **results** dict/list into `questions` and keeps `task_prompt` as the `prompt` sent to Purple.

---

## 3) Quick Start (macOS / Linux)

> **Prereqs**: Python 3.12 (or 3.11), Docker Desktop (if you want to try the AutoGPT‑CORE path), and an OpenAI API key for the LLM fallback.

### 3.1 Create virtual envs and install deps

Open two terminals—one for **Purple**, one for **Green**.

```bash
# Terminal A – Purple
cd corebench_a2a_ab/purple_agent
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Terminal B – Green
cd corebench_a2a_ab/green_agent
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 Start the services

```bash
# Terminal A – Purple (LLM fallback mode by default)
export OPENAI_API_KEY=sk-...           # do NOT commit
python -m uvicorn purple_agent.main:app --host 0.0.0.0 --port 9002

# Terminal B – Green
python -m uvicorn green_agent.main:app  --host 0.0.0.0 --port 8000
```

You can check health:

```bash
curl -s http://localhost:9002/health     # Purple
curl -s http://localhost:8000/health     # Green
```

### 3.3 Run your first task (LLM fallback)

Dispatch from **any terminal**:

```bash
curl -X POST http://localhost:8000/admin/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "purple_base_url": "http://localhost:9002",
    "auth_token": "change-me-token",
    "parameters": {
      "local_corebench_train": "/ABSOLUTE/PATH/TO/corebench_a2a_ab/core_train.json",
      "capsule_id": "capsule-8197429",
      "env": { "openai_model": "gpt-4o-mini" }
    }
  }'
```

Then poll the run:

```bash
curl http://localhost:8000/runs/<RUN_ID>
```

- **Green** will show `status: graded` when Purple has submitted a report.
- **score** is “exact‑match” on strings as a simple baseline (see `corebench_dataset_grader.py`).

> If the LLM cannot infer answers reliably, it may return `"UNKNOWN"`—that’s by design to keep the pipeline end‑to‑end functional while you iterate on a stronger solver.

---

## 4) (Optional) Use the official AutoGPT‑CORE solver

The AutoGPT‑CORE scripts in `siegelz/core-bench` assume a **Linux** environment with `apt` and sometimes a specific Python (3.10). On macOS, the simplest path to parity is to run them **inside Docker**.

### 4.1 Clone `core-bench` locally

```bash
export CORE_BENCH_HOME=$HOME/src/core-bench
git clone https://github.com/siegelz/core-bench.git "$CORE_BENCH_HOME"
```

### 4.2 Build a parity image (tested on Apple Silicon)

We provide a conservative base image that already has Python 3.10 and common build tools so the agent scripts don’t try to `apt install` interactively.

```Dockerfile
# Dockerfile.autogpt-a2a (root of this repo)
FROM python:3.10-slim

# Basic OS deps used by a number of agent scripts
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
# Nothing else installed here; we bind-mount core-bench and the environment at runtime.
```

Build it:

```bash
docker build -t autogpt-core-a2a:py3.10 -f Dockerfile.autogpt-a2a .
```

### 4.3 Start the services (same as §3.2)

Purple and Green should already be running.

### 4.4 Dispatch a run using the Dockerized AutoGPT‑CORE path

```bash
curl -X POST http://localhost:8000/admin/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "purple_base_url": "http://localhost:9002",
    "auth_token": "change-me-token",
    "parameters": {
      "local_corebench_train": "/ABSOLUTE/PATH/TO/corebench_a2a_ab/core_train.json",
      "capsule_id": "capsule-8197429",
      "env": {
        "agent_kind": "autogpt-core",
        "agent_script": "agents/autogpt-core/coreagent_easy_gpt4o-mini.sh",
        "agent_docker_image": "autogpt-core-a2a:py3.10",
        "timeout_sec": 1800
      }
    }
  }'
```

**What the harness does** (see `purple_agent/corebench_runner.py`):

- Writes `task_spec.json` and a `singleton_dataset.json` into `purple_agent/data/<RUN_ID>/work/`.
- Runs the agent script. If `agent_docker_image` is set, it executes:

  ```bash
  docker run --rm \
    -v $CORE_BENCH_HOME:/workspace:ro \
    -v <RUN_WORK_DIR>/autogpt-core/environment:/workspace/agents/autogpt-core/environment \
    -w /workspace \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e A2A_TIMEOUT_SEC=<TIMEOUT> \
    autogpt-core-a2a:py3.10 \
    bash -lc "bash /workspace/agents/autogpt-core/coreagent_easy_gpt4o-mini.sh"
  ```

- Searches for `report.json` in the work dir and returns it. If nothing is found or the script exits non‑zero, it **writes a fallback** report where every answer is `"UNKNOWN"` and still submits to Green—so your pipeline never hangs.

> **Note**: Some upstream scripts try to `sudo apt install ...` or expect `python3.10` via `apt`. Running inside our Docker image avoids interactive sudo and provides Python out‑of‑the‑box.

---

## 5) API Endpoints

### Green (port 8000)
- `GET /health` → `{status:"ok"}`
- `POST /admin/dispatch` → starts a run; body contains Purple URL, token, `parameters.local_corebench_train`, `parameters.capsule_id`, optional `env`
- `GET /runs/{run_id}` → run status + last grade
- `POST /submit_results?run_id=...` → **called by Purple**; expects a `Report`

### Purple (port 9002)
- `GET /health` → `{status:"ok", mode: "llm" | "autogpt-core"}`
- `POST /start_task` → enqueue a run; Purple calls solver and then POSTs a `Report` to Green
- `POST /evaluation` → **called by Green**; Purple stores the last `Grade` for convenience

---

## 6) Request/Response Schemas (simplified)

```python
# shared/schemas.py (selected fields)
class TaskSpec(BaseModel):
    task_id: str
    task_name: str | None = None
    prompt: str
    questions: list[Question]  # [{"id":"q1","text":"..."}]
    time_limit_sec: int = 1800
    env: dict = {}
    callback_submit_url: str
    callback_auth_token: str | None = None
    callback_results_url: str | None = None

class Report(BaseModel):
    task_id: str
    answers: list[AnswerItem]  # [{"id":"q1","answer":"..."}]
    metadata: dict = {}

class Grade(BaseModel):
    task_id: str
    passed: bool
    score: float
    details: dict = {}  # per-question breakdown
```

---

## 7) Troubleshooting Cheatsheet

- **`status: running` forever**  
  Purple never posted back. Check Purple logs: `purple_agent/data/<RUN_ID>/harness_stdout.log` & `harness_stderr.log`. If you see `sudo: a password is required` or `apt ... interactive`, switch to the **Docker** path and the reference image in §4.

- **`UNKNOWN` answers**  
  That’s the harness fallback when the agent crashed or didn’t emit a `report.json`. Look at `harness_error.txt` and the stdout/stderr logs to fix the upstream script. The fallback is intentional so the Green pipeline completes and you still get a grade.

- **`ModuleNotFoundError: shared` when starting Purple**  
  Ensure `sys.path` includes the project root; `purple_agent/main.py` already does `sys.path.insert(0, ROOT)`. Start Uvicorn from the **project root**, e.g.,  
  `python -m uvicorn purple_agent.main:app --host 0.0.0.0 --port 9002`

- **OpenAI `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`**  
  You probably have an older `openai` package. `pip install --upgrade "openai>=1.40.0"` or remove custom proxy kwargs from env/wrappers.

- **Docker image not found**  
  Build it first: `docker build -t autogpt-core-a2a:py3.10 -f Dockerfile.autogpt-a2a .` and make sure the tag matches `agent_docker_image` in your dispatch payload.

- **AutoGPT‑CORE scripts expect `python3.10` from apt**  
  Use the provided image (§4.2) which already ships with Python 3.10; don’t rely on interactive `apt` inside the container.

---

## 8) How grading works

Green first tries `core_bench.evaluate` **if available**; otherwise it uses `corebench_dataset_grader.py` (string exact‑match) to compute per‑question correctness and an overall score. For figure questions, answers are compared as normalized strings.

---

## 9) Security & Secrets

- Put API keys in environment variables (e.g., `OPENAI_API_KEY`) — never in code or JSON requests.
- `GREEN_CALLBACK_TOKEN` protects the Green `/submit_results` endpoint; set it in both agents’ env if you change it.

---

## 10) Minimal CURL Cookbook

LLM fallback:
```bash
curl -X POST http://localhost:8000/admin/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "purple_base_url":"http://localhost:9002",
    "auth_token":"change-me-token",
    "parameters":{
      "local_corebench_train":"/ABSOLUTE/PATH/TO/core_train.json",
      "capsule_id":"capsule-8197429",
      "env":{"openai_model":"gpt-4o-mini"}
    }
  }'
```

AutoGPT‑CORE via Docker (recommended for parity):
```bash
curl -X POST http://localhost:8000/admin/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "purple_base_url":"http://localhost:9002",
    "auth_token":"change-me-token",
    "parameters":{
      "local_corebench_train":"/ABSOLUTE/PATH/TO/core_train.json",
      "capsule_id":"capsule-8197429",
      "env":{
        "agent_kind":"autogpt-core",
        "agent_script":"agents/autogpt-core/coreagent_easy_gpt4o-mini.sh",
        "agent_docker_image":"autogpt-core-a2a:py3.10",
        "timeout_sec":1800
      }
    }
  }'
```

---

## 11) Known Limitations

- AutoGPT‑CORE scripts upstream occasionally assume root privileges or specific distro packages. Our Docker recipe mitigates this, but some capsules may still require manual adaptation.
- The current grader uses exact‑match for string answers; for numeric answers there’s no tolerance band in the fallback (you can extend the grader).

---

## 12) License and Credits

- Local A2A scaffolding (this repo): MIT License (adjust as you wish).  
- AutoGPT‑CORE and COREBench assets remain under their original licenses and ownership.
