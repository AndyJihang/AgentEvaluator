# purple_agent/corebench_runner.py
from __future__ import annotations

import os, sys, json, time, tarfile, shutil, signal, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import urllib.request

@dataclass
class RunnerConfig:
    core_bench_home: str                   # path to your local clone of siegelz/core-bench
    agent_kind: str = "autogpt-core"       # keep for compatibility
    agent_script: Optional[str] = None     # defaulted below
    benchmark_level: str = "a2a"           # informational only
    no_gpu: bool = True                    # informational only
    env_overrides: Dict[str, str] = None
    agent_docker_image: Optional[str] = "autogpt-core-a2a:py3.10"
    timeout_sec: int = 1800

def _write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _spawn(cmd: List[str], cwd: Path, env: Dict[str, str], run_dir: Path, timeout_sec: int) -> int:
    (run_dir / "harness_stdout.log").parent.mkdir(parents=True, exist_ok=True)
    with (run_dir / "harness_stdout.log").open("w", encoding="utf-8") as fo, \
         (run_dir / "harness_stderr.log").open("w", encoding="utf-8") as fe:
        fo.write("$ " + " ".join(subprocess.list2cmdline([x]) for x in cmd) + "\n")
        fo.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=fo, stderr=fe, start_new_session=True)
        try:
            return proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            fe.write(f"\n[HARNESS] Timeout after {timeout_sec}s. Killing process group.\n")
            fe.flush()
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            return 124

def _ensure_env_dirs(run_dir: Path) -> Dict[str, Path]:
    work = run_dir / "work"
    env_dir = work / "autogpt-core" / "environment"
    autogpt_dir = work / "autogpt-core" / "autogpt"
    for p in [env_dir, autogpt_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {"work": work, "env_dir": env_dir, "autogpt_dir": autogpt_dir}

def _write_task_txt(task_dict: Dict[str, Any], env_dir: Path):
    # Minimal task.txt content the upstream script can read
    capsule_id = task_dict.get("task_id", "")
    prompt = (task_dict.get("prompt") or "").strip()
    qs = [q.get("text", "") for q in (task_dict.get("questions") or [])]
    lines = []
    lines.append(f"capsule_id: {capsule_id}")
    lines.append("task:")
    lines.append(prompt)
    lines.append("")
    lines.append("questions:")
    for i, q in enumerate(qs, 1):
        lines.append(f"- {q}")
    (env_dir / "task.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_openai_env(autogpt_dir: Path):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return
    dot_env = autogpt_dir / ".env"
    buf = []
    buf.append(f"OPENAI_API_KEY={key}")
    # keep optional model override if user set it in env
    model = os.getenv("AUTOGPT_MODEL")
    if model:
        buf.append(f"AUTOGPT_MODEL={model}")
    dot_env.write_text("\n".join(buf) + "\n", encoding="utf-8")

def _download_and_unpack_capsule(capsule_id: str, target_env_dir: Path):
    """
    Mirror harness behavior: download capsule tarball and unpack under environment/.
    URL per README: https://corebench.cs.princeton.edu/capsules/capsule-XXXXXXX.tar.gz
    """
    url = f"https://corebench.cs.princeton.edu/capsules/{capsule_id}.tar.gz"
    tgt = target_env_dir / f"{capsule_id}.tar.gz"
    try:
        with urllib.request.urlopen(url) as r, open(tgt, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:
        # Give the agent a directory stub anyway; some tasks can still run without full code
        (target_env_dir / capsule_id).mkdir(parents=True, exist_ok=True)
        (target_env_dir / "capsule_download_error.txt").write_text(str(e), encoding="utf-8")
        return
    try:
        with tarfile.open(tgt, "r:gz") as tf:
            tf.extractall(path=target_env_dir)
    finally:
        try:
            tgt.unlink(missing_ok=True)
        except Exception:
            pass

def _find_report(env_dir: Path) -> Optional[Path]:
    # The script writes report.json in environment/
    p = env_dir / "report.json"
    if p.exists() and p.stat().st_size > 0:
        return p
    # search just in case
    for q in env_dir.glob("**/report.json"):
        try:
            if q.stat().st_size > 0:
                return q
        except Exception:
            pass
    return None

def run_autogpt_core(task, run_dir: Path, cfg: RunnerConfig) -> str:
    """
    Run the upstream AutoGPT-CORE wrapper inside a Docker container:
      - create harness environment (task.txt + capsule) under run_dir/work/autogpt-core/environment
      - mount it at /workspace/agents/autogpt-core/environment
      - mount a writable /workspace/agents/autogpt-core/autogpt (for .env and anything it creates)
      - mount the user's core-bench repo at /workspace
      - execute agents/autogpt-core/coreagent_easy_gpt4o-mini.sh
      - return the path to report.json; on failure, create an UNKNOWN report
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    # 1) persist TaskSpec for debugging
    task_dict = task.model_dump()
    _write_json(run_dir / "task_spec.json", task_dict)

    # 2) prepare environment folders + task.txt + OpenAI .env
    paths = _ensure_env_dirs(run_dir)
    env_dir = paths["env_dir"]
    autogpt_dir = paths["autogpt_dir"]
    _write_task_txt(task_dict, env_dir)
    _write_openai_env(autogpt_dir)

    # 3) (optional) download capsule unless disabled
    capsule_id = task_dict.get("task_id", "")
    if os.getenv("A2A_SKIP_CAPSULE_DOWNLOAD", "0") not in ("1", "true", "True"):
        _download_and_unpack_capsule(capsule_id, env_dir)

    # 4) run the upstream shell wrapper in a container
    repo = Path(cfg.core_bench_home).resolve()
    agent_script = cfg.agent_script or "agents/autogpt-core/coreagent_easy_gpt4o-mini.sh"

    env = os.environ.copy()
    # pass through useful env vars
    env["A2A_TIMEOUT_SEC"] = str(int(task_dict.get("time_limit_sec") or cfg.timeout_sec))
    for k, v in (cfg.env_overrides or {}).items():
        env[str(k)] = str(v)

    # NOTE: mount /workspace as RW (autogpt script may create subfolders under agents/autogpt-core)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{repo}:/workspace",  # RW on purpose
        "-v", f"{(run_dir / 'work' / 'autogpt-core' / 'environment')}:/workspace/agents/autogpt-core/environment",
        "-v", f"{(run_dir / 'work' / 'autogpt-core' / 'autogpt')}:/workspace/agents/autogpt-core/autogpt",
        "-w", "/workspace",
    ]

    # pass env to container
    for key in ["OPENAI_API_KEY", "AUTOGPT_MODEL", "A2A_TIMEOUT_SEC"]:
        if env.get(key):
            cmd += ["-e", f"{key}={env[key]}"]
    # any additional overrides
    for k, v in (cfg.env_overrides or {}).items():
        cmd += ["-e", f"{k}={v}"]

    image = cfg.agent_docker_image or "autogpt-core-a2a:py3.10"
    # IMPORTANT: do not prepend apt-get here; let the upstream script manage itself
    cmd += [image, "bash", "-lc", f"bash /workspace/{agent_script}"]

    rc = _spawn(cmd, cwd=run_dir, env=os.environ.copy(), run_dir=run_dir, timeout_sec=cfg.timeout_sec)

    report = _find_report(env_dir)
    if rc == 0 and report:
        return str(report.resolve())

    # Fallback: write UNKNOWN report so Green still grades
    (run_dir / "harness_error.txt").write_text(
        f"AutoGPT-CORE run failed or report not found. rc={rc}\n", encoding="utf-8"
    )
    answers_map = {}
    for q in (task_dict.get("questions") or []):
        qtext = q.get("text") or q.get("id") or ""
        answers_map[qtext] = "UNKNOWN"
    rp = env_dir / "report.json"
    _write_json(rp, answers_map)
    return str(rp.resolve())
