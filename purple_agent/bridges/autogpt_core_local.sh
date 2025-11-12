#!/usr/bin/env bash
set -euo pipefail

# Resolve environment dir (where task.txt/report.json live)
ENV_DIR="${ENV_DIR:-$(cd "$(dirname "$0")"/.. && pwd)/environment}"
TASK_TXT="$ENV_DIR/task.txt"
REPORT_JSON="$ENV_DIR/report.json"

if [[ ! -f "$TASK_TXT" ]]; then
  echo "ERROR: task.txt not found at $TASK_TXT" >&2
  echo '{}' > "$REPORT_JSON"
  exit 0
fi

# Use your current Python to call OpenAI and write report.json
python3 - <<'PY' "$TASK_TXT" "$REPORT_JSON"
import os, sys, json, re
from pathlib import Path

task_txt = Path(sys.argv[1])
report_json = Path(sys.argv[2])

text = task_txt.read_text(encoding="utf-8")

# ---- Parse questions from "TASK QUESTIONS:" section ----
qs=[]
in_q=False
for line in text.splitlines():
    if line.strip().startswith("TASK QUESTIONS:"):
        in_q=True
        continue
    if in_q:
        s=line.strip()
        if not s:
            continue
        m=re.match(r"^\s*\d+\.\s*(.+)$", s)
        if m:
            qs.append(m.group(1).strip())

# Extract the prompt block (optional, improves answers)
prompt = ""
m=re.search(r"TASK PROMPT:\s*(.*?)(?:\n\s*\n|$)", text, flags=re.S)
if m:
    prompt = m.group(1).strip()

# ---- Call OpenAI ----
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("AUTOGPT_MODEL","gpt-4o-mini")

sys_msg = "Answer strictly with a single JSON object whose keys are exactly the given questions."
user = "Task:\n" + prompt + "\n\nReturn a JSON object mapping each question to a short answer.\nQuestions:\n" + json.dumps(qs, indent=2)

try:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys_msg},{"role":"user","content":user}],
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    # Extract a JSON object from the model output
    m = re.search(r"\{(?:[^{}]|(?0))*\}", raw, flags=re.S)
    obj = {}
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = {}
    out = {}
    for q in qs:
        out[q] = obj.get(q, "UNKNOWN")
    report_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
except Exception:
    # Failsafe: UNKNOWN for everything
    out = {q:"UNKNOWN" for q in qs}
    report_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
PY

