# purple_agent/llm_solver.py
import json, os, re
from typing import List, Dict
from openai import OpenAI

def _extract_json(text: str) -> Dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    m = re.search(r"(\{(?:[^{}]|(?1))*\})", text, flags=re.S)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    return {}

def solve_with_openai(prompt: str, questions: List[str], model: str = "gpt-4o-mini") -> List[Dict]:
    # Make sure you’re on a recent client:
    #   pip install -U "openai>=1.42"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sys_msg = "Reply with a pure JSON object; no Markdown fences."
    user = f"Task:\n{prompt}\n\nReturn a JSON object with these exact keys:\n{json.dumps(questions, indent=2)}\n"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys_msg},{"role":"user","content":user}],
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    obj = _extract_json(raw)
    out: List[Dict] = []
    for i, q in enumerate(questions, 1):
        out.append({"id": f"q{i}", "answer": str(obj.get(q, "UNKNOWN"))})
    return out
