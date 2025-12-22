import os
import json
from openai import OpenAI
from shared.schemas import TaskSpec, Report, Grade
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

GRADE_PROMPT = """
你是一个公正的 AI 评估员。请根据以下信息判断 Agent 是否完成了任务。

【任务目标】
{prompt}

【标准答案 / 预期结果】
{expected_answers}

【Agent 提交的答案】
{agent_answers}

【Agent 执行轨迹】
{trajectory}

请注意：
1. 不要纠结于格式（如 "10.5" 和 "10.50" 视为相同）。
2. 如果 Agent 的执行轨迹显示它通过错误的逻辑蒙对了答案，请判失败。
3. 如果答案正确且逻辑合理，判为通过。

请以 JSON 格式返回：
{{
    "passed": true/false,
    "score": 0.0 到 1.0,
    "feedback": "简短的评价，说明为什么给这个分"
}}
"""

def grade_with_llm(task: TaskSpec, report: Report, gold_answer: str) -> Grade:
    # 构造 Prompt
    content = GRADE_PROMPT.format(
        prompt=task.prompt,
        expected_answers=str(gold_answer),  
        agent_answers=str(report.answers),
        trajectory="\n".join(report.trajectory[-10:]) 
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"}
        )
      
        result = json.loads(response.choices[0].message.content)
        
        return Grade(
            task_id=report.task_id,
            passed=result["passed"],
            score=result["score"],
            feedback=result["feedback"],
            details=result
        )
    except Exception as e:
        print(f"Grading failed: {e}")
        return Grade(task_id=report.task_id, passed=False, score=0.0, feedback="LLM Grading Error")
