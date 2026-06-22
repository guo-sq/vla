import json
import os
import urllib.request


def write_output(name: str, value: str) -> None:
    output_path = os.environ["GITHUB_OUTPUT"]
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}<<EOF\n{value}\nEOF\n")


def read_event_payload() -> dict:
    with open(os.environ["GITHUB_EVENT_PATH"], "r", encoding="utf-8") as handle:
        return json.load(handle)


def request(url: str, token: str, accept: str = "application/vnd.github+json"):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        content_type = response.headers.get_content_type()
        body = response.read()
    if content_type == "application/json":
        return json.loads(body.decode("utf-8"))
    return body.decode("utf-8")


def call_glm_request(api_key: str, prompt: str, diff: str, *, diff_limit: int, max_tokens: int) -> dict:
    payload = {
        "model": "glm-5",
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "system", "content": "你是专业的 GitHub Pull Request 代码审查助手。"},
            {"role": "user", "content": prompt + "\n\n```diff\n" + diff[:diff_limit] + "\n```"},
        ],
    }
    req = urllib.request.Request(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def extract_glm_review(body: dict) -> str:
    choices = body.get("choices") or [{}]
    choice = choices[0] if choices else {}
    message = choice.get("message", {})
    content = (message.get("content") or "").strip()
    if content:
        return content

    error_message = body.get("error", {}).get("message")
    if error_message:
        return str(error_message)

    finish_reason = choice.get("finish_reason") or "unknown"
    usage = body.get("usage") or {}
    reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
    if reasoning_tokens is None:
        return f"GLM 返回空内容，finish_reason={finish_reason}。"
    return f"GLM 返回空内容，finish_reason={finish_reason}，reasoning_tokens={reasoning_tokens}。"


def call_glm(api_key: str, diff: str) -> str:
    prompt = """你是一位资深的 Pull Request 代码审查专家。请基于下面的 diff 输出中文审查意见，重点关注：
1. 正确性和潜在 bug
2. 行为回归风险
3. 性能和资源使用
4. 安全和敏感信息泄露
5. 缺失的测试或验证

请优先给出高风险问题；如果没有明显问题，请明确写出“未发现阻塞性问题”，再补充改进建议。"""
    body = call_glm_request(api_key, prompt, diff, diff_limit=15000, max_tokens=2048)
    review = (body.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    if review:
        return review

    retry_prompt = """请直接输出最终代码审查结论，不要输出思考过程或分析过程。
输出格式要求：
1. 优先列出阻塞性问题，最多 5 条
2. 每条包含：标题、风险、建议
3. 如果没有阻塞性问题，只输出“未发现阻塞性问题”，然后补充最多 3 条改进建议
4. 请使用中文，保持简洁"""
    retry_body = call_glm_request(api_key, retry_prompt, diff, diff_limit=8000, max_tokens=1024)
    retry_review = (retry_body.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    if retry_review:
        return retry_review

    return extract_glm_review(retry_body)


def main() -> None:
    payload = read_event_payload()
    event_name = os.environ["GITHUB_EVENT_NAME"]
    token = os.environ.get("GH_TOKEN", "")
    api_key = os.environ.get("ZHIPU_API_KEY", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    if event_name == "workflow_dispatch":
        pr_number = str(payload.get("inputs", {}).get("pr_number", "") or "")
    else:
        pr_number = str(payload.get("pull_request", {}).get("number", "") or "")

    if not pr_number:
        raise SystemExit("Failed to resolve pull request number.")

    write_output("pr_number", pr_number)

    if not api_key:
        write_output("review", "未配置 ZHIPU_API_KEY secret，跳过 GLM Code Review。")
        return

    if not token or not repo:
        write_output("review", "缺少 GitHub API 上下文，无法拉取 Pull Request diff。")
        return

    try:
        diff = request(f"https://api.github.com/repos/{repo}/pulls/{pr_number}", token, "application/vnd.github.v3.diff")
    except Exception as exc:  # noqa: BLE001
        write_output("review", f"拉取 Pull Request diff 失败：{exc}")
        return

    if not str(diff).strip():
        write_output("review", "无代码变更。")
        return

    try:
        review = call_glm(api_key, str(diff))
    except Exception as exc:  # noqa: BLE001
        review = f"GLM API 调用失败: {exc}"

    write_output("review", review)


if __name__ == "__main__":
    main()