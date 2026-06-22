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


def request_json(url: str, token: str) -> dict | list:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def call_glm(api_key: str, prompt: str) -> str:
    payload = {
        "model": "glm-5",
        "temperature": 0.3,
        "max_tokens": 2048,
        "messages": [
            {
                "role": "system",
                "content": "你是一个 GitHub 仓库助手。用户会通过评论 @glm 请求帮助。请基于仓库上下文用中文给出具体、可执行的建议。",
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.load(response)
    return body.get("choices", [{}])[0].get("message", {}).get("content") or body.get("error", {}).get("message") or "GLM API 调用失败，请检查配置。"


def main() -> None:
    payload = read_event_payload()
    event_name = os.environ["GITHUB_EVENT_NAME"]
    gh_token = os.environ.get("GH_TOKEN", "")
    api_key = os.environ.get("ZHIPU_API_KEY", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    prompt = ""
    pr_number = ""

    if event_name == "workflow_dispatch":
        inputs = payload.get("inputs", {})
        prompt = inputs.get("prompt", "")
        pr_number = str(inputs.get("pr_number", "") or "")
    elif event_name in {"issue_comment", "pull_request_review_comment"}:
        prompt = payload.get("comment", {}).get("body", "")
        pr_number = str(payload.get("issue", {}).get("number", "") or payload.get("pull_request", {}).get("number", ""))
    elif event_name == "pull_request_review":
        prompt = payload.get("review", {}).get("body", "")
        pr_number = str(payload.get("pull_request", {}).get("number", ""))
    elif event_name == "issues":
        issue = payload.get("issue", {})
        prompt = ((issue.get("title", "") + "\n\n" + (issue.get("body") or "")).strip())
        pr_number = str(issue.get("number", ""))

    prompt = " ".join(part for part in prompt.replace("@glm", " ").split())
    if not prompt:
        prompt = "请结合当前 Issue 或 Pull Request 上下文提供帮助。"

    if pr_number and gh_token and repo:
        try:
            pull = request_json(f"https://api.github.com/repos/{repo}/pulls/{pr_number}", gh_token)
            files = request_json(f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files?per_page=100", gh_token)
            pr_info = {
                "title": pull.get("title", ""),
                "body": pull.get("body", ""),
                "files": [item.get("filename", "") for item in files],
            }
            prompt = f"当前 PR 信息：\n{json.dumps(pr_info, ensure_ascii=False)}\n\n用户请求：{prompt}"
        except Exception:
            pass

    if not api_key:
        reply = "未配置 ZHIPU_API_KEY secret，无法调用 GLM Assistant。"
    else:
        try:
            reply = call_glm(api_key, prompt)
        except Exception as exc:  # noqa: BLE001
            reply = f"GLM API 调用失败，请检查配置。错误信息：{exc}"

    write_output("pr_number", pr_number)
    write_output("reply", reply)


if __name__ == "__main__":
    main()