from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_OUTPUT = str(SCRIPT_DIR / "result" / "vaka_qa_result.csv")

# 1. 默认 user/account，与 import_to_ov.py 保持一致
DEFAULT_USER_ID = "default"
DEFAULT_AGENT_ID = "default"
DEFAULT_ACCOUNT = "default"
DEFAULT_OPENVIKING_URL = "http://10.37.100.10:1934"

FIELDNAMES = [
    "question_index",
    "question",
    "standard_answer",
    "response",
    "response_without_ref",
    "time_cost",
    "result",
    "reasoning",
]


# 2. 从 CSV 加载 query 和 standard_answer
def load_qa_from_csv(input_path: str, count: int | None = None) -> list[dict]:
    """从 CSV 文件加载 QA 数据，取 query 和 standard_answer 字段"""
    qa_list = []
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = (row.get("query") or row.get("question") or "").strip()
            if not question:
                continue
            qa_list.append(
                {
                    "question": question,
                    "standard_answer": (row.get("standard_answer") or "").strip(),
                }
            )
    if count is not None:
        qa_list = qa_list[:count]
    return qa_list


# 3. 调用 OpenViking /bot/v1/chat 生成回答
async def chat_with_bot(
    question: str,
    *,
    openviking_url: str,
    session_id: str = "default",
    user_id: str | None = None,
    account: str | None = None,
    api_key: str | None = None,
) -> tuple[str, float]:
    """调用 OpenViking /bot/v1/chat 端点生成回答，返回 (回答文本, 耗时秒)"""
    import httpx

    url = f"{openviking_url.rstrip('/')}/bot/v1/chat"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    # 4. trusted mode 需要携带 account 和 user 头
    if account:
        headers["X-OpenViking-Account"] = account
    if user_id:
        headers["X-OpenViking-User"] = user_id

    # 4. 构造 ChatRequest 请求体
    body = {
        "message": question,
        "session_id": session_id,
        "stream": False,
    }
    if user_id:
        body["user_id"] = user_id

    start_time = time.time()
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=body, headers=headers)

    time_cost = time.time() - start_time

    if response.status_code != 200:
        return f"[HTTP ERROR] status={response.status_code}, body={response.text[:200]}", time_cost

    try:
        data = response.json()
        # 5. 解析 ChatResponse，提取 message 字段
        message = data.get("message", "")
        if not message:
            return f"[EMPTY RESPONSE] {json.dumps(data, ensure_ascii=False)[:200]}", time_cost
        return message, time_cost
    except (json.JSONDecodeError, ValueError) as exc:
        return f"[PARSE ERROR] {str(exc)}: {response.text[:200]}", time_cost


# 6. 单个 QA 处理
async def process_single_qa(
    qa_item: dict,
    idx: int,
    total_count: int,
    *,
    openviking_url: str,
    user_id: str | None,
    account: str | None,
    api_key: str | None,
) -> dict:
    """处理单个 QA：调用 /bot/v1/chat 生成回答"""
    question = qa_item["question"]
    standard_answer = qa_item.get("standard_answer", "")
    print(f"Processing {idx}/{total_count}: {question[:60]}...")

    # 7. 每个问题使用独立 session，避免上下文干扰
    session_id = f"vaka_eval_q{idx}"
    response, time_cost = await chat_with_bot(
        question,
        openviking_url=openviking_url,
        session_id=session_id,
        user_id=user_id,
        account=account,
        api_key=api_key,
    )
    print(f"Completed {idx}/{total_count}, time cost: {round(time_cost, 2)}s")

    return {
        "question_index": idx - 1,
        "question": question,
        "standard_answer": standard_answer,
        "response": response,
        "response_without_ref": response,
        "time_cost": round(time_cost, 2),
        "result": "",
        "reasoning": "",
    }


async def run_eval(args: argparse.Namespace) -> None:
    # 8. 加载 QA 数据
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        raise SystemExit(1)

    qa_list = load_qa_from_csv(str(input_path), args.count)
    total = len(qa_list)
    print(f"Loaded {total} question(s) from {input_path}")
    print(f"OpenViking: {args.openviking_url}")
    print(f"User: {args.user_id}")

    # 确保输出目录存在
    output_path = Path(args.output).expanduser()
    os.makedirs(output_path.parent, exist_ok=True)

    # 9. 并发处理所有 QA
    semaphore = asyncio.Semaphore(args.parallel)

    async def limited_process(qa_item: dict, idx: int) -> dict:
        async with semaphore:
            return await process_single_qa(
                qa_item,
                idx,
                total,
                openviking_url=args.openviking_url,
                user_id=args.user_id,
                account=args.account,
                api_key=args.api_key,
            )

    tasks = [limited_process(qa, i) for i, qa in enumerate(qa_list, 1)]
    rows = await asyncio.gather(*tasks)

    # 10. 按 question_index 排序后写文件
    rows.sort(key=lambda r: r.get("question_index", 0))

    if args.update_mode and output_path.exists():
        print(f"Update mode: updating existing file {output_path}")
        with open(output_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fieldnames = list(reader.fieldnames or FIELDNAMES)

        updated_count = 0
        for new_row in rows:
            q_idx = str(new_row.get("question_index", ""))
            found = False
            for row in existing_rows:
                if str(row.get("question_index", "")) == q_idx:
                    row.update(new_row)
                    found = True
                    updated_count += 1
                    break
            if not found:
                existing_rows.append(new_row)
                updated_count += 1

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=existing_fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
        print(f"Updated {updated_count} rows in {output_path}")
    else:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Evaluation completed, results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Vaka QA evaluation using OpenViking /bot/v1/chat API"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(Path("~/Downloads/vaka_judge.csv").expanduser()),
        help="Path to QA CSV file (with query/standard_answer columns)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to output result CSV, default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate, default: all",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Number of concurrent requests, default: 5",
    )
    # OpenViking 配置
    parser.add_argument(
        "--openviking-url",
        default=DEFAULT_OPENVIKING_URL,
        help=f"OpenViking service URL, default: {DEFAULT_OPENVIKING_URL}",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help=f"OpenViking user_id, default: {DEFAULT_USER_ID}",
    )
    parser.add_argument(
        "--account",
        default=DEFAULT_ACCOUNT,
        help=f"OpenViking account, default: {DEFAULT_ACCOUNT}",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="OpenViking API key (X-API-Key header), default: empty",
    )
    parser.add_argument(
        "--update-mode",
        action="store_true",
        help="Update mode: if output file exists, update matching rows instead of overwriting",
    )
    args = parser.parse_args()

    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
