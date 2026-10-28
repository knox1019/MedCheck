"""模型连通性探测。

换后端、换模型、加 API key 之后先跑这个，别一上来就跑完整流程——
报错信息在这里最干净，混在检索日志里很难看出来。

    uv run python -m scripts.probe_llm
    uv run python -m scripts.probe_llm --question "华法林的禁忌是什么" --show-reasoning
    uv run python -m scripts.probe_llm --model deepseek-v4-pro --effort max

它会告诉你四件事：
    1. 端点通不通、鉴权过不过
    2. 这个模型到底有没有返回思考过程
    3. 思考花了多少 token（推理模型的成本大头在这里）
    4. 一次调用要多久
"""

import argparse
import sys
import time

from app.core.config import get_settings
from app.llm.client import ChatClient


def mask_key(key: str) -> str:
    """密钥只露前 6 位，避免截图分享时泄露。"""
    if not key:
        return "(空)"
    if len(key) <= 10:
        return key[:2] + "***"
    return key[:6] + "***" + key[-2:]


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="探测大模型连通性与思考过程")
    parser.add_argument(
        "--question",
        default="维生素 C 和他汀类药物能一起吃吗？请只回答一句。",
        help="探测用的提问",
    )
    parser.add_argument("--model", default=None, help="覆盖配置里的模型名")
    parser.add_argument("--base-url", default=None, help="覆盖配置里的端点")
    parser.add_argument("--effort", default=None, help="思考强度：low / high / max")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="显式关闭思考模式（省时间省费用）",
    )
    parser.add_argument("--show-reasoning", action="store_true", help="完整打印思考过程")
    args = parser.parse_args()

    thinking = "disabled" if args.no_thinking else None
    client = ChatClient(
        base_url=args.base_url,
        model=args.model,
        reasoning_effort=args.effort,
        thinking=thinking,
    )

    print("=" * 68)
    print(f"端点      {client.base_url}")
    print(f"模型      {client.model}")
    print(f"密钥      {mask_key(client.api_key)}")
    # 这一行说的是「本次请求有没有主动发思考参数」，不是「模型会不会思考」。
    # 两者是两回事：有的网关默认就返回思维链，即使你没传任何参数。
    if client.thinking_active:
        effort = f"（effort={client.reasoning_effort}）" if client.reasoning_effort else ""
        print(f"思考参数  已发送{effort}")
    else:
        print("思考参数  未发送，走服务端默认")
    print(f"温度      {'不传（思考参数已开）' if client.thinking_active else client.temperature}")
    print(f"超时      {client.timeout:.0f}s")
    print("=" * 68)

    messages = [
        {"role": "system", "content": "你是简洁的助手，回答不超过 80 字。"},
        {"role": "user", "content": args.question},
    ]

    started = time.perf_counter()
    try:
        reply = client.complete(messages)
    except Exception as exc:  # noqa: BLE001 —— 探测脚本就是要原样展示错误
        elapsed = time.perf_counter() - started
        print(f"\n❌ 调用失败（耗时 {elapsed:.1f}s）")
        print(f"   {type(exc).__name__}: {exc}")
        print("\n常见原因：")
        print("  · 401 → API key 不对，或 .env 里 LLM_API_KEY 没填")
        print("  · 404 → base_url 或模型名写错。DeepSeek 的端点不带 /v1")
        print("  · 402 → 余额不足")
        print("  · 超时 → 思考模式较慢，调大 LLM_TIMEOUT")
        return 1

    elapsed = time.perf_counter() - started
    usage = reply.usage or {}

    print(f"\n✅ 调用成功，耗时 {elapsed:.1f}s")
    print(f"   答案 {len(reply.content)} 字")
    if reply.has_reasoning:
        note = "" if client.thinking_active else "（没发任何思考参数，说明该模型默认就返回思维链）"
        print(f"   思考 {len(reply.reasoning)} 字，思考 token {reply.reasoning_tokens}{note}")
    else:
        print("   思考 —— 没有 reasoning_content（非推理模型，或该请求未开启思考）")

    if usage:
        prompt_tokens = usage.get("prompt_tokens", "?")
        completion_tokens = usage.get("completion_tokens", "?")
        total_tokens = usage.get("total_tokens", "?")
        print(
            f"   token：输入 {prompt_tokens} / 输出 {completion_tokens} / 合计 {total_tokens}"
        )
        reasoning_tokens = reply.reasoning_tokens
        if reasoning_tokens and isinstance(completion_tokens, int) and completion_tokens:
            share = reasoning_tokens / completion_tokens * 100
            print(f"   其中思考占输出的 {share:.0f}% —— 这就是推理模型的成本大头")

    if reply.has_reasoning:
        print("\n" + "-" * 68)
        print("【思考过程】")
        if args.show_reasoning:
            print(reply.reasoning)
        else:
            head = reply.reasoning[:300]
            print(head + ("…" if len(reply.reasoning) > 300 else ""))
            print("\n（完整内容加 --show-reasoning）")

    print("\n" + "-" * 68)
    print("【答案】")
    print(reply.content)

    return 0


if __name__ == "__main__":
    sys.exit(main())
