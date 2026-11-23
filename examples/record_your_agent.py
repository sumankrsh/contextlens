"""Record a small agent loop with contextlens.recorder, then analyse it.

Run:  python examples/record_your_agent.py

The "model" here is a stand-in function, so this runs offline. In your own code,
call rec.record(...) right after each real chat-completions request, passing the
same messages and tools you sent.
"""

from __future__ import annotations

import json

from contextlens.analysis import analyse
from contextlens.loaders import load
from contextlens.recorder import Recorder

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Fetch an order by id.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund",
            "description": "Refund an order. Needs approval.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
        },
    },
]


def fake_model(messages: list[dict]) -> dict:
    """Ask for the order once, then answer."""
    if not any(m["role"] == "tool" for m in messages):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_order", "arguments": json.dumps({"order_id": "ORD-1001"})},
                }
            ],
        }
    return {"role": "assistant", "content": "Order ORD-1001 shipped on 18 November and should arrive within two days."}


def main() -> None:
    rec = Recorder("order-helper example", context_limit=8000)
    messages = [
        {"role": "system", "content": "You answer questions about orders for a fictional shop. Be brief."},
        {"role": "user", "content": "Where is my order ORD-1001?"},
    ]
    while True:
        reply = fake_model(messages)
        rec.record(messages, tools=TOOLS, output=reply, model="fake-model")
        messages.append(reply)
        if not reply.get("tool_calls"):
            break
        order = {"order_id": "ORD-1001", "status": "shipped", "shipped": "2026-11-18", "carrier": "synthetic post"}
        messages.append({"role": "tool", "tool_call_id": "c1", "content": json.dumps(order)})

    path = rec.save("order-helper.trace.json")
    a = analyse(load(path))
    print(f"wrote {path} with {len(a.calls)} calls; wasted {a.wasted_tokens:.0f} of {a.total_tokens:.0f} tokens")
    for f in a.findings:
        print(f"  {f.severity}: {f.rule}: {f.title}")


if __name__ == "__main__":
    main()
