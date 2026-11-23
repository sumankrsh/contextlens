"""Write the synthetic sample traces in examples/.

Run from the project root:  python scripts/make_examples.py

Every name, policy, claim and message here is made up. The traces imitate
three common agent shapes:

* claims-agent.otel.json: a tool-using claims assistant, exported as
  OpenTelemetry GenAI spans. It re-runs a policy search that returns clauses it
  already has, and is sent seven tool definitions while using three.
* support-chat.json: a customer-support chat in contextlens JSON whose framework
  drops the oldest messages in a batch when the history grows past 12 messages,
  so the customer's first message, which says how they want to be contacted,
  falls out of the context.
* research-agent.indexed.otel.json: a small trace in the older indexed
  attribute style (gen_ai.prompt.0.role, ...), to show that format is read too.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# ---------------------------------------------------------------- shared text

CLAUSES = {
    "4.1": (
        "Section 4.1 Escape of water. We will pay for loss or damage to the buildings and contents caused by water "
        "escaping suddenly from a fixed water installation, washing machine, dishwasher, fish tank or heating system. "
        "We will also pay the reasonable cost of finding the source of the leak and repairing the damage caused by "
        "tracing it, up to 5,000 pounds for any one claim."
    ),
    "4.2": (
        "Section 4.2 What is not covered under escape of water. We will not pay for damage caused by water escaping "
        "gradually over a period of time, failure or lack of sealant or grout, water escaping from guttering or "
        "drains, or damage to the installation itself unless the damage is caused by freezing. We will not pay for "
        "any loss while the home is unoccupied for more than 60 days in a row."
    ),
    "4.3": (
        "Section 4.3 Excess for escape of water. The first 450 pounds of every escape of water claim is not covered. "
        "This compulsory excess applies in addition to any voluntary excess shown on the schedule. Where the same "
        "event damages both buildings and contents, only one escape of water excess is applied to the claim."
    ),
    "6.2": (
        "Section 6.2 Alternative accommodation. If the home cannot be lived in following an insured event, we will "
        "pay the reasonable cost of alternative accommodation of a similar standard for the policyholder, their "
        "family and their pets, up to 20 percent of the buildings sum insured, for up to 12 months."
    ),
    "6.3": (
        "Section 6.3 Loss of rent. If the home is let and cannot be lived in following an insured event, we will pay "
        "the rent the tenants would have paid, up to 20 percent of the buildings sum insured, until the home is fit "
        "to live in again."
    ),
    "9.1": (
        "Section 9.1 How we settle claims. We may repair, reinstate or replace damaged property, or pay the cost of "
        "doing so. We will not pay more than the sum insured shown on the schedule. Where items form part of a set "
        "or suite, we will only pay for the damaged part."
    ),
    "2.4": (
        "Section 2.4 Your duty to prevent loss. You must take all reasonable steps to prevent loss or damage and keep "
        "the home in a good state of repair. If you do not, we may reduce or decline a claim to the extent that the "
        "failure caused or increased the loss."
    ),
}


def search_result(query: str, hits: list[tuple[str, float]], variant: dict[str, str] | None = None) -> str:
    variant = variant or {}
    results = [{"clause": cid, "score": score, "text": variant.get(cid, CLAUSES[cid])} for cid, score in hits]
    return json.dumps({"query": query, "policy_wording": "HM-HOME-2026-04", "results": results}, indent=1)


def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


CLAIMS_TOOLS = [
    tool(
        "get_claim",
        "Fetch a claim record by claim number, including the policy number, loss date, loss description, the "
        "customer's estimate and any notes from the first notice of loss call.",
        {"claim_id": {"type": "string", "description": "Claim number, for example CLM-20419."}},
        ["claim_id"],
    ),
    tool(
        "search_policy",
        "Search the policy wording that applies to a policy and return the most relevant clauses with their "
        "section numbers and relevance scores.",
        {
            "policy_id": {"type": "string", "description": "Policy number."},
            "query": {"type": "string", "description": "What to look for, in plain words."},
            "top_k": {"type": "integer", "description": "How many clauses to return.", "default": 4},
        },
        ["policy_id", "query"],
    ),
    tool(
        "calculate_payout",
        "Calculate the settlement for a claim from the assessed loss amounts, the excesses that apply and the "
        "policy limits. Returns a line-by-line breakdown.",
        {
            "claim_id": {"type": "string"},
            "items": {
                "type": "array",
                "description": "Loss items with amounts in pounds.",
                "items": {"type": "object", "properties": {"item": {"type": "string"}, "amount": {"type": "number"}}},
            },
            "excess": {"type": "number", "description": "Total excess in pounds."},
        },
        ["claim_id", "items", "excess"],
    ),
    tool(
        "send_email",
        "Send an email to the customer or a third party. The message is queued for a handler to approve before "
        "it is sent.",
        {
            "to": {"type": "string", "description": "Recipient email address."},
            "subject": {"type": "string"},
            "body": {"type": "string", "description": "Plain text body."},
        },
        ["to", "subject", "body"],
    ),
    tool(
        "escalate_to_human",
        "Pass the claim to a human claims handler with a reason and a priority, when the claim is outside the "
        "assistant's authority or the customer asks for a person.",
        {
            "claim_id": {"type": "string"},
            "reason": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "normal", "urgent"]},
        },
        ["claim_id", "reason"],
    ),
    tool(
        "translate_text",
        "Translate text between English, Welsh, Polish and Urdu for customer letters.",
        {
            "text": {"type": "string"},
            "target_language": {"type": "string", "enum": ["en", "cy", "pl", "ur"]},
        },
        ["text", "target_language"],
    ),
    tool(
        "lookup_weather_history",
        "Return recorded weather for a postcode district and date range, such as rainfall, wind speed and "
        "freezing temperatures, to check storm or freeze claims.",
        {
            "postcode_district": {"type": "string"},
            "start_date": {"type": "string", "format": "date"},
            "end_date": {"type": "string", "format": "date"},
        },
        ["postcode_district", "start_date", "end_date"],
    ),
]

CLAIMS_SYSTEM = (
    "You are the claims assessment assistant for Harbourline Mutual, a fictional home insurer. You help claims "
    "handlers assess new home claims. For each claim, fetch the claim record, check the policy wording that "
    "applies, and recommend whether the loss is covered, which exclusions or excesses apply, and an estimated "
    "settlement. Quote the section numbers you rely on. Never promise a payment to a customer: your "
    "recommendation is reviewed by a handler. If the claim is above 25,000 pounds, involves injury, or the "
    "customer disputes a decision, escalate it to a human. Keep answers short and factual, and say what "
    "information is missing.\n\n"
    "House style: use plain English, pounds with no pence, and dates as 14 March 2026. Do not include the "
    "customer's phone number or email address in recommendations."
)

CLAIM_RECORD = json.dumps(
    {
        "claim_id": "CLM-20419",
        "policy_id": "HM-552190",
        "policyholder": "A. Rowe (synthetic)",
        "loss_date": "2026-11-02",
        "reported": "2026-11-03",
        "cover": ["buildings", "contents"],
        "description": (
            "Burst pipe under the kitchen sink while the customer was at work. Water spread across the kitchen and "
            "hallway floors. Kitchen base units and laminate flooring are swollen. The customer turned off the "
            "stopcock on returning home and a plumber repaired the pipe the same evening."
        ),
        "customer_estimate": {"kitchen units": 3800, "flooring": 2100, "plumber": 240, "drying": 600},
        "notes": "Customer asks whether they can stay with family and claim costs while the floor dries.",
    },
    indent=1,
)


def claims_agent() -> tuple[list[dict], str]:
    """The calls of the claims assistant, as (messages, output) pairs in GenAI semantic convention shape."""
    user = {
        "role": "user",
        "parts": [
            {
                "type": "text",
                "content": "Please assess claim CLM-20419 for cover, excess and an estimated settlement. "
                "The customer also asks about alternative accommodation while the floor dries.",
            }
        ],
    }

    def call(name: str, cid: str, args: dict) -> dict:
        return {"role": "assistant", "parts": [{"type": "tool_call", "id": cid, "name": name, "arguments": args}]}

    def result(cid: str, response: str) -> dict:
        return {"role": "tool", "parts": [{"type": "tool_call_response", "id": cid, "response": response}]}

    turns: list[tuple[dict, dict]] = [
        (call("get_claim", "call_1", {"claim_id": "CLM-20419"}), result("call_1", CLAIM_RECORD)),
        (
            call("search_policy", "call_2", {"policy_id": "HM-552190", "query": "burst pipe escape of water"}),
            result(
                "call_2",
                search_result(
                    "burst pipe escape of water", [("4.1", 0.91), ("4.2", 0.84), ("4.3", 0.77), ("2.4", 0.52)]
                ),
            ),
        ),
        (
            call("search_policy", "call_3", {"policy_id": "HM-552190", "query": "excess for escape of water claims"}),
            result(
                "call_3",
                search_result(
                    "excess for escape of water claims",
                    [("4.3", 0.93), ("4.1", 0.81), ("4.2", 0.70), ("9.1", 0.48)],
                    # The search index holds an older copy of 4.1 with slightly different wording.
                    variant={
                        "4.1": CLAUSES["4.1"]
                        .replace("for any one claim", "for each claim")
                        .replace("reasonable cost", "reasonable costs")
                    },
                ),
            ),
        ),
        (
            call(
                "calculate_payout",
                "call_4",
                {
                    "claim_id": "CLM-20419",
                    "items": [
                        {"item": "kitchen units", "amount": 3800},
                        {"item": "flooring", "amount": 2100},
                        {"item": "trace and access", "amount": 240},
                        {"item": "drying", "amount": 600},
                    ],
                    "excess": 450,
                },
            ),
            result(
                "call_4",
                json.dumps(
                    {
                        "claim_id": "CLM-20419",
                        "lines": [
                            {"item": "kitchen units", "amount": 3800},
                            {"item": "flooring", "amount": 2100},
                            {"item": "trace and access", "amount": 240},
                            {"item": "drying", "amount": 600},
                            {"item": "escape of water excess", "amount": -450},
                        ],
                        "settlement": 6290,
                        "within_limits": True,
                    },
                    indent=1,
                ),
            ),
        ),
        (
            call(
                "search_policy",
                "call_5",
                {"policy_id": "HM-552190", "query": "alternative accommodation while repairs"},
            ),
            result(
                "call_5",
                search_result(
                    "alternative accommodation while repairs",
                    [("6.2", 0.89), ("6.3", 0.71), ("4.1", 0.55), ("4.3", 0.41)],
                ),
            ),
        ),
    ]
    final = (
        "Recommendation for CLM-20419: covered under section 4.1 (sudden escape of water from a fixed installation). "
        "Section 4.2 does not apply: the leak was sudden and the home was occupied. The 450 pound escape of water "
        "excess in section 4.3 applies once. Estimated settlement 6,290 pounds, subject to the handler confirming "
        "the kitchen unit quote. Alternative accommodation is payable under section 6.2 only if the home cannot be "
        "lived in; staying with family while the floor dries would need receipts for any extra costs. Missing: "
        "photos of the damage and the plumber's invoice."
    )

    calls = []
    history: list[dict] = [user]
    for assistant, tool_result in turns:
        calls.append((list(history), [assistant]))
        history += [assistant, tool_result]
    calls.append(
        (list(history), [{"role": "assistant", "parts": [{"type": "text", "content": final}], "finish_reason": "stop"}])
    )
    return calls, final


def _attr(key: str, value) -> dict:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, list):
        return {"key": key, "value": {"arrayValue": {"values": [{"stringValue": v} for v in value]}}}
    return {"key": key, "value": {"stringValue": str(value)}}


def otlp(
    service: str, trace_id: str, root_name: str, llm_spans: list[list[dict]], extra_spans: list[dict] | None = None
) -> dict:
    root_id = "a000000000000001"
    start = 1_793_000_000_000_000_000  # a fixed time in late 2026, in nanoseconds
    spans = [
        {
            "traceId": trace_id,
            "spanId": root_id,
            "name": root_name,
            "kind": 1,
            "startTimeUnixNano": str(start),
            "endTimeUnixNano": str(start + 60_000_000_000),
            "attributes": [_attr("gen_ai.operation.name", "invoke_agent"), _attr("gen_ai.agent.name", root_name)],
        }
    ]
    for i, attrs in enumerate(llm_spans):
        t0 = start + (i + 1) * 5_000_000_000
        spans.append(
            {
                "traceId": trace_id,
                "spanId": f"b{i + 1:015x}",
                "parentSpanId": root_id,
                "name": next((a["value"]["stringValue"] for a in attrs if a["key"] == "span.name"), "chat"),
                "kind": 3,
                "startTimeUnixNano": str(t0),
                "endTimeUnixNano": str(t0 + 2_000_000_000),
                "attributes": [a for a in attrs if a["key"] != "span.name"],
            }
        )
    for i, span in enumerate(extra_spans or []):
        t0 = start + (i + 1) * 5_000_000_000 + 2_500_000_000
        spans.append(
            {
                "traceId": trace_id,
                "spanId": f"c{i + 1:015x}",
                "parentSpanId": root_id,
                "kind": 1,
                "startTimeUnixNano": str(t0),
                "endTimeUnixNano": str(t0 + 400_000_000),
                **span,
            }
        )
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attr("service.name", service)]},
                "scopeSpans": [{"scope": {"name": "synthetic-example"}, "spans": spans}],
            }
        ]
    }


def write_claims_agent() -> None:
    calls, _ = claims_agent()
    system = json.dumps([{"type": "text", "content": CLAIMS_SYSTEM}])
    tools = json.dumps(CLAIMS_TOOLS)
    llm_spans = []
    tool_spans = []
    for messages, output in calls:
        llm_spans.append(
            [
                _attr("span.name", "chat claims-model-mock"),
                _attr("gen_ai.operation.name", "chat"),
                _attr("gen_ai.provider.name", "mock"),
                _attr("gen_ai.request.model", "claims-model-mock"),
                _attr("gen_ai.system_instructions", system),
                _attr("gen_ai.tool.definitions", tools),
                _attr("gen_ai.input.messages", json.dumps(messages)),
                _attr("gen_ai.output.messages", json.dumps(output)),
            ]
        )
        part = output[0]["parts"][0]
        if part["type"] == "tool_call":
            tool_spans.append(
                {
                    "name": f"execute_tool {part['name']}",
                    "attributes": [
                        _attr("gen_ai.operation.name", "execute_tool"),
                        _attr("gen_ai.tool.name", part["name"]),
                    ],
                }
            )
    doc = otlp(
        "claims-assistant", "5c1a1e0000000000000000000000c419", "claims-assistant CLM-20419", llm_spans, tool_spans
    )
    (EXAMPLES / "claims-agent.otel.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- support chat

SUPPORT_SYSTEM = (
    "You are the online help assistant for Harbourline Mutual, a fictional home insurer. Answer questions about "
    "policies, claims and payments using the help articles you are given. Be friendly and brief. If you cannot "
    "help, offer to pass the conversation to a person. Never ask for full card numbers or passwords."
)

FAQ = {
    "excess": (
        "Help article: What is an excess? The excess is the part of a claim you pay yourself. Your schedule shows a "
        "compulsory excess, set by us, and any voluntary excess you chose. Both are taken off the settlement. "
        "Escape of water claims have a separate compulsory excess, shown on your schedule."
    ),
    "payment": (
        "Help article: Changing your payment date. You can move your monthly payment date once in any 12 months, to "
        "any day between the 1st and the 28th. The change applies from the next payment. If the new date is later, "
        "the next payment may include extra days of cover."
    ),
    "claim_status": (
        "Help article: Tracking a claim. Once a claim is registered you receive a claim number by email. You can see "
        "its status, upload photos and invoices, and message your handler in the online account. Most claims are "
        "assessed within ten working days of receiving all documents."
    ),
    "documents": (
        "Help article: What documents do we need? For damage claims, send photos of the damage, repair quotes or "
        "invoices, and receipts for anything you have paid for. For theft, we also need the crime reference number. "
        "Upload documents in your online account or reply to the claim email."
    ),
}


def write_support_chat() -> None:
    first_user = (
        "Hi, I made a claim last week for a burst pipe, claim CLM-20419. I'm travelling for work until 12 December "
        "and can only be reached by email, please don't call me. I have a few questions."
    )
    turns = [
        (
            first_user,
            None,
            "Hello, thanks for getting in touch. I can help with questions about claim CLM-20419. What would you like to know?",
        ),
        (
            "How long will it take to assess?",
            "claim_status",
            "Most claims are assessed within ten working days of receiving all the documents. You can track it in your online account.",
        ),
        (
            "What documents do you still need from me?",
            "documents",
            "For a damage claim we need photos of the damage, repair quotes or invoices, and receipts for anything you have paid. You can upload them in your online account.",
        ),
        (
            "Will I have to pay part of it myself?",
            "excess",
            "Yes. The excess is taken off the settlement. Escape of water claims have their own compulsory excess, shown on your schedule.",
        ),
        (
            "Is the escape of water excess on top of the normal one?",
            "excess",
            "The escape of water excess replaces the standard compulsory excess for that claim, and any voluntary excess you chose still applies.",
        ),
        (
            "Separately, can I move my payment date to the 30th?",
            "payment",
            "You can move it once in 12 months to any day from the 1st to the 28th, so the 30th isn't possible. Would the 28th work?",
        ),
        (
            "OK, the 28th. And can someone update me on the claim when the assessor has been?",
            None,
            "Of course. I'll ask the handler to call you after the assessment.",
        ),
    ]
    # The framework trims the history in a batch when it grows past 12 messages,
    # keeping the most recent turns from a user message onwards.
    history: list[dict] = []
    calls = []
    system = {"role": "system", "content": SUPPORT_SYSTEM}
    for i, (user_text, article, answer) in enumerate(turns):
        if len(history) > 12:
            cut = len(history) - 6
            while history[cut]["role"] != "user":
                cut += 1
            del history[:cut]
        history.append({"role": "user", "content": user_text})
        if article:
            tool_call = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"faq_{i}",
                        "type": "function",
                        "function": {"name": "search_help", "arguments": json.dumps({"query": user_text})},
                    }
                ],
            }
            calls.append({"id": f"turn-{i + 1}-search", "messages": [system, *history], "output": tool_call})
            history.append(tool_call)
            # The help search returns the matching article and, as background, the two before it.
            keys = list(FAQ)
            k = keys.index(article)
            docs = [FAQ[keys[(k - j) % len(keys)]] for j in range(3)]
            history.append({"role": "tool", "tool_call_id": f"faq_{i}", "content": "\n\n".join(docs)})
        calls.append({"id": f"turn-{i + 1}", "messages": [system, *history], "output": {"content": answer}})
        history.append({"role": "assistant", "content": answer})

    for c in calls:
        c["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "search_help",
                    "description": "Search the help articles.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
    doc = {
        "name": "support-chat CLM-20419",
        "model": "chat-model-mock",
        "context_limit": 8192,
        "calls": calls,
    }
    (EXAMPLES / "support-chat.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- indexed attributes


def write_research_agent() -> None:
    system = "You summarise public guidance for an internal AI governance team. Cite the source title for every point."
    question = "What does the guidance say about keeping logs for high-risk AI systems?"
    source = (
        "Source: Synthetic guidance note 7, record keeping. Providers of high-risk systems should make sure the "
        "system can record events automatically over its lifetime. Deployers should keep the logs the system "
        "generates, to the extent they control them, for a period appropriate to the purpose, and for at least six "
        "months unless other law says otherwise."
    )

    def span(
        messages: list[tuple[str, str]], completion: str, tool_calls: list[tuple[str, str]] | None = None
    ) -> list[dict]:
        attrs = [
            _attr("span.name", "openai.chat"),
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.request.model", "research-model-mock"),
            _attr("llm.request.functions.0.name", "fetch_guidance"),
            _attr("llm.request.functions.0.description", "Fetch a guidance note by topic."),
            _attr(
                "llm.request.functions.0.parameters",
                json.dumps({"type": "object", "properties": {"topic": {"type": "string"}}}),
            ),
        ]
        for i, (role, content) in enumerate(messages):
            attrs += [_attr(f"gen_ai.prompt.{i}.role", role), _attr(f"gen_ai.prompt.{i}.content", content)]
        attrs += [_attr("gen_ai.completion.0.role", "assistant"), _attr("gen_ai.completion.0.content", completion)]
        for j, (name, args) in enumerate(tool_calls or []):
            attrs += [
                _attr(f"gen_ai.completion.0.tool_calls.{j}.name", name),
                _attr(f"gen_ai.completion.0.tool_calls.{j}.arguments", args),
            ]
        return attrs

    answer = "Deployers should keep the logs they control for at least six months (Synthetic guidance note 7)."
    spans = [
        span([("system", system), ("user", question)], "", [("fetch_guidance", '{"topic": "record keeping"}')]),
        span(
            [("system", system), ("user", question), ("tool", source)],
            "",
            [("fetch_guidance", '{"topic": "logs retention"}')],
        ),
        span([("system", system), ("user", question), ("tool", source), ("tool", source)], answer),
    ]
    doc = otlp("research-agent", "7e5ea2c4000000000000000000000007", "research-agent", spans)
    (EXAMPLES / "research-agent.indexed.otel.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    EXAMPLES.mkdir(exist_ok=True)
    write_claims_agent()
    write_support_chat()
    write_research_agent()
    print(
        "wrote examples/claims-agent.otel.json, examples/support-chat.json, examples/research-agent.indexed.otel.json"
    )


if __name__ == "__main__":
    main()
