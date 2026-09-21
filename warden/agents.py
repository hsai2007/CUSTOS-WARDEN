"""M7 (build last): two LangGraph agents that write to a WardenGateway
through its public API only (gateway.submit). One agent establishes a fact
about a shared key; the other later submits a claim that flatly contradicts
it (no supersession language) -- WARDEN must visibly block that write.

No metric in results/ is derived from anything in this module (see
test_m7_agents.py::test_no_results_module_imports_agents and DECISIONS.md).
This module exists to demonstrate the gateway API end-to-end, not to
produce evaluation numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, StateGraph

from warden.gateway import WardenGateway, SubmitResult

SHARED_KEY = "agents::vendor_contract_ship_date"


@dataclass
class AgentWriteLog:
    agent: str
    key: str
    claim: str
    predicted_class: str
    action: str
    escalated: bool


class AgentState(TypedDict):
    agent: str
    key: str
    claim: str
    result: SubmitResult | None


def _make_submit_node(gateway: WardenGateway):
    def submit_node(state: AgentState) -> AgentState:
        result = gateway.submit(state["key"], state["claim"])
        return {**state, "result": result}
    return submit_node


def make_agent_graph(gateway: WardenGateway):
    graph = StateGraph(AgentState)
    graph.add_node("submit", _make_submit_node(gateway))
    graph.set_entry_point("submit")
    graph.add_edge("submit", END)
    return graph.compile()


def run_demo(gateway: WardenGateway | None = None) -> list[AgentWriteLog]:
    """Two agents, agent_alpha and agent_beta, write to the same gateway.

    agent_alpha establishes and refines the vendor contract's ship date.
    agent_beta twice submits a flatly conflicting date with no supersession
    language -- the fast-path fact gate must classify each as a
    contradiction and the gate must block it, visibly, in the returned log.
    agent_alpha's later genuine, marker-bearing date change is escalated
    and applied instead of being blocked.
    """
    gateway = gateway or WardenGateway()
    graph = make_agent_graph(gateway)
    logs: list[AgentWriteLog] = []

    turns = [
        ("agent_alpha", SHARED_KEY, "The vendor contract renewal ships on Friday."),
        ("agent_alpha", SHARED_KEY, "The vendor contract renewal ships on Friday, pending final legal sign-off."),
        ("agent_beta", SHARED_KEY, "The vendor contract renewal ships on Monday."),
        ("agent_alpha", SHARED_KEY, "The vendor contract renewal has been moved to Wednesday."),
        ("agent_beta", SHARED_KEY, "The vendor contract renewal ships on Monday."),
    ]

    for agent, key, claim in turns:
        final_state = graph.invoke({"agent": agent, "key": key, "claim": claim, "result": None})
        result: SubmitResult = final_state["result"]
        logs.append(AgentWriteLog(
            agent=agent, key=key, claim=claim,
            predicted_class=result.predicted_class, action=result.action, escalated=result.escalated,
        ))

    return logs
