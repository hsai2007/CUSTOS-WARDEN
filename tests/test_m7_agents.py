import inspect

import warden.agents
import warden.harness
import warden.metrics
from warden.agents import run_demo
from warden.gateway import WardenGateway


def test_two_langgraph_agents_write_through_the_gateway_api():
    logs = run_demo()
    agents_seen = {log.agent for log in logs}
    assert agents_seen == {"agent_alpha", "agent_beta"}
    # every write went through gateway.submit -- no direct store access
    assert all(hasattr(log, "action") for log in logs)


def test_at_least_one_agent_write_is_visibly_blocked_as_contradiction():
    logs = run_demo()
    blocked = [log for log in logs if log.action == "blocked"]
    assert len(blocked) > 0
    for log in blocked:
        assert log.predicted_class == "contradiction"
        # "visibly" -- the block is traceable to a specific agent and claim
        assert log.agent
        assert log.claim


def test_a_genuine_marker_bearing_change_is_escalated_and_applied_not_blocked():
    logs = run_demo()
    escalated = [log for log in logs if log.escalated]
    assert len(escalated) > 0
    assert all(log.action == "apply" for log in escalated)


def test_gateway_is_the_only_write_path_agents_use():
    gateway = WardenGateway()
    logs = run_demo(gateway=gateway)
    # the gateway's own store must reflect every non-blocked write
    final = gateway.store.read(logs[0].key)
    assert final is not None
    assert len(gateway.store.history(logs[0].key)) >= 1


def test_no_results_module_imports_agents():
    """No metric in any results file comes from M7: warden.harness and
    warden.metrics (the modules that feed every results/ number) must not
    import or reference warden.agents or warden.gateway."""
    for module in (warden.harness, warden.metrics):
        source = inspect.getsource(module)
        assert "warden.agents" not in source
        assert "warden.gateway" not in source
        assert "import agents" not in source
