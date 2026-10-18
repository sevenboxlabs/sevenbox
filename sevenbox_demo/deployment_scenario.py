"""
sevenbox_demo.deployment_scenario -- a fifth scenario, in a domain distinct
from refund/kyc/security-incident/procurement, wired into the same
server/visualization layer. Same public-API-only posture: only sevenbox_sdk
and warrant_core's public surface.

Exercises two vocabulary items no scenario before this one had ever used
end to end through the real server: `@human` (every prior scenario only
used oracle/tool/derive) and `@compensable` (every prior irreversible-ish
action used `@irreversible`; nothing had ever shown the Operations Center
a COMPENSABLE reversibility class, which the compensation algebra treats
differently -- see warrant_core.compensation's module docstring). Also a
deeper causal chain (five steps before the correction) than any prior
scenario, for a genuinely non-trivial Evidence Chain/average_causal_depth.
"""
from __future__ import annotations

from typing import Dict

from sevenbox_sdk import SevenboxSession, compensable, derive, human, oracle, tool
from warrant_core import (
    EffectRecord,
    EpistemicClass,
    ReversibilityClass,
    SourceKind,
    WarrantStore,
    supersede,
)

_call_counts: Dict[str, int] = {"infer_deployment_risk": 0, "run_ci_checks": 0}


@oracle
def infer_deployment_risk(deploy_request):
    _call_counts["infer_deployment_risk"] += 1
    return {"risk_level": "moderate", "reason": "touches payment service dependency"}


@tool
def run_ci_checks(risk):
    _call_counts["run_ci_checks"] += 1
    return {"build": "passed", "tests_passed": 42, "tests_failed": 0}


@human
def manual_approval(ci_results):
    return {"approved": True, "approver": "jordan.k", "note": "looks good, ship it"}


# COMPENSABLE, not IRREVERSIBLE: a production deploy can be rolled back --
# unlike a refund already sent or a purchase order already issued in the
# other new scenarios. The compensation algebra (warrant_core.compensation)
# treats this differently from IRREVERSIBLE; no scenario before this one
# had ever exercised that distinction against the real Operations Center.
@compensable
@tool
def deploy_to_production(approval):
    if not approval["approved"]:
        return None
    return {"action": "deployed", "service": "checkout-api", "version": "v2.14.0"}


@derive
def post_deploy_health_check(deployment):
    return {"status": "healthy", "error_rate": "0.02%"}


def _run(session: SevenboxSession) -> None:
    deploy_request = session.axiom({
        "service": "checkout-api", "version": "v2.14.0", "target_env": "production",
    })
    risk = infer_deployment_risk(deploy_request)
    ci_results = run_ci_checks(risk)
    approval = manual_approval(ci_results)
    deployment = deploy_to_production(approval)
    post_deploy_health_check(deployment)

    # CI results turn out to have missed a regression, discovered after
    # the deploy already went out -- targets an ancestor of the
    # compensable deploy action, the same transitive-correction shape
    # every other scenario in this project uses.
    supersede(
        session.store, target_id=ci_results.warrant_id,
        content={"claim": "a flaky test masked a real regression in the checkout flow"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(
            request="post-incident CI re-run", response="regression confirmed"
        ),
    )


def run_live() -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["infer_deployment_risk"] = 0
    _call_counts["run_ci_checks"] = 0
    session = SevenboxSession()
    with session:
        _run(session)
    return session, dict(_call_counts)


def run_replay(reference_store: WarrantStore) -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["infer_deployment_risk"] = 0
    _call_counts["run_ci_checks"] = 0
    session = SevenboxSession(replay_from=reference_store)
    with session:
        _run(session)
    return session, dict(_call_counts)
