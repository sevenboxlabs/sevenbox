from sevenbox_sdk import SevenboxSession, oracle
from warrant_core import RedactionAction, RedactionPolicy


@oracle(redaction=RedactionPolicy(fields={"ssn": RedactionAction.HASH}))
def verify_identity(applicant):
    return {"verified": True}


@oracle
def verify_identity_no_policy(applicant):
    return {"verified": True}


def test_sdk_effect_record_is_redacted_when_declared():
    with SevenboxSession() as session:
        applicant = session.axiom({"ssn": "123-45-6789", "name": "Ana"})
        result = verify_identity(applicant)

    w = session.store.get(result.warrant_id)
    assert w.effect_record.request["args"][0]["ssn"].startswith("sha256:")
    assert w.effect_record.request["args"][0]["name"] == "Ana"
    # content is untouched -- it's the typed claim
    assert w.content == {"verified": True}
    assert w.effect_record.redacted is True


def test_sdk_effect_record_unaffected_without_a_policy():
    with SevenboxSession() as session:
        applicant = session.axiom({"ssn": "123-45-6789"})
        result = verify_identity_no_policy(applicant)

    w = session.store.get(result.warrant_id)
    assert w.effect_record.request["args"][0]["ssn"] == "123-45-6789"
    assert w.effect_record.redacted is False
