import pytest
from unittest.mock import MagicMock

from agent_honesty.receipts import (
    FactMatrix,
    PayloadNormalizer,
    resolve_keypath,
    HMACReceipt,
)
from agent_honesty import audit_tool, HonestyAuditor


def test_resolve_keypath():
    data = {
        "user": {
            "profile": {
                "name": "Alice",
                "emails": ["alice@example.com", "work@example.com"]
            }
        },
        "orders": [
            {"id": 101, "total": 45.50},
            {"id": 102, "total": 99.00}
        ]
    }
    assert resolve_keypath(data, "user.profile.name") == "Alice"
    assert resolve_keypath(data, "user.profile.emails[1]") == "work@example.com"
    assert resolve_keypath(data, "orders[0].id") == 101
    assert resolve_keypath(data, "orders[5].id") is None
    assert resolve_keypath(data, "nonexistent.field") is None


def test_normalizer_standard_success():
    normalizer = PayloadNormalizer()
    payload = {"status": "ok", "rows_affected": 3, "data": [{"id": 1}, {"id": 2}]}
    fm = normalizer.normalize(payload)

    assert fm.is_error is False
    assert fm.status_code == 200
    assert fm.records_mutated == 3
    assert fm.is_empty is False
    assert "status" in fm.data_keys
    assert fm.payload_sha256 != ""


def test_normalizer_soft_failure_detection():
    normalizer = PayloadNormalizer()

    # Case 1: Status = error
    fm1 = normalizer.normalize({"status": "error", "message": "Unauthorized access"})
    assert fm1.is_error is True
    assert fm1.error_type == "SoftError"
    assert "Unauthorized access" in fm1.error_message

    # Case 2: success = False
    fm2 = normalizer.normalize({"success": False, "error": "Insufficient funds"})
    assert fm2.is_error is True
    assert fm2.error_type == "SoftError"
    assert "Insufficient funds" in fm2.error_message

    # Case 3: Error list present
    fm3 = normalizer.normalize({"data": None, "errors": ["Invalid token"]})
    assert fm3.is_error is True
    assert fm3.error_type == "SoftError"


def test_normalizer_custom_keypath():
    normalizer = PayloadNormalizer(custom_error_keypaths=["response.internal_err_code"])
    payload = {
        "status": 200,
        "response": {
            "internal_err_code": "ERR_DATABASE_LOCKED"
        }
    }
    fm = normalizer.normalize(payload)
    assert fm.is_error is True
    assert "ERR_DATABASE_LOCKED" in fm.error_message


def test_normalizer_http_response_mock():
    normalizer = PayloadNormalizer()
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable"
    mock_resp.json.side_effect = Exception("Not JSON")

    fm = normalizer.normalize(mock_resp)
    assert fm.is_error is True
    assert fm.status_code == 503
    assert fm.error_type == "HTTP_503_Error"


def test_hmac_receipt_signature_and_tamper_detection():
    normalizer = PayloadNormalizer()
    fm = normalizer.normalize({"status": "success", "count": 10})

    receipt = HMACReceipt(
        execution_id="exec-123",
        tool_name="test_tool",
        timestamp="2026-08-14T12:00:00Z",
        duration_ms=15.5,
        args_hash=FactMatrix.compute_sha256([]),
        kwargs_hash=FactMatrix.compute_sha256({}),
        fact_matrix=fm,
    )
    receipt.sign(secret_key="my-secret-key")
    assert receipt.signature != ""
    assert receipt.verify(secret_key="my-secret-key") is True

    # Verification should fail with wrong key
    assert receipt.verify(secret_key="wrong-secret-key") is False

    # Tampering test: modify fact_matrix
    receipt.fact_matrix.records_mutated = 999
    assert receipt.verify(secret_key="my-secret-key") is False


def test_audit_tool_generates_verified_receipts():
    @audit_tool(name="soft_failing_api")
    def fetch_records():
        # Soft failure: returns 200-style payload with error flag
        return {"status": "error", "message": "Rate limit reached on upstream vendor"}

    with HonestyAuditor() as auditor:
        res = fetch_records()
        assert res["status"] == "error"

        assert len(auditor.receipts) == 1
        receipt = auditor.receipts[0]

        assert receipt.tool_name == "soft_failing_api"
        assert receipt.verify() is True
        assert receipt.fact_matrix.is_error is True
        assert "Rate limit reached" in receipt.fact_matrix.error_message

        assert len(auditor.fact_matrices) == 1
        assert auditor.fact_matrices[0].is_error is True
