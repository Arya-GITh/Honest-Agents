"""
Example 01: Basic Tool Auditing with @audit_tool and HonestyAuditor
-------------------------------------------------------------------
Demonstrates wrapping Python functions with @audit_tool, soft-failure detection,
and capturing cryptographically signed HMAC execution receipts.
"""

from agent_honesty import audit_tool, HonestyAuditor, VerificationRouter


# 1. Wrap your tool with @audit_tool
@audit_tool(name="fetch_user_profile")
def fetch_user_profile(user_id: str):
    if user_id == "usr_missing":
        # Simulating an API soft-error (HTTP 200 with error body)
        return {"status": "error", "error_code": "NOT_FOUND", "message": f"User {user_id} not found."}
    
    return {
        "status": "success",
        "user_id": user_id,
        "name": "Alice Smith",
        "role": "Admin",
    }


def main():
    router = VerificationRouter()

    # 2. Run under HonestyAuditor supervision
    with HonestyAuditor() as auditor:
        print("🔹 Executing tool call with user_id='usr_missing'...")
        result = fetch_user_profile("usr_missing")
        print("Result:", result)

        # Inspect generated HMAC receipt
        receipt = auditor.receipts[-1]
        print(f"\n🔐 HMAC Receipt: ID={receipt.receipt_id}")
        print(f"   Signature Valid: {receipt.verify()}")
        print(f"   FactMatrix: is_error={receipt.fact_matrix.is_error}, status={receipt.fact_matrix.status_code}")

        # 3. Verify a deceptive claim
        deceptive_claim = "Successfully retrieved Alice's profile and confirmed admin role."
        verdict = router.verify(
            user_prompt="Look up profile for usr_missing",
            agent_claim=deceptive_claim,
            receipts=auditor.receipts,
        )
        print(f"\n🛡️ Verification of Deceptive Claim: is_honest={verdict.is_honest}, type={verdict.deception_type}")
        print(f"   Explanation: {verdict.explanation}")

        # 4. Verify an honest claim
        honest_claim = "The user usr_missing was not found in the system."
        verdict_honest = router.verify(
            user_prompt="Look up profile for usr_missing",
            agent_claim=honest_claim,
            receipts=auditor.receipts,
        )
        print(f"\n🛡️ Verification of Honest Claim: is_honest={verdict_honest.is_honest}, type={verdict_honest.deception_type}")


if __name__ == "__main__":
    main()
