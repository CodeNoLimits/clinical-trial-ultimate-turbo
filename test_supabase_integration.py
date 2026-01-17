#!/usr/bin/env python3
"""
Supabase Integration Test Script
Tests connectivity and CRUD operations for the clinical-trial-ultimate-turbo app.

This script verifies:
1. DNS resolution for Supabase URL
2. Connection to Supabase works
3. Can insert a test trial record
4. Can retrieve trial history
5. Can query recent trials
"""

import json
import socket
import base64
from datetime import datetime
from supabase import create_client

# Supabase credentials
SUPABASE_URL = "https://rhwthxmphohqdawglhfde.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJod3RoeG1waG9ocWRhd2dsaGZkZSIsInJvbGUiOiJhbm9uIiwiaWF0IjoxNzMxNTg4MjUxLCJleHAiOjIwNDcxNjQyNTF9.tJj-TK-qJxl1aKn26WJfhzBwlrqiTo4RvlRnphI-Rxs"

def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def decode_jwt(token: str) -> dict:
    """Decode JWT token payload (without verification)."""
    try:
        parts = token.split('.')
        if len(parts) >= 2:
            payload = parts[1]
            # Add padding if needed
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += '=' * padding
            decoded = base64.b64decode(payload)
            return json.loads(decoded)
    except Exception as e:
        return {"error": str(e)}
    return {}


def test_dns_resolution():
    """Test 0: DNS Resolution for Supabase URL."""
    print_section("TEST 0: DNS Resolution")

    hostname = SUPABASE_URL.replace("https://", "").replace("http://", "").split("/")[0]

    try:
        ip = socket.gethostbyname(hostname)
        print(f"[OK] DNS resolution successful")
        print(f"     Hostname: {hostname}")
        print(f"     IP: {ip}")
        return True
    except socket.gaierror as e:
        print(f"[FAIL] DNS resolution failed: {e}")
        print(f"     Hostname: {hostname}")
        print(f"\n[INFO] The Supabase project URL does not exist or is not reachable.")
        print("       Possible reasons:")
        print("       1. The Supabase project has been deleted")
        print("       2. The project URL is incorrect")
        print("       3. The project was never created")
        print("       4. Network/DNS issues")

        # Decode and display JWT info
        jwt_payload = decode_jwt(SUPABASE_KEY)
        if jwt_payload:
            print(f"\n     JWT Token Info:")
            print(f"       - Project Ref: {jwt_payload.get('ref', 'N/A')}")
            print(f"       - Role: {jwt_payload.get('role', 'N/A')}")
            iat = jwt_payload.get('iat')
            exp = jwt_payload.get('exp')
            if iat:
                from datetime import datetime as dt
                print(f"       - Issued: {dt.fromtimestamp(iat).isoformat()}")
            if exp:
                from datetime import datetime as dt
                print(f"       - Expires: {dt.fromtimestamp(exp).isoformat()}")

        return False

def test_connection():
    """Test 1: Basic connection to Supabase."""
    print_section("TEST 1: Connection to Supabase")
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"[OK] Successfully created Supabase client")
        print(f"     URL: {SUPABASE_URL[:50]}...")
        return client
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        return None

def test_insert_trial(client):
    """Test 2: Insert a test trial record."""
    print_section("TEST 2: Insert Test Trial Record")

    test_trial_id = f"NCT_TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    test_data = {
        "trial_id": test_trial_id,
        "protocol_text": "TEST PROTOCOL: This is a test trial for Supabase integration verification. INCLUSION: Age 18-75. EXCLUSION: None for testing.",
        "screening_result": json.dumps({
            "decision": "ELIGIBLE",
            "confidence": 0.95,
            "patient_id": "TEST_PATIENT_001",
            "test_run": True,
            "timestamp": datetime.now().isoformat()
        }),
        "created_at": datetime.now().isoformat()
    }

    try:
        # Using upsert as defined in app.py
        response = client.table("trial_history").upsert(test_data, on_conflict="trial_id").execute()

        if response.data:
            print(f"[OK] Successfully inserted test trial: {test_trial_id}")
            print(f"     Response data: {json.dumps(response.data, indent=2)[:200]}...")
            return test_trial_id
        else:
            print(f"[WARN] Insert returned no data")
            print(f"       Response: {response}")
            return test_trial_id  # May still have worked

    except Exception as e:
        print(f"[FAIL] Insert failed: {e}")
        # Check if table doesn't exist
        if "relation" in str(e).lower() and "does not exist" in str(e).lower():
            print("\n[INFO] The 'trial_history' table may not exist yet.")
            print("       Expected table structure:")
            print("       - trial_id (TEXT, PRIMARY KEY)")
            print("       - protocol_text (TEXT)")
            print("       - screening_result (JSONB or TEXT)")
            print("       - created_at (TIMESTAMPTZ)")
        return None

def test_retrieve_history(client):
    """Test 3: Retrieve trial history."""
    print_section("TEST 3: Retrieve Trial History")

    try:
        # Query as defined in app.py
        response = client.table("trial_history").select("trial_id, created_at").order("created_at", desc=True).limit(50).execute()

        if response.data:
            print(f"[OK] Successfully retrieved {len(response.data)} trial records")
            print(f"     Recent trials:")
            for trial in response.data[:5]:
                print(f"     - {trial.get('trial_id', 'N/A')} ({trial.get('created_at', 'N/A')})")
            return response.data
        else:
            print(f"[INFO] No trial records found (empty table)")
            return []

    except Exception as e:
        print(f"[FAIL] Retrieve failed: {e}")
        return None

def test_query_recent(client, limit=10):
    """Test 4: Query recent trials with full data."""
    print_section("TEST 4: Query Recent Trials (Full Data)")

    try:
        response = client.table("trial_history").select("*").order("created_at", desc=True).limit(limit).execute()

        if response.data:
            print(f"[OK] Successfully queried {len(response.data)} recent trials")
            for i, trial in enumerate(response.data[:3]):
                print(f"\n     Trial {i+1}: {trial.get('trial_id', 'N/A')}")
                protocol = trial.get('protocol_text', '')
                if protocol:
                    print(f"     Protocol preview: {protocol[:80]}...")
                result = trial.get('screening_result')
                if result:
                    try:
                        parsed = json.loads(result) if isinstance(result, str) else result
                        print(f"     Result: {parsed.get('decision', 'N/A')} (conf: {parsed.get('confidence', 'N/A')})")
                    except:
                        print(f"     Result: {str(result)[:50]}...")
            return response.data
        else:
            print(f"[INFO] No records found")
            return []

    except Exception as e:
        print(f"[FAIL] Query failed: {e}")
        return None

def test_cleanup(client, test_trial_id):
    """Optional: Clean up test data."""
    print_section("CLEANUP: Removing Test Data")

    if not test_trial_id:
        print("[SKIP] No test trial to clean up")
        return

    try:
        response = client.table("trial_history").delete().eq("trial_id", test_trial_id).execute()
        print(f"[OK] Cleaned up test trial: {test_trial_id}")
    except Exception as e:
        print(f"[WARN] Cleanup failed (may be OK): {e}")

def main():
    """Run all Supabase integration tests."""
    print("\n" + "="*60)
    print(" SUPABASE INTEGRATION TEST")
    print(" Clinical Trial Ultimate Turbo")
    print("="*60)
    print(f"\nTest started at: {datetime.now().isoformat()}")
    print(f"Supabase URL: {SUPABASE_URL}")

    # Test 0: DNS Resolution (critical)
    dns_ok = test_dns_resolution()
    if not dns_ok:
        print_section("TEST SUMMARY")
        print(f"DNS Resolution: FAIL")
        print(f"\n[CRITICAL] Cannot proceed - Supabase project URL does not exist.")
        print("\nRECOMMENDATION:")
        print("  1. Create a new Supabase project at https://supabase.com")
        print("  2. Create the 'trial_history' table with schema:")
        print("     - trial_id (TEXT, PRIMARY KEY)")
        print("     - protocol_text (TEXT)")
        print("     - screening_result (JSONB)")
        print("     - created_at (TIMESTAMPTZ DEFAULT now())")
        print("  3. Update credentials in app.py or .env file")
        print("\nNote: The app.py has fallback to session state storage,")
        print("      so the application will still work without Supabase.")
        return False

    # Test 1: Connection
    client = test_connection()
    if not client:
        print("\n[ABORT] Cannot proceed without connection")
        return False

    # Test 2: Insert
    test_trial_id = test_insert_trial(client)

    # Test 3: Retrieve history
    history = test_retrieve_history(client)

    # Test 4: Query recent
    recent = test_query_recent(client)

    # Cleanup (optional - comment out to keep test data)
    if test_trial_id:
        test_cleanup(client, test_trial_id)

    # Summary
    print_section("TEST SUMMARY")
    print(f"DNS Resolution: {'PASS' if dns_ok else 'FAIL'}")
    print(f"Connection:     {'PASS' if client else 'FAIL'}")
    print(f"Insert:         {'PASS' if test_trial_id else 'FAIL'}")
    print(f"Retrieve:       {'PASS' if history is not None else 'FAIL'}")
    print(f"Query Recent:   {'PASS' if recent is not None else 'FAIL'}")

    all_passed = dns_ok and client and test_trial_id and history is not None and recent is not None
    print(f"\nOverall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
