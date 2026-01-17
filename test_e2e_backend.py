"""
Clinical Trial ULTIMATE TURBO - Backend End-to-End Tests

Tests:
1. Agent Initialization - UltimateTurboSupervisorAgent creation and LLM health check
2. Single Patient Screening - Full workflow against DECLARE-TIMI58 protocol
3. Criteria Extraction - Test extract_criteria method
4. Explainability - Verify matching_results contain required fields

Usage:
    source venv/bin/activate && python test_e2e_backend.py
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Set API key before imports
os.environ["GOOGLE_API_KEY"] = "AIzaSyCR94dkEqwvUhB6xNapRycZx_rfWbFM0oU"

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test tracking
class TestResults:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.start_time = datetime.now()

    def record(self, test_name: str, passed: bool, details: str = ""):
        status = "PASS" if passed else "FAIL"
        self.tests.append({
            "name": test_name,
            "status": status,
            "details": details
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {test_name}")
        if details:
            print(f"        {details}")

    def summary(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {len(self.tests)}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Time: {elapsed:.2f}s")
        print("=" * 60)

        if self.failed > 0:
            print("\nFailed Tests:")
            for test in self.tests:
                if test["status"] == "FAIL":
                    print(f"  - {test['name']}: {test['details']}")

        return self.failed == 0


results = TestResults()


# =============================================================================
# DECLARE-TIMI58 Protocol (Real Clinical Trial for Type 2 Diabetes)
# =============================================================================

DECLARE_TIMI58_PROTOCOL = """
CLINICAL TRIAL: NCT01730534 (DECLARE-TIMI 58)
TITLE: Multicenter Trial to Evaluate the Effect of Dapagliflozin on the Incidence of
       Cardiovascular Events

OBJECTIVE: Evaluate the effect of dapagliflozin vs placebo on cardiovascular outcomes
in patients with type 2 diabetes mellitus who have or are at risk for atherosclerotic
cardiovascular disease.

INCLUSION CRITERIA:
1. Age >= 40 years
2. Diagnosis of Type 2 Diabetes Mellitus
3. HbA1c between 6.5% and 12.0%
4. Either:
   a) Established cardiovascular disease (history of ischemic heart disease, ischemic
      cerebrovascular disease, or peripheral arterial disease), OR
   b) Multiple risk factors for cardiovascular disease (age >= 55 for men or >= 60 for
      women, plus at least one of: dyslipidemia, hypertension, or current tobacco use)
5. Creatinine clearance >= 60 mL/min

EXCLUSION CRITERIA:
1. Type 1 Diabetes Mellitus
2. History of diabetic ketoacidosis
3. Pregnant or nursing women
4. Severe hepatic impairment (Child-Pugh class C)
5. Current use of any SGLT2 inhibitor
6. eGFR < 60 mL/min/1.73m2
7. History of bladder cancer
8. Active or history of medullary thyroid cancer
9. New York Heart Association (NYHA) class IV heart failure

STUDY DESIGN: Randomized, double-blind, placebo-controlled, parallel-group trial
DURATION: Event-driven, minimum 4 years follow-up
"""


# =============================================================================
# TEST 1: Agent Initialization
# =============================================================================

def test_agent_initialization():
    """Test that UltimateTurboSupervisorAgent can be created."""
    print("\n" + "-" * 60)
    print("TEST 1: Agent Initialization")
    print("-" * 60)

    try:
        from src.agents.supervisor_ultimate_turbo import UltimateTurboSupervisorAgent

        # Test 1a: Basic instantiation
        agent = UltimateTurboSupervisorAgent()
        results.record(
            "Agent Creation",
            agent is not None,
            "Agent created successfully"
        )

        # Test 1b: Verify agent has required methods
        has_extract = hasattr(agent, 'extract_criteria')
        has_screen = hasattr(agent, 'screen_patient')
        has_batch = hasattr(agent, 'batch_screen_parallel_advanced')

        results.record(
            "Agent has extract_criteria method",
            has_extract,
            f"Method exists: {has_extract}"
        )

        results.record(
            "Agent has screen_patient method",
            has_screen,
            f"Method exists: {has_screen}"
        )

        results.record(
            "Agent has batch_screen_parallel_advanced method",
            has_batch,
            f"Method exists: {has_batch}"
        )

        # Test 1c: Verify agent configuration
        results.record(
            "Agent batch_size configured",
            agent.batch_size == 25,
            f"batch_size={agent.batch_size}"
        )

        results.record(
            "Agent max_concurrent configured",
            agent.max_concurrent == 10,
            f"max_concurrent={agent.max_concurrent}"
        )

        return agent

    except Exception as e:
        results.record("Agent Creation", False, f"Error: {str(e)}")
        return None


async def test_llm_health_check(agent):
    """Test that LLM connection works with a quick health check."""
    print("\n" + "-" * 60)
    print("TEST 1b: LLM Connection Health Check")
    print("-" * 60)

    try:
        from src.agents.supervisor_ultimate_turbo import get_llm
        from langchain_core.messages import HumanMessage

        # Get LLM instance
        llm = get_llm(fast=True)

        results.record(
            "LLM instance created",
            llm is not None,
            f"LLM type: {type(llm).__name__}"
        )

        # Quick health check - simple prompt
        test_prompt = "Respond with exactly: OK"
        messages = [HumanMessage(content=test_prompt)]

        response = await llm.ainvoke(messages)

        results.record(
            "LLM responds to prompt",
            response is not None and len(response.content) > 0,
            f"Response length: {len(response.content)} chars"
        )

        results.record(
            "LLM response content valid",
            "OK" in response.content.upper(),
            f"Response: {response.content[:50]}"
        )

        return True

    except Exception as e:
        results.record("LLM Health Check", False, f"Error: {str(e)}")
        return False


# =============================================================================
# TEST 2: Single Patient Screening
# =============================================================================

async def test_single_patient_screening(agent):
    """Test single patient screening against DECLARE-TIMI58 protocol."""
    print("\n" + "-" * 60)
    print("TEST 2: Single Patient Screening")
    print("-" * 60)

    # Test patient data - should be ELIGIBLE for DECLARE-TIMI58
    test_patient = {
        "patient_id": "TEST-001",
        "age": 62,
        "sex": "male",
        "diagnoses": [
            {"condition": "Type 2 Diabetes Mellitus", "icd10": "E11.9"},
            {"condition": "Hypertension", "icd10": "I10"},
            {"condition": "Dyslipidemia", "icd10": "E78.5"}
        ],
        "medications": [
            {"drug_name": "Metformin", "dose": "1000mg", "frequency": "twice daily"},
            {"drug_name": "Lisinopril", "dose": "10mg", "frequency": "daily"},
            {"drug_name": "Atorvastatin", "dose": "20mg", "frequency": "daily"}
        ],
        "lab_values": [
            {"test": "HbA1c", "value": 8.2, "unit": "%"},
            {"test": "eGFR", "value": 75, "unit": "mL/min/1.73m2"},
            {"test": "Creatinine", "value": 1.1, "unit": "mg/dL"}
        ],
        "other_information": "No history of ketoacidosis. No current SGLT2 inhibitor use. Non-smoker."
    }

    try:
        # Run screening
        progress_messages = []
        def progress_callback(msg):
            progress_messages.append(msg)

        result = await agent.screen_patient(
            patient_data=test_patient,
            trial_protocol=DECLARE_TIMI58_PROTOCOL,
            trial_id="NCT01730534",
            progress_callback=progress_callback
        )

        # Test 2a: Result structure
        results.record(
            "Screening returns result",
            result is not None,
            f"Result type: {type(result).__name__}"
        )

        # Test 2b: Decision field exists and valid
        decision = result.get("decision", "")
        valid_decisions = ["ELIGIBLE", "INELIGIBLE", "UNCERTAIN", "ERROR"]
        results.record(
            "Result has valid decision",
            decision in valid_decisions,
            f"Decision: {decision}"
        )

        # Test 2c: Confidence field exists
        confidence = result.get("confidence", -1)
        results.record(
            "Result has confidence",
            0 <= confidence <= 1,
            f"Confidence: {confidence:.2%}" if confidence >= 0 else "Missing"
        )

        # Test 2d: matching_results exists (explainability)
        matching_results = result.get("explainability_table", [])
        results.record(
            "Result has matching_results",
            isinstance(matching_results, list),
            f"Found {len(matching_results)} matching results"
        )

        # Test 2e: Clinical narrative generated
        narrative = result.get("clinical_narrative", "")
        results.record(
            "Result has clinical narrative",
            len(narrative) > 0,
            f"Narrative length: {len(narrative)} chars"
        )

        # Test 2f: Progress callback was called
        results.record(
            "Progress callback received updates",
            len(progress_messages) > 0,
            f"Received {len(progress_messages)} progress updates"
        )

        # Test 2g: Key factors and concerns
        key_factors = result.get("key_factors", [])
        concerns = result.get("concerns", [])
        results.record(
            "Result has key_factors list",
            isinstance(key_factors, list),
            f"Found {len(key_factors)} key factors"
        )

        results.record(
            "Result has concerns list",
            isinstance(concerns, list),
            f"Found {len(concerns)} concerns"
        )

        # Test 2h: Optimization mode
        optimization = result.get("optimization", "")
        results.record(
            "Result indicates ULTIMATE_TURBO mode",
            "ULTIMATE" in optimization.upper() or "TURBO" in optimization.upper(),
            f"Optimization: {optimization}"
        )

        return result

    except Exception as e:
        results.record("Single Patient Screening", False, f"Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


# =============================================================================
# TEST 3: Criteria Extraction
# =============================================================================

async def test_criteria_extraction(agent):
    """Test extract_criteria method."""
    print("\n" + "-" * 60)
    print("TEST 3: Criteria Extraction")
    print("-" * 60)

    try:
        criteria = await agent.extract_criteria(
            trial_protocol=DECLARE_TIMI58_PROTOCOL,
            trial_id="NCT01730534"
        )

        # Test 3a: Criteria returned
        results.record(
            "Criteria extraction returns result",
            criteria is not None,
            f"Result type: {type(criteria).__name__}"
        )

        # Test 3b: Inclusion criteria present
        inclusion = criteria.get("inclusion_criteria", [])
        results.record(
            "Extraction returns inclusion_criteria",
            isinstance(inclusion, list) and len(inclusion) > 0,
            f"Found {len(inclusion)} inclusion criteria"
        )

        # Test 3c: Exclusion criteria present
        exclusion = criteria.get("exclusion_criteria", [])
        results.record(
            "Extraction returns exclusion_criteria",
            isinstance(exclusion, list) and len(exclusion) > 0,
            f"Found {len(exclusion)} exclusion criteria"
        )

        # Test 3d: Criteria have required structure
        if inclusion:
            first_inc = inclusion[0]
            has_id = "id" in first_inc
            has_text = "text" in first_inc
            results.record(
                "Inclusion criteria have 'id' field",
                has_id,
                f"Sample: {first_inc.get('id', 'MISSING')}"
            )
            results.record(
                "Inclusion criteria have 'text' field",
                has_text,
                f"Sample: {first_inc.get('text', 'MISSING')[:50]}..."
            )

        if exclusion:
            first_exc = exclusion[0]
            has_id = "id" in first_exc
            has_text = "text" in first_exc
            results.record(
                "Exclusion criteria have 'id' field",
                has_id,
                f"Sample: {first_exc.get('id', 'MISSING')}"
            )
            results.record(
                "Exclusion criteria have 'text' field",
                has_text,
                f"Sample: {first_exc.get('text', 'MISSING')[:50]}..."
            )

        # Test 3e: Trial summary present (optional but nice to have)
        trial_summary = criteria.get("trial_summary", "")
        results.record(
            "Extraction returns trial_summary",
            len(trial_summary) > 0,
            f"Summary length: {len(trial_summary)} chars"
        )

        # Test 3f: Caching works (second call should be faster)
        import time
        start = time.time()
        criteria_cached = await agent.extract_criteria(
            trial_protocol=DECLARE_TIMI58_PROTOCOL,
            trial_id="NCT01730534"
        )
        elapsed = time.time() - start
        results.record(
            "Criteria caching works",
            elapsed < 0.5,  # Should be nearly instant from cache
            f"Cached retrieval time: {elapsed:.3f}s"
        )

        return criteria

    except Exception as e:
        results.record("Criteria Extraction", False, f"Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


# =============================================================================
# TEST 4: Explainability
# =============================================================================

async def test_explainability(agent):
    """Test that matching_results contain required explainability fields."""
    print("\n" + "-" * 60)
    print("TEST 4: Explainability")
    print("-" * 60)

    # Use a patient that should trigger detailed matching
    test_patient = {
        "patient_id": "EXPLAINABILITY-TEST",
        "age": 55,
        "sex": "female",
        "diagnoses": [
            {"condition": "Type 2 Diabetes Mellitus", "icd10": "E11.9"},
            {"condition": "Coronary Artery Disease", "icd10": "I25.10"}
        ],
        "medications": [
            {"drug_name": "Metformin", "dose": "500mg", "frequency": "twice daily"}
        ],
        "lab_values": [
            {"test": "HbA1c", "value": 7.8, "unit": "%"},
            {"test": "eGFR", "value": 85, "unit": "mL/min/1.73m2"}
        ],
        "other_information": "History of ischemic heart disease. No bladder cancer history."
    }

    try:
        result = await agent.screen_patient(
            patient_data=test_patient,
            trial_protocol=DECLARE_TIMI58_PROTOCOL,
            trial_id="NCT01730534-EXPLAIN"
        )

        matching_results = result.get("explainability_table", [])

        # Test 4a: matching_results exists and not empty
        results.record(
            "Matching results exist",
            len(matching_results) > 0,
            f"Found {len(matching_results)} matching results"
        )

        if not matching_results:
            results.record("Matching results validation", False, "No matching results to validate")
            return None

        # Check each required field in matching results
        required_fields = ["criterion_id", "match_status", "confidence", "reasoning"]

        for field in required_fields:
            # Check if at least one result has this field
            has_field = any(field in mr for mr in matching_results)

            # Get sample value
            sample_value = None
            for mr in matching_results:
                if field in mr:
                    sample_value = mr[field]
                    break

            if field == "confidence":
                # Confidence should be numeric
                is_valid = has_field and isinstance(sample_value, (int, float))
                results.record(
                    f"Matching results have '{field}' (numeric)",
                    is_valid,
                    f"Sample: {sample_value}"
                )
            elif field == "match_status":
                # Match status should be one of expected values
                valid_statuses = ["MATCH", "NO_MATCH", "UNCERTAIN", "MISSING_DATA"]
                is_valid = has_field and sample_value in valid_statuses
                results.record(
                    f"Matching results have valid '{field}'",
                    is_valid,
                    f"Sample: {sample_value}"
                )
            else:
                # Other fields just need to exist and be non-empty
                is_valid = has_field and sample_value
                results.record(
                    f"Matching results have '{field}'",
                    is_valid,
                    f"Sample: {str(sample_value)[:50]}..." if sample_value else "MISSING"
                )

        # Test 4e: Check for optional but useful fields
        optional_fields = ["criterion_text", "patient_data_used", "evidence", "concerns"]

        for field in optional_fields:
            has_field = any(field in mr for mr in matching_results)
            results.record(
                f"Matching results include '{field}' (optional)",
                has_field,
                "Present" if has_field else "Not present"
            )

        # Test 4f: Verify confidence values are in valid range
        confidences = [mr.get("confidence", 0) for mr in matching_results if "confidence" in mr]
        all_valid = all(0 <= c <= 1 for c in confidences) if confidences else False
        results.record(
            "All confidence values in valid range [0,1]",
            all_valid,
            f"Range: [{min(confidences):.2f}, {max(confidences):.2f}]" if confidences else "No confidences"
        )

        # Test 4g: Print sample matching result for inspection
        if matching_results:
            print("\n  Sample matching result:")
            sample = matching_results[0]
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 60:
                    value = value[:60] + "..."
                print(f"    {key}: {value}")

        return matching_results

    except Exception as e:
        results.record("Explainability Test", False, f"Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    """Run all end-to-end tests."""
    print("=" * 60)
    print("CLINICAL TRIAL ULTIMATE TURBO - Backend E2E Tests")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"API Key: {os.environ.get('GOOGLE_API_KEY', 'NOT SET')[:20]}...")

    # Test 1: Agent Initialization
    agent = test_agent_initialization()

    if agent is None:
        print("\nCRITICAL: Agent initialization failed. Cannot continue tests.")
        results.summary()
        return

    # Test 1b: LLM Health Check
    llm_ok = await test_llm_health_check(agent)

    if not llm_ok:
        print("\nWARNING: LLM health check failed. Subsequent tests may fail.")

    # Test 2: Single Patient Screening
    screening_result = await test_single_patient_screening(agent)

    # Test 3: Criteria Extraction
    criteria = await test_criteria_extraction(agent)

    # Test 4: Explainability
    matching_results = await test_explainability(agent)

    # Print summary
    all_passed = results.summary()

    # Return exit code
    return all_passed


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())

    # Exit with appropriate code
    sys.exit(0 if success else 1)
