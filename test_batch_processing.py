#!/usr/bin/env python3
"""
Test script for batch processing - tests ULTIMATE TURBO with 10 patients
"""

import os
import sys
import json
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
from dotenv import load_dotenv
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)

# Use second API key (first one was flagged as leaked)
api_key = "AIzaSyCR94dkEqwvUhB6xNapRycZx_rfWbFM0oU"
os.environ["GOOGLE_API_KEY"] = api_key
print(f"Using API key: {api_key[:20]}...")

# Import the agent
from src.agents.supervisor_ultimate_turbo import UltimateTurboSupervisorAgent

# Load DECLARE-TIMI58 protocol
PROTOCOL = """# Trial: DECLARE-TIMI58

### Inclusion Criteria
- Female or male aged ≥40 years
- Diagnosed with Type 2 Diabetes
- High Risk for Cardiovascular events
- HbA1c between 6.5% and 12%

### Exclusion Criteria
- Diagnosis of Type 1 diabetes mellitus
- History of bladder cancer or history of radiation therapy to the lower abdomen or pelvis at any time
- Chronic cystitis and/or recurrent urinary tract infections
- Pregnant or breast-feeding patients
- eGFR < 30 mL/min/1.73m2 (severe renal impairment)
"""


def load_patients(csv_path: str, limit: int = 10) -> list:
    """Load patients from CSV file with PROPER FIELD MAPPING."""
    df = pd.read_csv(csv_path, nrows=limit)
    patients = []

    for _, row in df.iterrows():
        # Extract comorbidities for diagnosis matching
        comorbidities = str(row.get("comorbidities", ""))

        patient = {
            "patient_id": str(row.get("patient_id", "UNKNOWN")),
            "age": row.get("age"),
            "sex": row.get("gender", "Unknown"),  # Map gender -> sex
            "gender": row.get("gender", "Unknown"),

            # LABS - structured for agent matching
            "labs": {
                "hba1c": row.get("hba1c"),
                "egfr": row.get("egfr"),
            },
            "hba1c": row.get("hba1c"),  # Also at top level
            "egfr": row.get("egfr"),    # Also at top level

            # DIAGNOSES - from comorbidities column
            "diagnoses": comorbidities,
            "conditions": comorbidities,

            # MEDICATIONS
            "medications": str(row.get("current_medications", "")),
            "current_medications": str(row.get("current_medications", "")),

            "insulin_user": row.get("insulin_user", False),
        }
        patients.append(patient)

    return patients


async def run_batch_test():
    """Run batch processing test."""
    print("\n" + "="*60)
    print("ULTIMATE TURBO BATCH PROCESSING TEST")
    print("="*60 + "\n")

    # Load patients
    csv_path = PROJECT_ROOT / "patients_for_trial_screening.csv"
    if not csv_path.exists():
        print(f"ERROR: Patients file not found at {csv_path}")
        return

    patients = load_patients(csv_path, limit=300)  # FULL 300 patients with FIXED mapping
    print(f"Loaded {len(patients)} patients for testing\n")

    # Show sample patient
    print("Sample patient data:")
    print(json.dumps(patients[0], indent=2, default=str))
    print()

    # Initialize agent with balanced settings
    # Using gemini-2.0-flash (non-exp) with moderate concurrency
    print("Initializing ULTIMATE TURBO agent...")
    agent = UltimateTurboSupervisorAgent(
        batch_size=20,       # Balanced batches
        max_concurrent=5,    # Moderate concurrency for stability
        enable_calibration=True
    )
    print("Agent ready (batch_size=20, max_concurrent=5)\n")

    # Run batch screening
    print(f"Starting batch screening for {len(patients)} patients...")
    print("Trial: DECLARE-TIMI58")
    print("-"*60)

    start_time = datetime.now()

    try:
        results = await agent.batch_screen_parallel_advanced(
            patients=patients,
            trial_protocol=PROTOCOL,
            trial_id="DECLARE-TIMI58",
            progress_callback=lambda done, total, msg: print(f"  Progress: {done}/{total} - {msg}")
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("-"*60)
        print(f"\nCompleted in {duration:.1f} seconds")
        print(f"Average: {duration/len(patients):.2f}s per patient\n")

        # Summarize results
        print("="*60)
        print("RESULTS SUMMARY")
        print("="*60 + "\n")

        decisions = {}
        for r in results:
            d = r.get("decision", "ERROR")
            decisions[d] = decisions.get(d, 0) + 1

        print("Decision distribution:")
        for decision, count in sorted(decisions.items()):
            print(f"  {decision}: {count} ({count/len(results)*100:.0f}%)")

        # Show detailed results for first 3 patients
        print("\n" + "-"*60)
        print("DETAILED RESULTS (first 3 patients):")
        print("-"*60 + "\n")

        for r in results[:3]:
            print(f"Patient: {r.get('patient_id')}")
            print(f"  Decision: {r.get('decision')}")
            print(f"  Confidence: {r.get('confidence', r.get('confidence_score', 0))*100:.0f}%")
            print(f"  Human Review: {r.get('requires_human_review', 'N/A')}")

            # Show criteria matches if available
            matches = r.get("matching_results", r.get("criteria_matches", []))
            if matches:
                print(f"  Criteria evaluated: {len(matches)}")
                for m in matches[:3]:
                    status = m.get("match_status", m.get("status", "?"))
                    crit_id = m.get("criterion_id", m.get("id", "?"))
                    print(f"    - {crit_id}: {status}")

            # Show narrative if available
            narrative = r.get("narrative")
            if narrative:
                print(f"  Narrative: {narrative[:200]}...")
            print()

        # Save full results
        output_file = PROJECT_ROOT / f"batch_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nFull results saved to: {output_file}")

        return results

    except Exception as e:
        print(f"\nERROR during batch processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = asyncio.run(run_batch_test())
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
