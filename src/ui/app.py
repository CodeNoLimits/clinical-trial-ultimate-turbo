"""
Clinical Trial Screening - ULTIMATE TURBO Edition

Features:
- TURBO parallel batch processing (300 patients in 60-90s)
- ADVANCED explainability tables (criterion-by-criterion)
- Confidence calibration (epistemic/aleatoric uncertainty)
- Multi-provider LLM support (Google + OpenAI fallback)
- Developer mode for JSON access
- Trial history with persistence
- Clean professional UI
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import hashlib
import requests

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env only if exists (local dev)
from dotenv import load_dotenv
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)


# =============================================================================
# DEVELOPER AUTHENTICATION
# =============================================================================

DEVELOPER_CREDENTIALS = {
    "david": hashlib.sha256("clinicaltrial".encode()).hexdigest(),
    "melea": hashlib.sha256("clinicaltrial".encode()).hexdigest(),
}


def check_developer_login(username: str, password: str) -> bool:
    """Verify developer credentials."""
    if username.lower() in DEVELOPER_CREDENTIALS:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return DEVELOPER_CREDENTIALS[username.lower()] == password_hash
    return False


def is_developer_mode() -> bool:
    """Check if user is logged in as developer."""
    return st.session_state.get("developer_logged_in", False)


# =============================================================================
# DATABASE - TRIAL HISTORY PERSISTENCE
# =============================================================================

def get_supabase_client():
    """Get Supabase client for database operations."""
    try:
        from supabase import create_client
        url = None
        key = None
        try:
            url = st.secrets.get("supabase", {}).get("url")
            key = st.secrets.get("supabase", {}).get("key")
        except Exception:
            pass
        if not url:
            url = os.getenv("SUPABASE_URL")
        if not key:
            key = os.getenv("SUPABASE_KEY")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None


def save_trial_to_db(trial_id: str, protocol: str = None, result: dict = None):
    """Save trial to database for history."""
    client = get_supabase_client()
    if client:
        try:
            data = {
                "trial_id": trial_id,
                "protocol_text": protocol[:5000] if protocol else None,
                "screening_result": json.dumps(result) if result else None,
                "created_at": datetime.now().isoformat()
            }
            client.table("trial_history").upsert(data, on_conflict="trial_id").execute()
            return True
        except Exception:
            pass

    # Fallback to session state
    if "trial_history_local" not in st.session_state:
        st.session_state.trial_history_local = []
    existing_ids = [t["trial_id"] for t in st.session_state.trial_history_local]
    if trial_id not in existing_ids:
        st.session_state.trial_history_local.append({
            "trial_id": trial_id,
            "created_at": datetime.now().isoformat(),
            "has_result": result is not None
        })
        st.session_state.trial_history_local = st.session_state.trial_history_local[-50:]
    return False


def get_trial_history() -> List[dict]:
    """Get trial history from database or session."""
    client = get_supabase_client()
    if client:
        try:
            response = client.table("trial_history").select("trial_id, created_at").order("created_at", desc=True).limit(50).execute()
            return response.data
        except Exception:
            pass
    return st.session_state.get("trial_history_local", [])


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

def get_api_key() -> str:
    """Get Google API key."""
    try:
        api_key = st.secrets.get("google", {}).get("api_key")
        if api_key and api_key != "your_gemini_api_key_here":
            os.environ["GOOGLE_API_KEY"] = api_key
            return api_key
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if api_key and api_key != "your_gemini_api_key_here":
            os.environ["GOOGLE_API_KEY"] = api_key
            return api_key
    except Exception:
        pass
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "your_gemini_api_key_here":
        return api_key
    if "google_api_key" in st.session_state and st.session_state.google_api_key:
        os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
        return st.session_state.google_api_key
    return None


# =============================================================================
# CLINICALTRIALS.GOV API INTEGRATION
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_trial_from_clinicaltrials_gov(nct_id: str) -> Dict[str, Any]:
    """
    Fetch trial data from ClinicalTrials.gov API.
    API Documentation: https://clinicaltrials.gov/data-api/api
    """
    if not nct_id or not nct_id.strip():
        return None

    # Clean and validate NCT ID format
    nct_id = nct_id.strip().upper()
    if not nct_id.startswith("NCT"):
        return None

    try:
        # ClinicalTrials.gov API v2
        url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
        headers = {"Accept": "application/json"}

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Extract relevant protocol information
            protocol_section = data.get("protocolSection", {})
            identification = protocol_section.get("identificationModule", {})
            eligibility = protocol_section.get("eligibilityModule", {})
            description = protocol_section.get("descriptionModule", {})
            conditions = protocol_section.get("conditionsModule", {})

            # Build protocol text
            title = identification.get("officialTitle") or identification.get("briefTitle", "")
            brief_summary = description.get("briefSummary", "")
            detailed_desc = description.get("detailedDescription", "")

            # Eligibility criteria
            eligibility_text = eligibility.get("eligibilityCriteria", "")
            min_age = eligibility.get("minimumAge", "N/A")
            max_age = eligibility.get("maximumAge", "N/A")
            sex = eligibility.get("sex", "All")
            healthy_volunteers = eligibility.get("healthyVolunteers", "No")

            # Conditions
            conditions_list = conditions.get("conditions", [])
            keywords = conditions.get("keywords", [])

            # Format the protocol
            protocol_text = f"""
CLINICAL TRIAL: {nct_id}
TITLE: {title}

BRIEF SUMMARY:
{brief_summary}

CONDITIONS: {', '.join(conditions_list) if conditions_list else 'Not specified'}
KEYWORDS: {', '.join(keywords) if keywords else 'Not specified'}

AGE REQUIREMENTS: {min_age} to {max_age}
SEX: {sex}
HEALTHY VOLUNTEERS: {healthy_volunteers}

ELIGIBILITY CRITERIA:
{eligibility_text}

DETAILED DESCRIPTION:
{detailed_desc if detailed_desc else 'Not provided'}
"""

            return {
                "nct_id": nct_id,
                "title": title,
                "protocol_text": protocol_text.strip(),
                "conditions": conditions_list,
                "eligibility_criteria": eligibility_text,
                "min_age": min_age,
                "max_age": max_age,
                "sex": sex,
                "brief_summary": brief_summary,
                "status": data.get("protocolSection", {}).get("statusModule", {}).get("overallStatus", "Unknown"),
                "raw_data": data
            }
        elif response.status_code == 404:
            return {"error": f"Trial {nct_id} not found on ClinicalTrials.gov"}
        else:
            return {"error": f"API error: {response.status_code}"}

    except requests.exceptions.Timeout:
        return {"error": "Request timeout - ClinicalTrials.gov API slow"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error fetching trial: {str(e)}"}


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Clinical Trial Screening",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# THEME CONFIGURATION
# =============================================================================
from src.ui.styles import get_css

# Sidebar Theme Selector
theme_mode = st.sidebar.selectbox(
    "Theme Mode",
    ["Midnight Pro", "Clinical Clean"],
    index=1,  # Temporarily Clinical Clean for testing
    help="Midnight Pro (dark) or Clinical Clean (light)"
)

# Inject CSS based on selection
st.markdown(get_css(theme_mode), unsafe_allow_html=True)



# =============================================================================
# SESSION STATE
# =============================================================================

if "screening_result" not in st.session_state:
    st.session_state.screening_result = None
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "patient_validated" not in st.session_state:
    st.session_state.patient_validated = False
if "trial_history_local" not in st.session_state:
    st.session_state.trial_history_local = []
if "selected_trial_id" not in st.session_state:
    st.session_state.selected_trial_id = ""
if "developer_logged_in" not in st.session_state:
    st.session_state.developer_logged_in = False
if "uploaded_protocols" not in st.session_state:
    st.session_state.uploaded_protocols = {}
if "batch_patients" not in st.session_state:
    st.session_state.batch_patients = []
if "batch_trial_id_selected" not in st.session_state:
    st.session_state.batch_trial_id_selected = ""
if "patients_source" not in st.session_state:
    st.session_state.patients_source = None  # "single" or "batch"


def clear_session():
    """Clear all session state for new patient."""
    st.session_state.screening_result = None
    st.session_state.patient_data = None
    st.session_state.patient_validated = False


def validate_patient_data(data: dict) -> tuple:
    """Validate patient data structure (backend validation)."""
    # Normalize gender/sex field - accept both column names
    if "gender" in data and "sex" not in data:
        data["sex"] = data["gender"]

    required_fields = ["patient_id", "age", "sex"]
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    if not isinstance(data.get("age"), (int, float)) or data["age"] < 0:
        return False, "Invalid age value"
    # Accept various sex/gender values and normalize
    sex_value = str(data.get("sex", "")).lower().strip()
    valid_male = ["male", "m", "homme", "masculin", "man"]
    valid_female = ["female", "f", "femme", "féminin", "woman"]
    valid_other = ["other", "autre", "non-binary", "unknown", ""]

    if sex_value in valid_male:
        data["sex"] = "male"
    elif sex_value in valid_female:
        data["sex"] = "female"
    elif sex_value in valid_other:
        data["sex"] = "other"
    else:
        data["sex"] = "other"  # Default to other for unknown values

    return True, "Valid"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_eligibility_class(decision: str) -> str:
    return f"eligibility-{decision.lower()}"


def create_confidence_gauge(confidence: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#6f42c1"},
            'steps': [
                {'range': [0, 70], 'color': "#dc3545"},
                {'range': [70, 80], 'color': "#ffc107"},
                {'range': [80, 90], 'color': "#17a2b8"},
                {'range': [90, 100], 'color': "#28a745"}
            ],
        }
    ))
    fig.update_layout(height=280)
    return fig


def run_fast_screening(patient_data: dict, trial_protocol: str, trial_id: str, progress_container=None) -> dict:
    """Run FAST screening with real-time progress updates."""
    try:
        from src.agents.supervisor_fast import FastSupervisorAgent
        agent = FastSupervisorAgent()

        progress_bar = None
        status_text = None
        if progress_container:
            progress_bar = progress_container.progress(0, text="Initializing...")
            status_text = progress_container.empty()

        current_progress = [0]

        def update_progress(message: str):
            if progress_bar:
                if "Step 1" in message:
                    current_progress[0] = 30
                elif "Step 2" in message:
                    current_progress[0] = 70
                progress_bar.progress(current_progress[0], text=message)
            if status_text:
                status_text.info(f"Processing: {message}")

        async def _async_screen():
            return await agent.screen_patient(
                patient_data=patient_data,
                trial_protocol=trial_protocol,
                trial_id=trial_id,
                progress_callback=update_progress
            )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_async_screen())
        finally:
            loop.close()

        if progress_bar:
            progress_bar.progress(100, text="Complete!")
        if status_text:
            status_text.success("Screening complete!")

        return result

    except Exception as e:
        st.error(f"Screening error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def normalize_patient_data(patient: dict) -> dict:
    """
    Normalize patient data to the format expected by the agent.
    Maps common CSV column names to the expected field names.
    """
    normalized = dict(patient)  # Copy original

    # Map gender -> sex
    if "gender" in patient and "sex" not in patient:
        normalized["sex"] = patient["gender"]

    # Map comorbidities -> diagnoses
    if "comorbidities" in patient and "diagnoses" not in patient:
        normalized["diagnoses"] = patient["comorbidities"]
        normalized["conditions"] = patient["comorbidities"]

    # Map current_medications -> medications
    if "current_medications" in patient and "medications" not in patient:
        normalized["medications"] = patient["current_medications"]

    # Create labs dict if hba1c/egfr are at top level
    if "labs" not in patient:
        normalized["labs"] = {}
        if "hba1c" in patient:
            normalized["labs"]["hba1c"] = patient["hba1c"]
        if "egfr" in patient:
            normalized["labs"]["egfr"] = patient["egfr"]

    return normalized


def run_turbo_batch(patients: list, trial_protocol: str, trial_id: str, progress_bar, status_text) -> list:
    """
    Run ULTIMATE TURBO parallel batch screening with FULL EXPLAINABILITY.
    Processes all patients simultaneously with detailed criterion-by-criterion analysis.
    300 patients in ~60-90 seconds with complete audit trail.
    """
    try:
        from src.agents.supervisor_ultimate_turbo import UltimateTurboSupervisorAgent
        agent = UltimateTurboSupervisorAgent(batch_size=25, max_concurrent=10, enable_calibration=True)

        # Track progress with Streamlit-safe updates
        progress_state = {"completed": 0, "total": len(patients)}

        def progress_callback(completed: int, total: int, message: str):
            progress_state["completed"] = completed
            progress_state["total"] = total
            if progress_bar:
                pct = completed / total if total > 0 else 0
                progress_bar.progress(pct, text=message)
            if status_text:
                status_text.info(message)

        # Normalize patient data for agent compatibility
        normalized_patients = [normalize_patient_data(p) for p in patients]

        async def _async_batch():
            return await agent.batch_screen_parallel_advanced(
                patients=normalized_patients,
                trial_protocol=trial_protocol,
                trial_id=trial_id,
                progress_callback=progress_callback
            )

        # Streamlit-compatible event loop handling
        # Create a fresh event loop for each batch run
        try:
            # Try to get existing loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Loop closed")
        except RuntimeError:
            # Create new loop if none exists or is closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(_async_batch())
        except Exception as e:
            # If loop fails, try with completely fresh loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(_async_batch())

        return results

    except Exception as e:
        st.error(f"Turbo batch error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []


# =============================================================================
# SIDEBAR - API STATUS & CONTROLS
# =============================================================================

api_key = get_api_key()

# API Status indicator (small green/red dot)
if api_key:
    st.sidebar.markdown(
        '<div class="api-status api-status-ok"><span class="green-dot"></span> API Ready</div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<div class="api-status api-status-error"><span class="red-dot"></span> API not configured</div>',
        unsafe_allow_html=True
    )
    user_key = st.sidebar.text_input("Enter API Key", type="password", key="api_input")
    if user_key:
        st.session_state.google_api_key = user_key
        os.environ["GOOGLE_API_KEY"] = user_key
        st.rerun()

# Developer Login
st.sidebar.markdown("---")
if not is_developer_mode():
    with st.sidebar.expander("Developer Login"):
        dev_user = st.text_input("Username", key="dev_user")
        dev_pass = st.text_input("Password", type="password", key="dev_pass")
        if st.button("Login", key="dev_login"):
            if check_developer_login(dev_user, dev_pass):
                st.session_state.developer_logged_in = True
                st.success("Developer mode activated!")
                st.rerun()
            else:
                st.error("Invalid credentials")
else:
    st.sidebar.markdown('<span class="dev-badge">Developer Mode</span>', unsafe_allow_html=True)
    if st.sidebar.button("Logout Developer"):
        st.session_state.developer_logged_in = False
        st.rerun()

# Clear & New Patient button
st.sidebar.markdown("---")
if st.sidebar.button("Clear & New Patient", use_container_width=True):
    clear_session()
    st.rerun()

# Trial History
st.sidebar.markdown("---")
st.sidebar.subheader("Trial Selection")

trial_history = get_trial_history()
if trial_history:
    st.sidebar.caption("Recent Trials:")
    for i, trial in enumerate(trial_history[:8]):
        trial_id_hist = trial.get("trial_id", "Unknown")
        if st.sidebar.button(f"{trial_id_hist}", key=f"hist_{i}", use_container_width=True):
            st.session_state.selected_trial_id = trial_id_hist
            st.rerun()

# Trial input with auto-fetch from ClinicalTrials.gov
default_trial = st.session_state.get("selected_trial_id", "")
trial_id = st.sidebar.text_input("Trial ID", value=default_trial, placeholder="NCT12345678")

# Initialize fetched protocol in session state
if "fetched_protocol" not in st.session_state:
    st.session_state.fetched_protocol = ""
if "fetched_trial_info" not in st.session_state:
    st.session_state.fetched_trial_info = None

# Auto-fetch button
col1, col2 = st.sidebar.columns([2, 1])
with col1:
    fetch_btn = st.button("Fetch from ClinicalTrials.gov", key="fetch_trial", use_container_width=True)
with col2:
    clear_fetch = st.button("Clear", key="clear_fetch")

if clear_fetch:
    st.session_state.fetched_protocol = ""
    st.session_state.fetched_trial_info = None
    st.rerun()

if fetch_btn and trial_id:
    with st.sidebar:
        with st.spinner(f"Fetching {trial_id}..."):
            trial_data = fetch_trial_from_clinicaltrials_gov(trial_id)

            if trial_data and "error" not in trial_data:
                st.session_state.fetched_protocol = trial_data.get("protocol_text", "")
                st.session_state.fetched_trial_info = trial_data
                st.success(f"Fetched: {trial_data.get('title', '')[:50]}...")
            elif trial_data and "error" in trial_data:
                st.error(trial_data["error"])
            else:
                st.error(f"Could not fetch {trial_id}")

# Show fetched trial info
if st.session_state.fetched_trial_info:
    info = st.session_state.fetched_trial_info
    with st.sidebar.expander("Trial Info", expanded=False):
        st.markdown(f"**Status:** {info.get('status', 'N/A')}")
        st.markdown(f"**Conditions:** {', '.join(info.get('conditions', []))[:100]}")
        st.markdown(f"**Age:** {info.get('min_age', 'N/A')} - {info.get('max_age', 'N/A')}")
        st.markdown(f"**Sex:** {info.get('sex', 'All')}")

# Protocol text area (pre-filled if fetched)
default_protocol = st.session_state.get("fetched_protocol", "")
trial_protocol = st.sidebar.text_area(
    "Protocol",
    value=default_protocol,
    height=150,
    placeholder="Enter Trial ID above and click 'Fetch' to auto-fill, or paste protocol manually"
)


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("Clinical Trial Screening")
st.caption("**ULTIMATE TURBO** - Speed + Full Explainability")
st.markdown("**Optimized AI System** for rapid patient-trial matching with explainable decisions.")

# Create tabs based on developer mode
if is_developer_mode():
    tab1, tab2, tab3, tab4 = st.tabs(["Patient Form", "Batch Processing", "JSON Input (Dev)", "Results"])
else:
    tab1, tab2, tab3 = st.tabs(["Patient Form", "Batch Processing", "Results"])


# =============================================================================
# TAB 1: PATIENT FORM
# =============================================================================

with tab1:
    st.header("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        patient_id = st.text_input("Patient ID", value="PT001")
        age = st.number_input("Age", min_value=0, max_value=120, value=55)
        sex = st.selectbox("Sex", ["male", "female", "other"])

        st.subheader("Diagnoses")
        diagnosis = st.text_input("Primary Condition", placeholder="Type 2 Diabetes Mellitus")
        icd10 = st.text_input("ICD-10 Code", placeholder="E11.9")

    with col2:
        st.subheader("Medications")
        medication = st.text_input("Medication", placeholder="Metformin")
        dose = st.text_input("Dose", placeholder="1000mg twice daily")

        st.subheader("Lab Values")
        lab_test = st.text_input("Test Name", placeholder="HbA1c")
        lab_value = st.number_input("Value", value=8.0, format="%.1f")
        lab_unit = st.text_input("Unit", placeholder="%")

    # Other Information field
    st.subheader("Other Information")
    other_info = st.text_area(
        "Additional Notes",
        placeholder="Enter any other relevant patient information, medical history, allergies, lifestyle factors, etc.",
        height=100
    )

    if st.button("Load Patient", type="primary", key="build"):
        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "diagnoses": [{"condition": diagnosis, "icd10": icd10}] if diagnosis else [],
            "medications": [{"drug_name": medication, "dose": dose}] if medication else [],
            "lab_values": [{"test": lab_test, "value": lab_value, "unit": lab_unit}] if lab_test else [],
            "other_information": other_info if other_info else None
        }

        # Backend validation
        is_valid, msg = validate_patient_data(patient_data)
        if is_valid:
            st.session_state.patient_data = patient_data
            st.session_state.patient_validated = True
            st.session_state.patients_source = "single"
            st.success(f"Patient {patient_id} loaded successfully!")
        else:
            st.error(f"Validation error: {msg}")


# =============================================================================
# TAB 2: BATCH PROCESSING
# =============================================================================

with tab2:
    st.header("Batch Processing")
    st.markdown("Upload your patient files and clinical trial protocols for batch screening.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Files")
        patients_file = st.file_uploader(
            "Upload Patients (CSV or JSON)",
            type=["csv", "json"],
            help="CSV with columns: patient_id, age, sex, diagnoses, medications, lab_values. Or JSON array of patient objects."
        )

        if patients_file:
            try:
                if patients_file.name.endswith(".csv"):
                    patients_df = pd.read_csv(patients_file)
                    st.success(f"Loaded {len(patients_df)} patients from CSV")
                    st.dataframe(patients_df, use_container_width=True, height=300)
                    patients_list = patients_df.to_dict('records')
                else:
                    patients_list = json.loads(patients_file.read().decode('utf-8'))
                    if isinstance(patients_list, dict):
                        patients_list = [patients_list]
                    st.success(f"Loaded {len(patients_list)} patients from JSON")

                st.session_state.batch_patients = patients_list
                st.session_state.patients_source = "batch"
            except Exception as e:
                st.error(f"Error loading patients file: {e}")

    with col2:
        st.subheader("Clinical Trial Protocols")
        protocols_file = st.file_uploader(
            "Upload Protocols (CSV, JSON, or TXT)",
            type=["csv", "json", "txt", "md"],
            help="CSV with columns: trial_id, protocol_text. Or JSON with trial_id keys. Or single protocol text file."
        )

        if protocols_file:
            try:
                content = protocols_file.read().decode('utf-8')
                if protocols_file.name.endswith(".csv"):
                    import io
                    protocols_df = pd.read_csv(io.StringIO(content))
                    protocols_dict = dict(zip(protocols_df['trial_id'], protocols_df['protocol_text']))
                    st.success(f"Loaded {len(protocols_dict)} protocols from CSV")
                elif protocols_file.name.endswith(".json"):
                    protocols_dict = json.loads(content)
                    st.success(f"Loaded {len(protocols_dict)} protocols from JSON")
                else:
                    protocols_dict = {"UPLOADED": content}
                    st.success("Loaded protocol text file")

                st.session_state.uploaded_protocols = protocols_dict
            except Exception as e:
                st.error(f"Error loading protocols file: {e}")

    st.markdown("---")

    batch_trial_id = st.text_input("Default Trial ID for Batch", value="NCT12345678", key="batch_trial")

    # Update session state when trial ID changes
    if batch_trial_id:
        st.session_state.batch_trial_id_selected = batch_trial_id

    col1, col2, col3 = st.columns(3)

    with col1:
        run_batch = st.button("Run Batch Screening", type="primary", use_container_width=True)

    with col2:
        if st.button("Clear Batch Results", use_container_width=True):
            st.session_state.batch_results = []
            st.rerun()

    with col3:
        if st.session_state.batch_results:
            df_results = pd.DataFrame(st.session_state.batch_results)
            csv = df_results.to_csv(index=False)
            st.download_button(
                "Export Results CSV",
                csv,
                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

    if run_batch:
        patients_to_process = st.session_state.get("batch_patients", [])
        protocols = st.session_state.get("uploaded_protocols", {})

        if not patients_to_process:
            st.error("Please upload a patients file first")
        elif not api_key:
            st.error("API Key required for screening")
        else:
            # Validate all patients first
            valid_patients = []
            validation_errors = []

            for i, patient in enumerate(patients_to_process):
                is_valid, msg = validate_patient_data(patient)
                if is_valid:
                    valid_patients.append(patient)
                else:
                    validation_errors.append({
                        "patient_id": patient.get("patient_id", f"Patient_{i}"),
                        "trial_id": batch_trial_id,
                        "decision": "ERROR",
                        "confidence": 0,
                        "error": msg
                    })

            # Get protocol
            protocol = protocols.get(batch_trial_id) or protocols.get("UPLOADED") or f"""
            CLINICAL TRIAL: {batch_trial_id}
            INCLUSION: Age 18-75, Type 2 Diabetes, HbA1c 7-10%
            EXCLUSION: Type 1 Diabetes, Pregnancy, Severe renal impairment
            """

            # TURBO PARALLEL PROCESSING
            st.info(f"🚀 TURBO MODE: Processing {len(valid_patients)} patients in parallel...")
            progress_bar = st.progress(0, text="Initializing turbo batch...")
            status = st.empty()

            start_time = time.time()

            # Run turbo batch processing
            turbo_results = run_turbo_batch(
                patients=valid_patients,
                trial_protocol=protocol,
                trial_id=batch_trial_id,
                progress_bar=progress_bar,
                status_text=status
            )

            elapsed = time.time() - start_time

            # Format results for display
            formatted_results = []
            for r in turbo_results:
                formatted_results.append({
                    "patient_id": r.get("patient_id"),
                    "trial_id": r.get("trial_id"),
                    "decision": r.get("decision", "UNKNOWN"),
                    "confidence": r.get("confidence", 0),
                    "narrative": f"{r.get('key_match', '')} {r.get('key_concern', '')}".strip()[:100]
                })

            # Combine validation errors with results
            st.session_state.batch_results = validation_errors + formatted_results

            progress_bar.progress(100, text="Complete!")
            status.success(f"✅ TURBO Complete! {len(valid_patients)} patients in {elapsed:.1f}s ({len(valid_patients)/elapsed:.1f} patients/sec)")

    if st.session_state.batch_results:
        st.subheader("Batch Results")
        df = pd.DataFrame(st.session_state.batch_results)
        # Scrollable dataframe with dynamic height based on number of results
        display_height = min(600, max(200, len(df) * 35 + 50))
        st.dataframe(df, use_container_width=True, height=display_height)

        col1, col2, col3, col4 = st.columns(4)
        decisions = [r.get("decision") for r in st.session_state.batch_results]
        with col1:
            st.metric("Eligible", decisions.count("ELIGIBLE"))
        with col2:
            st.metric("Ineligible", decisions.count("INELIGIBLE"))
        with col3:
            st.metric("Uncertain", decisions.count("UNCERTAIN"))
        with col4:
            st.metric("Errors", decisions.count("ERROR"))


# =============================================================================
# TAB 3 (DEV ONLY): JSON INPUT
# =============================================================================

if is_developer_mode():
    with tab3:
        st.header("Developer JSON Input")
        st.warning("This tab is only visible in Developer Mode")

        json_template = """{
    "patient_id": "PT001",
    "age": 58,
    "sex": "male",
    "diagnoses": [{"condition": "Type 2 Diabetes Mellitus", "icd10": "E11.9"}],
    "medications": [{"drug_name": "Metformin", "dose": "1000mg twice daily"}],
    "lab_values": [{"test": "HbA1c", "value": 8.2, "unit": "%"}],
    "other_information": "No known allergies"
}"""

        json_input = st.text_area("Patient JSON", value=json_template, height=300)

        if st.button("Parse & Load JSON", key="parse_json"):
            try:
                parsed_data = json.loads(json_input)
                is_valid, msg = validate_patient_data(parsed_data)
                if is_valid:
                    st.session_state.patient_data = parsed_data
                    st.session_state.patient_validated = True
                    st.success(f"Patient {parsed_data.get('patient_id', 'N/A')} loaded!")
                    st.json(parsed_data)
                else:
                    st.error(f"Validation error: {msg}")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

        if st.session_state.patient_data:
            st.subheader("Current Patient Data (JSON)")
            st.json(st.session_state.patient_data)


# =============================================================================
# RESULTS TAB
# =============================================================================

results_tab = tab4 if is_developer_mode() else tab3

with results_tab:
    st.header("Run Screening & Results")

    # Determine patient status - check both single patient and batch patients
    batch_patients_count = len(st.session_state.get("batch_patients", []))
    single_patient = st.session_state.get("patient_data")
    patients_source = st.session_state.get("patients_source")

    # Determine trial status - check both sidebar trial_id and batch trial_id
    effective_trial_id = trial_id or st.session_state.get("batch_trial_id_selected", "") or st.session_state.get("selected_trial_id", "")

    col1, col2 = st.columns(2)
    with col1:
        if patients_source == "batch" and batch_patients_count > 0:
            st.success(f"Multiple Patients Loaded ({batch_patients_count} patients)")
        elif patients_source == "single" and single_patient:
            st.success(f"Patient loaded: {single_patient.get('patient_id', 'N/A')}")
        elif single_patient:
            st.success(f"Patient loaded: {single_patient.get('patient_id', 'N/A')}")
        elif batch_patients_count > 0:
            st.success(f"Multiple Patients Loaded ({batch_patients_count} patients)")
        else:
            st.warning("No patient loaded")
    with col2:
        if effective_trial_id:
            st.success(f"Trial: {effective_trial_id}")
        else:
            st.warning("No trial selected")

    progress_container = st.container()

    # Determine if we have batch or single patient data
    has_batch_data = batch_patients_count > 0
    has_single_data = single_patient is not None
    has_batch_results = len(st.session_state.get("batch_results", [])) > 0

    # Show batch results if they exist (from Batch Processing tab)
    if has_batch_results and has_batch_data:
        st.info(f"Batch screening already completed for {len(st.session_state.batch_results)} patients. See results below or go to 'Batch Processing' tab for details.")

    if st.button("Run Eligibility Screening", type="primary", use_container_width=True):
        if not api_key:
            st.error("API Key required")
        elif not has_single_data and not has_batch_data:
            st.error("Please load patient data first (single patient in 'Patient Form' or batch in 'Batch Processing')")
        elif not effective_trial_id:
            st.error("Please enter a trial ID")
        else:
            start_time = time.time()

            # Get protocol - either from sidebar, fetched, or default
            protocols = st.session_state.get("uploaded_protocols", {})
            protocol = trial_protocol if trial_protocol else protocols.get(effective_trial_id) or protocols.get("UPLOADED") or f"""
            CLINICAL TRIAL: {effective_trial_id}
            INCLUSION CRITERIA:
            1. Age 18-75 years
            2. Diagnosis of Type 2 Diabetes Mellitus
            3. HbA1c between 7.0% and 10.0%
            4. Currently on stable metformin therapy
            EXCLUSION CRITERIA:
            1. Type 1 Diabetes
            2. Pregnant or nursing women
            3. Severe renal impairment (eGFR < 30 mL/min)
            """

            if has_batch_data and not has_single_data:
                # Run TURBO batch screening for multiple patients
                st.info(f"Running TURBO batch screening for {batch_patients_count} patients...")
                progress_bar = st.progress(0, text="Initializing turbo batch...")
                status = st.empty()

                # Validate patients
                batch_patients = st.session_state.get("batch_patients", [])
                valid_patients = []
                for patient in batch_patients:
                    is_valid, msg = validate_patient_data(patient)
                    if is_valid:
                        valid_patients.append(patient)

                turbo_results = run_turbo_batch(
                    patients=valid_patients,
                    trial_protocol=protocol,
                    trial_id=effective_trial_id,
                    progress_bar=progress_bar,
                    status_text=status
                )

                elapsed = time.time() - start_time

                # Format results
                formatted_results = []
                for r in turbo_results:
                    formatted_results.append({
                        "patient_id": r.get("patient_id"),
                        "trial_id": r.get("trial_id"),
                        "decision": r.get("decision", "UNKNOWN"),
                        "confidence": r.get("confidence", 0),
                        "narrative": f"{r.get('key_match', '')} {r.get('key_concern', '')}".strip()[:100]
                    })

                st.session_state.batch_results = formatted_results
                progress_bar.progress(100, text="Complete!")
                status.success(f"TURBO Complete! {len(valid_patients)} patients in {elapsed:.1f}s")

            else:
                # Run single patient screening
                result = run_fast_screening(
                    st.session_state.patient_data,
                    protocol,
                    effective_trial_id,
                    progress_container
                )

                elapsed = time.time() - start_time

                if result:
                    result["elapsed_time"] = f"{elapsed:.1f}s"
                    st.session_state.screening_result = result
                    save_trial_to_db(effective_trial_id, protocol, result)

    if st.session_state.screening_result:
        result = st.session_state.screening_result
        st.divider()

        decision = result.get("decision", "UNCERTAIN")
        confidence = result.get("confidence", 0.0)
        elapsed = result.get("elapsed_time", "N/A")

        st.markdown(
            f'<div class="{get_eligibility_class(decision)}">{decision}</div>',
            unsafe_allow_html=True
        )

        st.caption(f"Completed in {elapsed}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Confidence", f"{confidence:.0%}")
        with col2:
            st.metric("Level", result.get("confidence_level", "N/A"))
        with col3:
            review = "Yes" if result.get("requires_human_review") else "No"
            st.metric("Human Review", review)
        with col4:
            st.metric("Trial", effective_trial_id)

        res_tab1, res_tab2, res_tab3 = st.tabs(["Analysis", "Explainability", "Narrative"])

        with res_tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = create_confidence_gauge(confidence)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Key Factors:**")
                for factor in result.get("key_factors", []):
                    st.markdown(f"- {factor}")
                if result.get("concerns"):
                    st.markdown("**Concerns:**")
                    for concern in result.get("concerns", []):
                        st.markdown(f"- {concern}")

        with res_tab2:
            data = result.get("explainability_table", [])
            if data:
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.info("No detailed explainability data available")

        with res_tab3:
            narrative = result.get("clinical_narrative", "")
            if narrative:
                st.markdown(narrative)
            else:
                st.info("No narrative generated")

    # Display batch results if available (from batch processing)
    if st.session_state.get("batch_results") and (has_batch_data or patients_source == "batch"):
        st.divider()
        st.subheader("ULTIMATE TURBO Batch Results")

        batch_results = st.session_state.batch_results

        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        decisions = [r.get("decision") for r in batch_results]
        with col1:
            st.metric("Eligible", decisions.count("ELIGIBLE"), delta=None)
        with col2:
            st.metric("Ineligible", decisions.count("INELIGIBLE"), delta=None)
        with col3:
            st.metric("Uncertain", decisions.count("UNCERTAIN"), delta=None)
        with col4:
            st.metric("Errors", decisions.count("ERROR"), delta=None)
        with col5:
            review_count = sum(1 for r in batch_results if r.get("requires_human_review"))
            st.metric("Need Review", review_count, delta=None)

        # Create display dataframe with key columns
        display_data = []
        for r in batch_results:
            display_data.append({
                "Patient ID": r.get("patient_id", "N/A"),
                "Decision": r.get("decision", "N/A"),
                "Confidence": f"{r.get('confidence', 0):.0%}",
                "Level": r.get("confidence_level", "N/A"),
                "Review": "Yes" if r.get("requires_human_review") else "No",
                "Key Factors": ", ".join(r.get("key_factors", [])[:2]) if r.get("key_factors") else "N/A",
                "Concerns": ", ".join(r.get("concerns", [])[:2]) if r.get("concerns") else "None"
            })
        df = pd.DataFrame(display_data)

        # Display results table
        display_height = min(400, max(200, len(df) * 35 + 50))
        st.dataframe(df, use_container_width=True, height=display_height)

        # Advanced Explainability Section
        st.divider()
        st.subheader("Detailed Patient Analysis")

        # Patient selector for detailed view
        patient_ids = [r.get("patient_id", f"Patient {i}") for i, r in enumerate(batch_results)]
        selected_patient = st.selectbox("Select patient for detailed analysis:", patient_ids)

        if selected_patient:
            # Find the selected patient's result
            patient_result = next((r for r in batch_results if r.get("patient_id") == selected_patient), None)

            if patient_result:
                # Decision banner
                decision = patient_result.get("decision", "UNKNOWN")
                confidence = patient_result.get("confidence", 0)
                decision_color = {"ELIGIBLE": "#28a745", "INELIGIBLE": "#dc3545", "UNCERTAIN": "#ffc107", "ERROR": "#6c757d"}.get(decision, "#6c757d")

                st.markdown(f"""
                <div style="background-color: {decision_color}; color: white; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px;">
                    <h2 style="margin: 0;">{decision}</h2>
                    <p style="margin: 5px 0;">Confidence: {confidence:.0%} ({patient_result.get('confidence_level', 'N/A')})</p>
                </div>
                """, unsafe_allow_html=True)

                # Tabs for different views
                detail_tab1, detail_tab2, detail_tab3 = st.tabs(["Matching Results", "Narrative", "Audit Trail"])

                with detail_tab1:
                    matching_results = patient_result.get("matching_results", [])
                    if matching_results:
                        st.markdown("**Criterion-by-Criterion Analysis:**")
                        for i, match in enumerate(matching_results):
                            status = match.get("match_status", "UNKNOWN")
                            status_emoji = {"MATCH": "✅", "NO_MATCH": "❌", "UNCERTAIN": "⚠️", "MISSING_DATA": "❓"}.get(status, "❓")

                            with st.expander(f"{status_emoji} {match.get('criterion_id', f'Criterion {i+1}')} - {status}"):
                                st.write(f"**Criterion:** {match.get('criterion_text', 'N/A')}")
                                st.write(f"**Type:** {match.get('type', 'N/A')}")
                                st.write(f"**Confidence:** {match.get('confidence', 0):.0%}")
                                st.write(f"**Patient Data:** {match.get('patient_data_used', 'N/A')}")
                                st.write(f"**Reasoning:** {match.get('reasoning', 'N/A')}")
                                if match.get("concerns"):
                                    st.warning(f"Concerns: {', '.join(match.get('concerns', []))}")
                    else:
                        st.info("No detailed matching results available")

                with detail_tab2:
                    narrative = patient_result.get("clinical_narrative", "")
                    if narrative:
                        st.markdown(narrative)
                    else:
                        st.info("No narrative generated")

                    # Key factors and concerns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Key Factors:**")
                        for factor in patient_result.get("key_factors", []):
                            st.markdown(f"- {factor}")
                    with col2:
                        if patient_result.get("concerns"):
                            st.markdown("**Concerns:**")
                            for concern in patient_result.get("concerns", []):
                                st.markdown(f"- {concern}")

                with detail_tab3:
                    if is_developer_mode():
                        audit = patient_result.get("audit_trail", {})
                        if audit:
                            st.json(audit)
                        else:
                            st.info("No audit trail available")
                    else:
                        st.info("Audit trail available in Developer Mode")

        # Export buttons
        st.divider()
        csv = df.to_csv(index=False)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Export Summary CSV",
                csv,
                f"batch_results_{effective_trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "Export Full Results JSON",
                json.dumps(batch_results, indent=2),
                f"batch_results_full_{effective_trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        with col3:
            if is_developer_mode():
                with st.expander("Raw JSON (Dev)"):
                    st.json(batch_results)


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown("""
**Clinical Trial Screening ULTIMATE TURBO** | v4.0.0 | Speed + Explainability

2026 CodeNoLimits - Melea & David
""")
