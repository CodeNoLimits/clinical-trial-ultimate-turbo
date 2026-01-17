import textwrap

def get_css(theme: str) -> str:
    """
    Returns the CSS string for the selected theme.

    Themes:
    - "Midnight Pro": Clean dark theme with readable text
    - "Clinical Clean": Professional light theme
    """

    themes = {
        "Midnight Pro": textwrap.dedent("""
        <style>
        /* MIDNIGHT PRO - DARK THEME */

        /* Main App Background */
        .stApp {
            background-color: #0d1117;
        }

        /* ALL TEXT - Make everything readable */
        .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
            color: #e6edf3 !important;
        }

        /* Headers */
        .stApp h1 {
            color: #ffffff !important;
            font-weight: 600;
            font-size: 2.2rem;
        }

        .stApp h2 {
            color: #ffffff !important;
            font-weight: 600;
            font-size: 1.5rem;
        }

        .stApp h3 {
            color: #e6edf3 !important;
            font-weight: 500;
            font-size: 1.2rem;
        }

        /* Markdown text */
        .stMarkdown, .stMarkdown p, .stMarkdown span {
            color: #e6edf3 !important;
        }

        /* Captions */
        .stApp small, .stApp .caption, [data-testid="stCaptionContainer"] {
            color: #8b949e !important;
        }

        /* Labels above inputs */
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stTextArea label,
        .stRadio label, .stCheckbox label, .stMultiSelect label {
            color: #e6edf3 !important;
            font-weight: 500;
        }

        /* Input fields */
        .stTextInput input, .stNumberInput input, .stTextArea textarea {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px;
            color: #e6edf3 !important;
        }

        .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
            border-color: #58a6ff !important;
            box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15) !important;
        }

        /* Selectbox - all parts */
        .stSelectbox > div > div {
            background-color: #161b22 !important;
            border-color: #30363d !important;
        }

        .stSelectbox [data-baseweb="select"] > div {
            background-color: #161b22 !important;
            border-color: #30363d !important;
        }

        .stSelectbox [data-baseweb="select"] span {
            color: #e6edf3 !important;
        }

        /* Dropdown menu */
        [data-baseweb="popover"] {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
        }

        [data-baseweb="menu"] {
            background-color: #161b22 !important;
        }

        [data-baseweb="menu"] li {
            background-color: #161b22 !important;
            color: #e6edf3 !important;
        }

        [data-baseweb="menu"] li:hover {
            background-color: #30363d !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #161b22 !important;
        }

        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #e6edf3 !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            color: #8b949e !important;
        }

        .stTabs [aria-selected="true"] {
            color: #58a6ff !important;
            border-bottom-color: #58a6ff !important;
        }

        /* Buttons */
        .stButton button {
            background-color: #21262d !important;
            color: #e6edf3 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px;
            font-weight: 500;
        }

        .stButton button:hover {
            background-color: #30363d !important;
            border-color: #8b949e !important;
        }

        /* Primary Button */
        .stButton button[kind="primary"],
        button[kind="primary"],
        .stButton button[data-testid="baseButton-primary"] {
            background-color: #238636 !important;
            color: #ffffff !important;
            border: 1px solid #2ea043 !important;
        }

        .stButton button[kind="primary"]:hover,
        button[kind="primary"]:hover {
            background-color: #2ea043 !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
        }

        [data-testid="stMetricLabel"] {
            color: #8b949e !important;
        }

        /* Dataframes */
        .stDataFrame {
            border: 1px solid #30363d !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #161b22 !important;
            color: #e6edf3 !important;
        }

        /* Alerts */
        .stAlert {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
        }

        /* Dividers */
        hr {
            border-color: #30363d !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background-color: #161b22 !important;
            border: 1px dashed #30363d !important;
        }

        [data-testid="stFileUploader"] label {
            color: #e6edf3 !important;
        }
        </style>
        """),

        "Clinical Clean": textwrap.dedent("""
        <style>
        /* CLINICAL CLEAN - LIGHT THEME */

        /* Main App Background */
        .stApp {
            background-color: #ffffff;
        }

        /* ALL TEXT - Force dark text everywhere */
        .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
            color: #1f2328 !important;
        }

        /* Headers */
        .stApp h1 {
            color: #1f2328 !important;
            font-weight: 600;
            font-size: 2.2rem;
        }

        .stApp h2 {
            color: #1f2328 !important;
            font-weight: 600;
            font-size: 1.5rem;
        }

        .stApp h3 {
            color: #1f2328 !important;
            font-weight: 500;
            font-size: 1.2rem;
        }

        /* Markdown text */
        .stMarkdown, .stMarkdown p, .stMarkdown span {
            color: #1f2328 !important;
        }

        /* Captions */
        .stApp small, .stApp .caption, [data-testid="stCaptionContainer"] {
            color: #656d76 !important;
        }

        /* Labels above inputs */
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stTextArea label,
        .stRadio label, .stCheckbox label, .stMultiSelect label {
            color: #1f2328 !important;
            font-weight: 500;
        }

        /* ============================================ */
        /* INPUT FIELDS - COMPREHENSIVE TEXT FIX */
        /* ============================================ */

        /* All text inputs */
        .stTextInput input,
        .stTextInput input::placeholder,
        .stTextInput div input {
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
            border-radius: 6px;
            color: #1f2328 !important;
            -webkit-text-fill-color: #1f2328 !important;
        }

        /* Number inputs */
        .stNumberInput input,
        .stNumberInput div input {
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
            color: #1f2328 !important;
            -webkit-text-fill-color: #1f2328 !important;
        }

        /* Textareas */
        .stTextArea textarea,
        .stTextArea div textarea {
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
            border-radius: 6px;
            color: #1f2328 !important;
            -webkit-text-fill-color: #1f2328 !important;
        }

        /* Placeholder text */
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #656d76 !important;
            -webkit-text-fill-color: #656d76 !important;
            opacity: 1 !important;
        }

        /* Focus states */
        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stTextArea textarea:focus {
            border-color: #0969da !important;
            box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.15) !important;
            background-color: #ffffff !important;
            color: #1f2328 !important;
            -webkit-text-fill-color: #1f2328 !important;
        }

        /* ============================================ */
        /* SELECTBOX - COMPREHENSIVE FIX */
        /* ============================================ */

        .stSelectbox > div > div,
        .stSelectbox [data-baseweb="select"],
        .stSelectbox [data-baseweb="select"] > div,
        .stSelectbox div[data-baseweb="select"] {
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
            border-radius: 6px;
        }

        .stSelectbox [data-baseweb="select"] span,
        .stSelectbox [data-baseweb="select"] div,
        .stSelectbox span {
            color: #1f2328 !important;
            background-color: transparent !important;
        }

        .stSelectbox svg {
            fill: #1f2328 !important;
        }

        /* Dropdown menu for light theme */
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div {
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12) !important;
        }

        [data-baseweb="menu"],
        [data-baseweb="menu"] ul {
            background-color: #ffffff !important;
        }

        [data-baseweb="menu"] li,
        [data-baseweb="menu"] li div,
        [data-baseweb="menu"] li span {
            background-color: #ffffff !important;
            color: #1f2328 !important;
        }

        [data-baseweb="menu"] li:hover {
            background-color: #f6f8fa !important;
        }

        /* ============================================ */
        /* SIDEBAR - ALL ELEMENTS */
        /* ============================================ */

        section[data-testid="stSidebar"] {
            background-color: #f6f8fa !important;
            border-right: 1px solid #d0d7de !important;
        }

        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div {
            color: #1f2328 !important;
        }

        /* Sidebar inputs - FORCE DARK TEXT */
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] input {
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
            color: #1f2328 !important;
            -webkit-text-fill-color: #1f2328 !important;
        }

        section[data-testid="stSidebar"] .stTextArea textarea,
        section[data-testid="stSidebar"] textarea {
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
            color: #1f2328 !important;
            -webkit-text-fill-color: #1f2328 !important;
        }

        section[data-testid="stSidebar"] input::placeholder,
        section[data-testid="stSidebar"] textarea::placeholder {
            color: #656d76 !important;
            -webkit-text-fill-color: #656d76 !important;
            opacity: 1 !important;
        }

        /* Sidebar selectbox */
        section[data-testid="stSidebar"] .stSelectbox > div > div,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
        }

        section[data-testid="stSidebar"] .stSelectbox span {
            color: #1f2328 !important;
        }

        /* ============================================ */
        /* TABS */
        /* ============================================ */

        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            color: #656d76 !important;
        }

        .stTabs [aria-selected="true"] {
            color: #0969da !important;
            border-bottom-color: #0969da !important;
        }

        /* ============================================ */
        /* BUTTONS */
        /* ============================================ */

        .stButton button {
            background-color: #f6f8fa !important;
            color: #1f2328 !important;
            border: 1px solid #d0d7de !important;
            border-radius: 6px;
            font-weight: 500;
        }

        .stButton button:hover {
            background-color: #eaeef2 !important;
            border-color: #afb8c1 !important;
        }

        /* Primary Button */
        .stButton button[kind="primary"],
        button[kind="primary"],
        .stButton button[data-testid="baseButton-primary"] {
            background-color: #1f883d !important;
            color: #ffffff !important;
            border: 1px solid #1a7f37 !important;
        }

        .stButton button[kind="primary"]:hover,
        button[kind="primary"]:hover {
            background-color: #1a7f37 !important;
        }

        /* ============================================ */
        /* FILE UPLOADER */
        /* ============================================ */

        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] > div,
        [data-testid="stFileUploader"] section {
            background-color: #ffffff !important;
            border: 2px dashed #d0d7de !important;
            border-radius: 6px;
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] div {
            color: #1f2328 !important;
        }

        [data-testid="stFileUploader"] small {
            color: #656d76 !important;
        }

        /* ============================================ */
        /* OTHER ELEMENTS */
        /* ============================================ */

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #1f2328 !important;
        }

        [data-testid="stMetricLabel"] {
            color: #656d76 !important;
        }

        /* Dataframes */
        .stDataFrame {
            border: 1px solid #d0d7de !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #f6f8fa !important;
            color: #1f2328 !important;
        }

        /* Alerts */
        .stAlert {
            background-color: #f6f8fa !important;
            border: 1px solid #d0d7de !important;
            color: #1f2328 !important;
        }

        /* Dividers */
        hr {
            border-color: #d0d7de !important;
        }

        /* Number input buttons */
        .stNumberInput button {
            background-color: #f6f8fa !important;
            border: 1px solid #d0d7de !important;
            color: #1f2328 !important;
        }

        .stNumberInput button:hover {
            background-color: #eaeef2 !important;
        }

        /* Info/Warning/Error boxes */
        .stInfo, .stWarning, .stError, .stSuccess {
            color: #1f2328 !important;
        }
        </style>
        """)
    }

    return themes.get(theme, themes["Midnight Pro"])
