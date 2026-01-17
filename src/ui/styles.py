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

        /* Selectbox */
        .stSelectbox div[data-baseweb="select"] {
            background-color: #161b22 !important;
            border-color: #30363d !important;
        }

        .stSelectbox div[data-baseweb="select"] span {
            color: #e6edf3 !important;
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

        /* ALL TEXT */
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

        /* Input fields */
        .stTextInput input, .stNumberInput input, .stTextArea textarea {
            background-color: #f6f8fa !important;
            border: 1px solid #d0d7de !important;
            border-radius: 6px;
            color: #1f2328 !important;
        }

        .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
            border-color: #0969da !important;
            box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.15) !important;
        }

        /* Selectbox */
        .stSelectbox div[data-baseweb="select"] {
            background-color: #f6f8fa !important;
            border-color: #d0d7de !important;
        }

        .stSelectbox div[data-baseweb="select"] span {
            color: #1f2328 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f6f8fa !important;
            border-right: 1px solid #d0d7de !important;
        }

        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #1f2328 !important;
        }

        /* Tabs */
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

        /* Buttons */
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
        }

        /* Dividers */
        hr {
            border-color: #d0d7de !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background-color: #f6f8fa !important;
            border: 1px dashed #d0d7de !important;
        }

        [data-testid="stFileUploader"] label {
            color: #1f2328 !important;
        }
        </style>
        """)
    }

    return themes.get(theme, themes["Midnight Pro"])
