
import textwrap

def get_css(theme: str) -> str:
    """
    Returns the CSS string for the selected theme.
    
    Themes:
    - "Standard": Clean, professional medical UI (Light)
    - "Midnight": High-contrast dark mode
    - "Neo-Glass": Premium, translucent glassmorphism with animations
    """
    
    base_css = textwrap.dedent("""
    <style>
    /* Global Reset & Base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        transition: background 0.5s ease;
    }

    /* Keyframe Animations */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(111, 66, 193, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(111, 66, 193, 0); }
        100% { box-shadow: 0 0 0 0 rgba(111, 66, 193, 0); }
    }
    </style>
    """)
    
    themes = {
        "Standard": textwrap.dedent("""
        <style>
        .stApp {
            background-color: #f8f9fa;
        }
        
        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .stButton button {
            background-color: #0d6efd;
            color: white;
            border-radius: 8px;
            border: none;
            transition: all 0.2s;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(13, 110, 253, 0.2);
        }
        </style>
        """),
        
        "Midnight": textwrap.dedent("""
        <style>
        .stApp {
            background-color: #0f111a;
            color: #e0e0e0;
        }
        
        h1, h2, h3, h4, h5, h6, label, .stMarkdown, p {
            color: #e0e0e0 !important;
        }
        
        .stTextInput input, .stNumberInput input, .stSelectbox, .stTextArea textarea {
            background-color: #1a1d2d;
            color: white;
            border: 1px solid #2d3248;
        }
        
        div[data-testid="stExpander"] {
            background-color: #1a1d2d;
            border: 1px solid #2d3248;
        }
        </style>
        """),
        
        "Neo-Glass": textwrap.dedent("""
        <style>
        /* GLASSMORPHISM BACKGROUND */
        .stApp {
            background: linear-gradient(-45deg, #e0c3fc, #8ec5fc, #90f2ff, #f3c4fb);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* GLASS CONTAINERS */
        .stMarkdown, div[data-testid="stExpander"], .stForm, div[data-testid="stVerticalBlock"] > div {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.4);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        
        /* SIDEBAR GLASS */
        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.3);
        }

        /* INPUTS */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea {
            background: rgba(255, 255, 255, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.5) !important;
            border-radius: 12px !important;
            color: #333 !important;
            transition: all 0.3s ease;
        }
        
        .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
            background: rgba(255, 255, 255, 0.7) !important;
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.6);
            transform: scale(1.01);
        }

        /* BUTTONS */
        .stButton button {
            background: linear-gradient(135deg, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0.1) 100%);
            border: 1px solid rgba(255, 255, 255, 0.5);
            backdrop-filter: blur(4px);
            color: #2c3e50;
            font-weight: 600;
            border-radius: 30px;
            padding: 0.5rem 1.5rem;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            animation: slideUp 0.6s ease-out backwards;
        }
        
        .stButton button:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            background: rgba(255, 255, 255, 0.6);
            border-color: #fff;
        }
        
        .stButton button:active {
            transform: translateY(-1px);
        }

        /* Primary Button "Wow" Effect */
        button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4);
            animation: pulseGlow 2s infinite;
        }

        /* ANIMATED HEADERS */
        h1 {
            background: linear-gradient(to right, #30cfd0 0%, #330867 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            letter-spacing: -1px;
            animation: fadeIn 1.2s ease-out;
        }
        
        h2, h3 {
            color: #4a4a4a;
            animation: slideUp 0.8s ease-out;
        }

        /* CARDS / CONTAINERS ANIMATION */
        .element-container, .stForm {
            animation: slideUp 0.5s ease-out backwards;
        }
        .element-container:nth-child(1) { animation-delay: 0.1s; }
        .element-container:nth-child(2) { animation-delay: 0.2s; }
        .element-container:nth-child(3) { animation-delay: 0.3s; }
        
        </style>
        """)
    }
    
    return base_css + themes.get(theme, themes["Standard"])
