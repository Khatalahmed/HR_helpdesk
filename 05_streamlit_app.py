"""
                                                                                                                                                                                                                                                
            SnailCloud Technologies     Enterprise HR Helpdesk RAG System            
            Step 5 : Answer Generation + Premium UI  (Gemini AI Studio)            
                                                                                                                                                                                                                                                

Run with:
    streamlit run 05_streamlit_app.py

Images (save these in the same folder as this file):
    images/snailcloud_office.jpg       SnailCloud Technologies office lobby image
    images/cloud_computing.jpg         Cloud computing visualization image
"""

import os
import base64
import requests
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

#        Environment                                                                                                                                                                                              
load_dotenv()

API_BASE_URL_DEFAULT = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY_DEFAULT = os.getenv("APP_API_KEY", "").strip()

SAMPLE_QUESTIONS = [
    "What is the salary credit date?",
    "What is the mediclaim sum insured for dependents?",
    "How does the promotion process work?",
    "How do I raise a payroll discrepancy?",
    "What are the DEI commitments of the company?",
    "How are Spot Awards taxed?",
    "What happens to health insurance when I resign?",
    "How many annual leaves do I get per year?",
    "What is the office working hours policy?",
    "Are there any upskilling or training benefits?",
]

#        Image Helpers                                                                                                                                                                                        
SCRIPT_DIR = Path(__file__).parent
IMAGE_DIR  = SCRIPT_DIR / "images"

def img_to_b64(filename: str) -> str | None:
    """Load an image from the images/ folder and return base64 string, or None."""
    path = IMAGE_DIR / filename
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def img_tag(filename: str, style: str = "") -> str:
    """Return an <img> HTML tag with base64 src, or empty string if file missing."""
    b64 = img_to_b64(filename)
    if b64:
        ext = filename.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        return f'<img src="data:{mime};base64,{b64}" style="{style}">'
    return ""

#                                                                                                                                                                                                                                                 
#                               PAGE CONFIG (must be first)                          
#                                                                                                                                                                                                                                                 
st.set_page_config(
    page_title="SnailCloud HR Helpdesk",
    page_icon="    ",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:hr@snailcloud.in",
        "About":    "###      SnailCloud HR Helpdesk\nPowered by RAG + Gemini 2.5 Flash",
    },
)

#                                                                                                                                                                                                                                                 
#                                GLOBAL CSS THEME                                    
#                                                                                                                                                                                                                                                 
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

:root {
    --primary:       #6C63FF;
    --primary-dark:  #4B44CC;
    --secondary:     #00D4AA;
    --accent:        #FF6B6B;
    --warning:       #FFB347;
    --gcloud:        #4285F4;
    --bg-dark:       #0F1117;
    --bg-card:       #1A1D2E;
    --bg-card2:      #1E2235;
    --border:        #2E3250;
    --text-primary:  #EAEAF4;
    --text-muted:    #8B8FA8;
    --gradient-1:    linear-gradient(135deg, #6C63FF 0%, #00D4AA 100%);
    --gradient-hero: linear-gradient(135deg, #1A1D2E 0%, #0F1117 50%, #1A1D2E 100%);
    --shadow-glow:   0 0 20px rgba(108,99,255,0.3);
    --shadow-cloud:  0 0 20px rgba(66,133,244,0.3);
    --radius:        12px;
    --radius-lg:     20px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary);
}
.stApp { background: var(--bg-dark); }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 3px; }

/*        Sidebar        */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/*        Office Image Banner        */
.office-image-banner {
    width: 100%;
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
    position: relative;
    max-height: 260px;
}
.office-image-banner img {
    width: 100%;
    height: 260px;
    object-fit: cover;
    object-position: center 30%;
    display: block;
    border-radius: var(--radius-lg);
}
.office-image-overlay {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    padding: 1.2rem 1.5rem;
    background: linear-gradient(to top, rgba(15,17,23,0.95) 0%, rgba(15,17,23,0.4) 70%, transparent 100%);
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
}

/*        Hero Banner        */
.hero-banner {
    background: var(--gradient-hero);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.8rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: "";
    position: absolute; bottom: -50%; left: 20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(66,133,244,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem; font-weight: 700;
    background: var(--gradient-1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; line-height: 1.2;
}
.hero-subtitle { font-size: 1rem; color: var(--text-muted); margin-top: 0.4rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(108,99,255,0.15); border: 1px solid rgba(108,99,255,0.3);
    color: #A29BFE; padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 500; margin-top: 0.8rem; margin-right: 0.5rem;
}
.hero-badge.cloud {
    background: rgba(66,133,244,0.15); border-color: rgba(66,133,244,0.3); color: #7BAAF7;
}

/*        Cloud Image Strip        */
.cloud-image-strip {
    width: 100%;
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border);
    position: relative;
    max-height: 120px;
}
.cloud-image-strip img {
    width: 100%;
    height: 120px;
    object-fit: cover;
    object-position: center 40%;
    display: block;
    opacity: 0.75;
    border-radius: var(--radius);
}
.cloud-image-label {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    text-align: center;
    padding: 0.4rem;
    background: linear-gradient(to top, rgba(15,17,23,0.85), transparent);
    font-size: 0.72rem; color: #8B8FA8; letter-spacing: 0.08em;
    text-transform: uppercase;
}

/*        Metric Cards        */
.metric-row {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;
}
.metric-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.1rem 1.3rem; text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-glow); }
.metric-card .icon { font-size: 1.8rem; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: var(--text-primary); margin: 0.2rem 0; }
.metric-card .label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card.purple { border-top: 3px solid var(--primary); }
.metric-card.teal   { border-top: 3px solid var(--secondary); }
.metric-card.red    { border-top: 3px solid var(--accent); }
.metric-card.blue   { border-top: 3px solid var(--gcloud); }

/*        Chat Messages     text visibility fix        */
[data-testid="stChatMessage"] {
    background: var(--bg-card2) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; margin-bottom: 0.8rem !important; padding: 1rem 1.2rem !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #EAEAF4 !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #EAEAF4 !important;
}

[data-testid="stChatInput"] {
    background: var(--bg-card) !important; border: 1.5px solid var(--primary) !important;
    border-radius: var(--radius) !important; color: var(--text-primary) !important;
}
[data-testid="stChatInput"]:focus-within { box-shadow: var(--shadow-glow) !important; }

/*        Buttons        */
.stButton > button {
    background: rgba(108,99,255,0.12) !important; color: #A29BFE !important;
    border: 1px solid rgba(108,99,255,0.3) !important; border-radius: 8px !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 0.35rem 0.7rem !important; transition: all 0.2s !important; width: 100%;
}
.stButton > button:hover {
    background: rgba(108,99,255,0.25) !important; border-color: var(--primary) !important;
    transform: translateX(2px) !important;
}


/*        Strong Button Contrast Override        */
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] > button {
    background-color: #222846 !important;
    color: #EAF0FF !important;
    border: 1px solid #5661A8 !important;
    opacity: 1 !important;
}

[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] > button:hover {
    background-color: #2B3360 !important;
    color: #FFFFFF !important;
}

[data-testid="stButton"] button:disabled,
[data-testid="stDownloadButton"] > button:disabled {
    background-color: rgba(120, 185, 255, 0.35) !important;
    color: #EAF4FF !important;
    border: 1px solid rgba(120, 185, 255, 0.75) !important;
    opacity: 1 !important;
}
/*        Source Cards        */
.source-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-left: 3px solid var(--secondary); border-radius: var(--radius);
    padding: 0.9rem 1rem; margin-bottom: 0.6rem;
}
.source-card .source-title { font-weight: 600; font-size: 0.88rem; color: var(--secondary); }
.source-card .source-section { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 0.5rem; }
.source-card .source-body {
    font-family: 'Fira Code', monospace; font-size: 0.78rem;
    color: var(--text-primary); line-height: 1.5; white-space: pre-wrap; opacity: 0.85;
}

/*        Info Pill        */
.info-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(0,212,170,0.1); border: 1px solid rgba(0,212,170,0.25);
    color: var(--secondary); padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 500;
}
.cloud-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(108,99,255,0.1); border: 1px solid rgba(108,99,255,0.25);
    color: #A29BFE; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 500; margin-left: 6px;
}


/*        Sidebar Controls: Force Blue, No White Blocks        */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
    background: #23355F !important;
    color: #EAF4FF !important;
    border: 1px solid #5E7EBE !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:hover {
    background: #2D4477 !important;
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] .stButton > button:disabled,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:disabled {
    background: #325088 !important;
    color: #DCEBFF !important;
    border: 1px solid #7EA7E6 !important;
    opacity: 1 !important;
}
/*        Sidebar Section        */
.sidebar-section {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--primary); margin: 1rem 0 0.5rem;
}

/*        Sidebar Image        */
.sidebar-cloud-img {
    width: 100%;
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 0.8rem;
    border: 1px solid var(--border);
}
.sidebar-cloud-img img {
    width: 100%;
    height: 90px;
    object-fit: cover;
    object-position: center;
    display: block;
    opacity: 0.7;
    border-radius: var(--radius);
}


/*        Form/Input Contrast Fixes        */
[data-testid="stTextInputRootElement"] input,
[data-testid="stTextArea"] textarea,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background: var(--bg-card2) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

[data-testid="stTextInputRootElement"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder,
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {
    color: #A7ACC7 !important;
    opacity: 1 !important;
}

/*        Chat Input Container Contrast        */
[data-testid="stBottomBlockContainer"] {
    background: linear-gradient(180deg, rgba(15,17,23,0.0) 0%, rgba(15,17,23,0.92) 45%, rgba(15,17,23,0.98) 100%) !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    background: transparent !important;
}


/*        Force High Contrast For Streamlit/BaseWeb Inputs        */
[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="base-input"],
[data-testid="stChatInput"] div[data-baseweb="base-input"],
[data-testid="stChatInput"] div[data-baseweb="input"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] > div > div > div,
div[data-baseweb="textarea"] {
    background-color: #1E2235 !important;
    color: #EAEAF4 !important;
    border-color: #3A3F63 !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stChatInput"] input,
[data-testid="stChatInput"] textarea,
[data-testid="stTextArea"] textarea {
    color: #EAEAF4 !important;
    -webkit-text-fill-color: #EAEAF4 !important;
    background-color: transparent !important;
}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: #B8BCD3 !important;
    opacity: 1 !important;
}
/*        Toggle/Slider Labels Readability        */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #DDE2F7 !important;
}

/*        Disabled Buttons Should Stay Readable        */
.stButton > button:disabled {
    color: #EAF4FF !important;
    background: rgba(120, 185, 255, 0.35) !important;
    border: 1px solid rgba(120, 185, 255, 0.75) !important;
    opacity: 1 !important;
}
/*        Expander        */
[data-testid="stExpander"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary { font-size: 0.88rem !important; color: var(--text-muted) !important; }


/*        Global Anti-White Surfaces        */
[data-testid="stApp"] button,
[data-testid="stSidebar"] button,
[data-testid="stApp"] [data-baseweb="input"],
[data-testid="stApp"] [data-baseweb="base-input"],
[data-testid="stApp"] [data-baseweb="textarea"],
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [data-baseweb="textarea"] {
    box-shadow: none !important;
}

[data-testid="stApp"] button,
[data-testid="stSidebar"] button {
    background-color: #23355F !important;
    color: #EAF4FF !important;
    border-color: #5E7EBE !important;
}

[data-testid="stApp"] button:hover,
[data-testid="stSidebar"] button:hover {
    background-color: #2D4477 !important;
    color: #FFFFFF !important;
}

[data-testid="stApp"] button:disabled,
[data-testid="stSidebar"] button:disabled {
    background-color: #325088 !important;
    color: #DCEBFF !important;
    border-color: #7EA7E6 !important;
    opacity: 1 !important;
}

[data-testid="stApp"] input,
[data-testid="stApp"] textarea,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background-color: #1E2235 !important;
    color: #EAEAF4 !important;
    -webkit-text-fill-color: #EAEAF4 !important;
}


/*        Expander Header Button Fix        */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    background: #23355F !important;
    color: #EAF4FF !important;
    border-color: #5E7EBE !important;
}

[data-testid="stExpander"] button,
[data-testid="stExpander"] button:disabled {
    background: #23355F !important;
    color: #EAF4FF !important;
    border: 1px solid #5E7EBE !important;
    opacity: 1 !important;
}

[data-testid="stExpander"] button:hover {
    background: #2D4477 !important;
    color: #FFFFFF !important;
}
/*        Expander Anti-White Override        */
[data-testid="stExpander"] > details {
    background: #1A1D2E !important;
    border: 1px solid #2E3250 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] > details > summary {
    background: #1E2235 !important;
    color: #EAF4FF !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] > details > summary:hover {
    background: #243052 !important;
    color: #FFFFFF !important;
}

[data-testid="stExpander"] > details > div {
    background: #1A1D2E !important;
    color: #EAEAF4 !important;
}

[data-testid="stExpander"] * {
    color: #EAEAF4 !important;
}
/*        Responsive        */
@media (max-width: 768px) {
    .metric-row { grid-template-columns: repeat(2, 1fr); }
    .hero-title  { font-size: 1.5rem; }
    .office-image-banner img { height: 160px; }
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


#        Auto-scroll to latest message                                                                                                                                        
AUTOSCROLL_JS = """
<script>
function scrollToBottom() {
    const chatContainer = window.parent.document.querySelector(
        '[data-testid="stAppScrollToBottomContainer"]'
    );
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    } else {
        window.parent.document.documentElement.scrollTop =
            window.parent.document.documentElement.scrollHeight;
    }
}
scrollToBottom();
setTimeout(scrollToBottom, 300);
setTimeout(scrollToBottom, 800);
</script>
<div id="scroll-anchor"></div>
"""


#                                                                                                                                                                                                                                                 
#                                  DATA CLASSES                                      
#                                                                                                                                                                                                                                                 
def get_api_base_url() -> str:
    return st.session_state.get("api_base_url", API_BASE_URL_DEFAULT).rstrip("/")


def get_api_key() -> str:
    return st.session_state.get("api_key", API_KEY_DEFAULT).strip()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_api_health(api_base_url: str) -> dict:
    try:
        response = requests.get(f"{api_base_url}/health", timeout=6)
        response.raise_for_status()
        return {"ok": True, "data": response.json(), "error": ""}
    except Exception as exc:
        return {"ok": False, "data": {}, "error": str(exc)}


def ask_api(question: str, top_k: int, include_sources: bool) -> dict:
    api_base_url = get_api_base_url()
    api_key = get_api_key()
    payload = {
        "question": question,
        "top_k": top_k,
        "include_sources": include_sources,
    }
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    response = requests.post(f"{api_base_url}/ask", json=payload, headers=headers, timeout=180)
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"API error ({response.status_code}): {detail}")

    data = response.json()
    timings = data.get("timings", {})
    sources = data.get("sources", [])
    return {
        "answer": data.get("answer", ""),
        "route": data.get("route"),
        "sources": sources,
        "scores": [float(item.get("score", 0.0)) for item in sources],
        "t_retrieval": float(timings.get("retrieval", 0.0)),
        "t_generation": float(timings.get("generation", 0.0)),
        "total_time": float(timings.get("total", 0.0)),
    }


def source_view_model(source, score_hint=None) -> dict:
    if isinstance(source, dict):
        return {
            "filename": source.get("filename", "Policy Document"),
            "heading": source.get("heading", "(no section)"),
            "content": source.get("preview", ""),
            "score": float(source.get("score", score_hint if score_hint is not None else 0.0)),
        }

    metadata = getattr(source, "metadata", {}) or {}
    return {
        "filename": metadata.get("filename", "Policy Document"),
        "heading": metadata.get("heading", "(no section)"),
        "content": getattr(source, "page_content", ""),
        "score": float(score_hint if score_hint is not None else 0.0),
    }


#                                                                                                                                                                                                                                                 
#                               SESSION STATE INIT                                   
#                                                                                                                                                                                                                                                 
def init_session():
    for k, v in {
        "messages": [], "total_queries": 0,
        "avg_time": [], "prefill": "",
        "api_base_url": API_BASE_URL_DEFAULT,
        "api_key": API_KEY_DEFAULT,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


#                                                                                                                                                                                                                                                 
#                                 SIDEBAR                                            
#                                                                                                                                                                                                                                                 
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.5rem;">
        <div style="font-size:3rem;">    </div>
        <div style="font-size:1.1rem; font-weight:700; color:#EAEAF4;">SnailCloud Tech</div>
        <div style="font-size:0.75rem; color:#8B8FA8; margin-top:2px;">HR Helpdesk Portal</div>
    </div>
    """, unsafe_allow_html=True)

    #        Cloud Computing Image in Sidebar       
    cloud_img = img_tag("cloud_computing.jpg",
                        "width:100%;height:90px;object-fit:cover;object-position:center 40%;"
                        "border-radius:10px;opacity:0.75;display:block;")
    if cloud_img:
        st.markdown(f"""
        <div class="sidebar-cloud-img">{cloud_img}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<hr style="border-color:#2E3250; margin:0.5rem 0 0.8rem;">', unsafe_allow_html=True)

    #        Model Info Card       
    st.markdown(f"""
    <div style="background:rgba(108,99,255,0.08); border:1px solid rgba(108,99,255,0.25);
         border-radius:10px; padding:0.8rem 1rem; margin-bottom:1rem;">
        <div style="font-size:0.7rem; color:#8B8FA8; text-transform:uppercase;
             letter-spacing:0.08em; margin-bottom:0.4rem;">     FastAPI Backend</div>
        <div style="font-size:0.88rem; font-weight:600; color:#A29BFE;">gemini-2.5-flash</div>
        <div style="font-size:0.72rem; color:#8B8FA8; margin-top:2px;">Fast    Accurate    Low cost</div>
        <div style="font-size:0.7rem; color:#8B8FA8; margin-top:6px;">
                 UI mode: Thin client<br>
                    Calls: /ask and /retrieve-debug<br>
                 Backend: FastAPI + RAG service<br>
                 Documents: 31 HR Policy Files
        </div>
    </div>
    """, unsafe_allow_html=True)

    #        Retrieval Settings       
    st.markdown('<div class="sidebar-section">       Retrieval Settings</div>', unsafe_allow_html=True)
    api_base_url = st.text_input(
        "Backend API URL",
        value=st.session_state.get("api_base_url", API_BASE_URL_DEFAULT),
        help="FastAPI base URL. Example: http://127.0.0.1:8000",
    ).rstrip("/")
    st.session_state["api_base_url"] = api_base_url
    st.caption("API key is loaded from server environment.")
    health = fetch_api_health(api_base_url)
    if health["ok"]:
        st.success("Backend connected")
    else:
        st.error(f"Backend unavailable: {health['error'][:120]}")

    top_k = st.slider(
        "Top-K Policy Chunks",
        min_value=2,
        max_value=12,
        value=6,
        help="Override backend retrieval top_k for this session.",
    )
    show_sources = st.checkbox("Show Source Chunks", value=True)
    show_scores  = st.checkbox("Show Relevance Scores", value=False)

    st.markdown('<hr style="border-color:#2E3250;">', unsafe_allow_html=True)

        #        Sample Questions       
    st.markdown('<div class="sidebar-section">     Sample Questions</div>', unsafe_allow_html=True)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if st.button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state["prefill"] = q
            st.rerun()

    st.markdown('<hr style="border-color:#2E3250;">', unsafe_allow_html=True)

        #        Actions       
    st.markdown('<div class="sidebar-section">Actions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True, key="clear_btn"):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.session_state.avg_time = []
            st.rerun()
    with col2:
        chat_log = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}"
            for m in st.session_state.messages
        )
        st.download_button(
            "Export Chat",
            data=chat_log or "No chat yet.",
            file_name="hr_chat_log.txt",
            mime="text/plain",
            key="export_btn",
            use_container_width=True,
        )

    st.markdown("""
    <div style="margin-top:1.5rem; padding:0.8rem; background:rgba(0,0,0,0.2);
         border-radius:8px; text-align:center;">
        <div style="font-size:0.68rem; color:#8B8FA8; line-height:1.7;">
                 hr@snailcloud.in<br>
                 snailcloud.in/hr-portal<br>
                 Ext: 1800 (HR Helpline)<br><br>
            <span style="color:#A29BFE;">Powered by Gemini 2.5 Flash    pgvector RAG</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


#                                                                                                                                                                                                                                                 
#                               LOAD RESOURCES                                       
#                                                                                                                                                                                                                                                 
# Thin client mode:
# Streamlit does not load vector DB or LLM directly.
# It calls FastAPI backend endpoints.


#                                                                                                                                                                                                                                                 
#             OFFICE IMAGE BANNER  (Image 1     SnailCloud Technologies)               
#                                                                                                                                                                                                                                                 
office_img = img_tag("snailcloud_office.jpg",
                     "width:100%;height:260px;object-fit:cover;object-position:center 30%;"
                     "display:block;border-radius:20px;")
if office_img:
    st.markdown(f"""
    <div class="office-image-banner">
        {office_img}
        <div class="office-image-overlay">
            <div style="font-size:1.4rem;font-weight:700;color:#EAEAF4;line-height:1.2;">
                     SnailCloud Technologies
            </div>
            <div style="font-size:0.82rem;color:#8B8FA8;margin-top:3px;">
                Cloud Computing    Data Solutions    Innovation
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


#                                                                                                                                                                                                                                                 
#                                HERO BANNER                                         
#                                                                                                                                                                                                                                                 
st.markdown("""
<div class="hero-banner">
    <div style="display:flex; align-items:center; gap:1.2rem;">
        <div style="font-size:3.5rem; line-height:1;">    </div>
        <div>
            <div class="hero-title">HR Helpdesk Assistant</div>
            <div class="hero-subtitle">
                Ask any HR policy question     answers grounded in SnailCloud Technologies' official documents
            </div>
            <div style="margin-top:0.8rem;">
                <span class="hero-badge cloud">     Gemini 2.5 Flash</span>
                <span class="hero-badge cloud">     gemini-embedding-001</span>
                <span class="hero-badge">        pgvector RAG</span>
                <span class="hero-badge">     31 Policy Docs</span>
                <span class="hero-badge">     MMR Retrieval</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


#                                                                                                                                                                                                                                                 
#             CLOUD IMAGE STRIP  (Image 2     Cloud Computing Visual)                  
#                                                                                                                                                                                                                                                 
cloud_strip = img_tag("cloud_computing.jpg",
                      "width:100%;height:120px;object-fit:cover;object-position:center 40%;"
                      "display:block;opacity:0.75;border-radius:12px;")
if cloud_strip:
    st.markdown(f"""
    <div class="cloud-image-strip">
        {cloud_strip}
        <div class="cloud-image-label">
                   Powered by pgvector    Gemini Embeddings    RAG Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)


#                                                                                                                                                                                                                                                 
#                             METRICS ROW                                            
#                                                                                                                                                                                                                                                 
avg_t = (
    f"{sum(st.session_state.avg_time)/len(st.session_state.avg_time):.1f}s"
    if st.session_state.avg_time else "   "
)
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card purple">
        <div class="icon">    </div>
        <div class="value">{st.session_state.total_queries}</div>
        <div class="label">Queries Asked</div>
    </div>
    <div class="metric-card teal">
        <div class="icon">    </div>
        <div class="value">510</div>
        <div class="label">Policy Chunks</div>
    </div>
    <div class="metric-card red">
        <div class="icon">    </div>
        <div class="value">31</div>
        <div class="label">HR Documents</div>
    </div>
    <div class="metric-card blue">
        <div class="icon">   </div>
        <div class="value">{avg_t}</div>
        <div class="label">Avg Response</div>
    </div>
</div>
""", unsafe_allow_html=True)


#                                                                                                                                                                                                                                                 
#                              CHAT HISTORY                                          
#                                                                                                                                                                                                                                                 
for msg in st.session_state.messages:
    icon = "🙂" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=icon):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if "meta" in msg:
                m = msg["meta"]
                route_txt = f' |     route: {m["route"]}' if m.get("route") else ""
                st.markdown(
                    f'<span class="info-pill">    {m["total_time"]}s'
                    f' |      {m["t_retrieval"]}s retrieval'
                    f' |      {m["t_generation"]}s generation'
                    f' |      {len(m["sources"])} chunks{route_txt}</span>'
                    f'<span class="cloud-pill">     Gemini 2.5 Flash</span>',
                    unsafe_allow_html=True
                )
            if show_sources and msg.get("sources"):
                with st.expander(f"     View {len(msg['sources'])} Source Chunks", expanded=False):
                    for i, (src, score) in enumerate(
                        zip(msg["sources"], msg.get("scores", [None]*len(msg["sources"]))), 1
                    ):
                        src_vm = source_view_model(src, score_hint=score)
                        filename = src_vm["filename"]
                        heading = src_vm["heading"]
                        src_score = src_vm["score"]
                        src_body = src_vm["content"]
                        score_html = ""
                        if show_scores and src_score is not None:
                            color = "#00D4AA" if src_score < 0.8 else "#FFB347" if src_score < 1.2 else "#FF6B6B"
                            score_html = f'<span style="float:right;color:{color};font-size:0.72rem;">dist:{src_score:.3f}</span>'
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">     [{i}] {filename} {score_html}</div>
                            <div class="source-section">     {heading}</div>
                            <div class="source-body">{src_body[:500]}{"   " if len(src_body)>500 else ""}</div>
                        </div>""", unsafe_allow_html=True)


#                                                                                                                                                                                                                                                 

# Quick clickable questions in main panel -> prefill chat input
st.markdown("### Quick Questions")
quick_cols = st.columns(2)
for i, q in enumerate(SAMPLE_QUESTIONS[:8]):
    with quick_cols[i % 2]:
        if st.button(q, key=f"main_sq_{i}", use_container_width=True):
            st.session_state["prefill"] = q
            st.rerun()
#                              CHAT INPUT                                            
#                                                                                                                                                                                                                                                 
prefill  = st.session_state.pop("prefill", "")
question = st.chat_input("      Ask an HR policy question    ") or prefill

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        status = st.status("     Calling backend API    ", expanded=False)
        try:
            with status:
                st.write(f"    POST {get_api_base_url()}/ask")
                result = ask_api(question, top_k=top_k, include_sources=show_sources)
                st.write(f"    Retrieved {len(result['sources'])} chunks in {result['t_retrieval']}s")
                st.write(f"    Answer generated in {result['t_generation']}s")
                if result.get("route"):
                    st.write(f"    Route selected: {result['route']}")
            status.update(label="    Answer ready!", state="complete", expanded=False)

            st.markdown(result["answer"])
            route_txt = f' |     route: {result["route"]}' if result.get("route") else ""
            st.markdown(
                f'<span class="info-pill">    {result["total_time"]}s'
                f' |      {result["t_retrieval"]}s |      {result["t_generation"]}s'
                f' |      {len(result["sources"])} chunks{route_txt}</span>'
                f'<span class="cloud-pill">     Gemini 2.5 Flash</span>',
                unsafe_allow_html=True
            )

            if show_sources and result["sources"]:
                with st.expander(f"     View {len(result['sources'])} Source Chunks", expanded=False):
                    for i, (src, score) in enumerate(zip(result["sources"], result["scores"]), 1):
                        src_vm = source_view_model(src, score_hint=score)
                        filename = src_vm["filename"]
                        heading = src_vm["heading"]
                        src_score = src_vm["score"]
                        src_body = src_vm["content"]
                        color = "#00D4AA" if src_score < 0.8 else "#FFB347" if src_score < 1.2 else "#FF6B6B"
                        score_html = (
                            f'<span style="float:right;color:{color};font-size:0.72rem;">dist:{src_score:.3f}</span>'
                        ) if show_scores else ""
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">     [{i}] {filename} {score_html}</div>
                            <div class="source-section">     {heading}</div>
                            <div class="source-body">{src_body[:500]}{"   " if len(src_body)>500 else ""}</div>
                        </div>""", unsafe_allow_html=True)

            #        Auto-scroll to latest answer       
            st.markdown(AUTOSCROLL_JS, unsafe_allow_html=True)

            st.session_state.total_queries += 1
            st.session_state.avg_time.append(result["total_time"])
            st.session_state.messages.append({
                "role": "assistant", "content": result["answer"],
                "sources": result["sources"], "scores": result["scores"], "meta": result,
            })
            st.rerun()

        except Exception as e:
            status.update(label="    Error", state="error")
            err = f"    **Error:** `{e}`\n\nPlease try again or contact hr@snailcloud.in"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})


#                                                                                                                                                                                                                                                 
#                              EMPTY STATE                                           
#                                                                                                                                                                                                                                                 
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:2.5rem 2rem; opacity:0.8;">
        <div style="font-size:4rem; margin-bottom:1rem;">    </div>
        <div style="font-size:1.2rem; font-weight:600; color:#EAEAF4; margin-bottom:0.5rem;">
            Ask your first HR question
        </div>
        <div style="font-size:0.9rem; color:#8B8FA8; max-width:500px; margin:0 auto;">
            Type a question below or click a sample question from the sidebar.
            Powered by Gemini 2.5 Flash + pgvector RAG.
        </div>
        <div style="display:flex; justify-content:center; gap:1rem; margin-top:1.5rem; flex-wrap:wrap;">
            <div style="background:rgba(108,99,255,0.1);border:1px solid rgba(108,99,255,0.2);
                 padding:8px 16px;border-radius:20px;font-size:0.82rem;color:#A29BFE;">
                     Salary & Payroll
            </div>
            <div style="background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.2);
                 padding:8px 16px;border-radius:20px;font-size:0.82rem;color:#00D4AA;">
                     Health & Benefits
            </div>
            <div style="background:rgba(255,107,107,0.1);border:1px solid rgba(255,107,107,0.2);
                 padding:8px 16px;border-radius:20px;font-size:0.82rem;color:#FF6B6B;">
                     Career & Growth
            </div>
            <div style="background:rgba(108,99,255,0.1);border:1px solid rgba(108,99,255,0.2);
                 padding:8px 16px;border-radius:20px;font-size:0.82rem;color:#A29BFE;">
                     Powered by Gemini 2.5 Flash
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)























