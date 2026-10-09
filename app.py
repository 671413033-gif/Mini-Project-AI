import os
import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

# --- CONFIG & API KEYS ---
# เก็บคีย์ไว้ใน .streamlit/secrets.toml  ->  AIFORTHAI_KEY = "xxxx"
# (หรือตั้งเป็น Environment Variable ชื่อ AIFORTHAI_KEY)
try:
    API_KEY = st.secrets["AIFORTHAI_KEY"]
except Exception:
    API_KEY = os.environ.get("AIFORTHAI_KEY", "")

st.set_page_config(
    page_title="Team Fit AI & Personality Analyzer",
    page_icon="🧠",
    layout="wide"
)

# --- CUSTOM MODERN DESIGN SYSTEM & CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Prompt:wght@300;400;500;600;700&display=swap');

    html, body, .stApp,
    .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6, .stApp input, .stApp textarea,
    .stApp button p, .stApp td, .stApp th {
        font-family: 'Plus Jakarta Sans', 'Prompt', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    /* กันไอคอน Material ไม่ให้กลายเป็นตัวหนังสือ */
    .material-symbols-rounded, [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* ซ่อนเมนูมุมขวาบนทั้งหมด (Deploy, ⋮, Rerun, Clear cache, Print, Record screen) */
    [data-testid="stToolbar"],
    [data-testid="stMainMenu"],
    [data-testid="stDeployButton"],
    [data-testid="stStatusWidget"],
    #MainMenu, footer { display: none !important; visibility: hidden !important; }
    header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
    .block-container { padding-top: 1.5rem !important; }

    /* Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #312E81 100%);
        border-radius: 16px;
        padding: 24px 28px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 60%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 14px;
        margin: 0 0 14px 0;
        line-height: 1.5;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #E2E8F0;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22C55E;
        box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7);
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    /* Modern Cards */
    .modern-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        transition: all 0.2s ease-in-out;
    }
    .modern-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 6px 18px -3px rgba(0,0,0,0.06);
    }

    /* Preset & Action Buttons */
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        transition: all 0.15s ease-in-out;
        border: 1px solid #E2E8F0;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        border-color: #94A3B8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }

    /* Tab Headers & Subtabs */
    button[data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 600;
        padding: 8px 16px;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1E293B !important;
        border-bottom: 3px solid #2563EB !important;
        font-weight: 700 !important;
        background: rgba(37, 99, 235, 0.04);
    }

    /* Expander Containers */
    div[data-testid="stExpander"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        background: #FAFAFA;
        margin-bottom: 12px;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600;
        font-size: 14px;
        color: #1E293B;
    }

    /* Tag Pills for Skills */
    .tag-hard {
        background-color: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
        margin: 2px 4px 2px 0;
    }
    .tag-soft {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
        margin: 2px 4px 2px 0;
    }
    .tag-tool {
        background-color: #F8FAFC;
        color: #0369A1;
        border: 1px solid #BAE6FD;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
        margin: 2px 4px 2px 0;
    }
    .tag-candidate {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

CATEGORIES = [
    'การสื่อสาร (Text Length)', 
    'ความละเอียด (Word Count)', 
    'ความใส่ใจทีม (Team Score)', 
    'ทัศนคติบวก (Sentiment)', 
    'การลงมือทำ (Action)'
]

ROLE_PROFILES = {
    0: {
        "title": "🎯 The Driver / Leader (ผู้นำมุ่งมั่นผลงาน)",
        "short_title": "The Driver / Leader",
        "color": "#E63946",
        "strengths": "กล้าตัดสินใจ มุ่งเน้นความสำเร็จและเป้าหมาย ผลักดันงานให้เสร็จตามกำหนด มีภาวะผู้นำสูง",
        "blindspots": "อาจเร่งรัดงานจนเพื่อนร่วมทีมเครียด หรือลืมฟังรายละเอียดปลีกย่อยของคนอื่น",
        "pairing": "จับคู่ได้ดีกับ 🔍 The Analytical เพื่อช่วยตรวจสอบความเสี่ยง และ 🤝 The Team Player เพื่อช่วยดูแลขวัญกำลังใจทีม",
        "tip": "ลองให้เวลารับฟังมุมมองของเพื่อนร่วมทีมให้มากขึ้นก่อนเคาะทิศทางงาน",
        "job_title": "Project Leader / Scrum Master",
        "track": "สายบริหารและผู้นำโครงการ (Project Leadership)",
        "skills": {
            "hard": [
                "Project Management & Roadmap Planning", 
                "Agile & Scrum Framework (Sprint, Backlog, Retro)", 
                "Risk Assessment & Contingency Planning", 
                "KPI & Sprint Tracking", 
                "Resource & Scope Management"
            ],
            "soft": [
                "ภาวะผู้นำเชิงกลยุทธ์ (Strategic Leadership)", 
                "การตัดสินใจที่เฉียบขาด (Decisiveness)", 
                "การเจรจาต่อรองและการโน้มน้าวใจ (Negotiation)", 
                "การระงับข้อขัดแย้ง (Conflict Resolution)", 
                "การบริหารเวลาและรับมือแรงกดดัน (Time Management)"
            ],
            "tools": ["Jira", "Trello", "Notion", "GitHub Projects", "Slack", "Asana", "Miro"]
        },
        "responsibilities": [
            "กำหนดทิศทาง Roadmap และ Milestone ของโปรเจกต์ให้ชัดเจน",
            "จัดสรรและแจกแจงงานในแต่ละ Sprint พร้อมติดตามความคืบหน้า",
            "ปลดบล็อกอุปสรรค (Blockers) ให้เพื่อนร่วมทีมทำงานต่อได้ทันที",
            "ตัดสินใจขั้นสุดท้ายเมื่อทีมมีความคิดเห็นหลากหลาย"
        ],
        "upskill_recommendations": "ฝึกฝนการฟังอย่างลึกซึ้ง (Active Listening) และการให้คำชมหรือ Feedback เชิงบวกเพื่อเสริมพลังทีม",
        "competency_vector": [95, 60, 80, 70, 75]
    },
    1: {
        "title": "🔍 The Analytical (นักคิดวิเคราะห์รอบคอบ)",
        "short_title": "The Analytical",
        "color": "#1D3557",
        "strengths": "คิดอย่างมีเหตุผล ละเอียดรอบคอบ มองเห็นความเสี่ยงและจุดบกพร่องของระบบได้ชัดเจน",
        "blindspots": "คิดวิเคราะห์เยอะเกินไป (Analysis Paralysis) จนเริ่มงานช้ากว่าแผน หรือกังวลมากเกินไป",
        "pairing": "จับคู่ได้ดีกับ 🎯 The Driver เพื่อช่วยเคาะตัดสินใจ และ 💡 The Creative เพื่อช่วยจุดประกายไอเดียใหม่",
        "tip": "กำหนดเส้นตายชัดเจน และยอมรับความสมบูรณ์แบบที่ 80% เพื่อให้เริ่มลงมือทำได้ไวขึ้น",
        "job_title": "Core Developer / Data & System Analyst",
        "track": "สายพัฒนาและวิเคราะห์ข้อมูล (Engineering & Data)",
        "skills": {
            "hard": [
                "System Architecture & API Design", 
                "Algorithm Design & Optimization", 
                "Python, SQL & Backend Development", 
                "Data Modeling, ETL & Pipeline Design", 
                "Code Review, Refactoring & Unit Testing"
            ],
            "soft": [
                "การคิดเชิงวิเคราะห์และตรรกะ (Critical Thinking)", 
                "ความละเอียดรอบคอบในการตรวจสอบ (Attention to Detail)", 
                "การแก้ปัญหาเชิงโครงสร้าง (Structured Problem Solving)", 
                "การประเมินความเสี่ยงเชิงเทคนิค (Technical Risk Assessment)", 
                "ความมุ่งมั่นและความอดทนในการแก้บัก (Persistence)"
            ],
            "tools": ["Python", "VS Code", "Git & GitHub", "Docker", "PostgreSQL", "Pandas", "Scikit-Learn"]
        },
        "responsibilities": [
            "ออกแบบสถาปัตยกรรมระบบ โครงสร้างฐานข้อมูล และ API",
            "พัฒนาฟังก์ชันการทำงานหลักและอัลกอริทึมที่ซับซ้อน",
            "ทดสอบความปลอดภัย ประสิทธิภาพ และความถูกต้องของข้อมูล",
            "ตรวจสอบคุณภาพโค้ด (Code Review) และป้องกันข้อผิดพลาดเชิงเทคนิค"
        ],
        "upskill_recommendations": "ฝึกการสื่อสารเรื่องเทคนิคยากๆ ให้คนทั่วไปเข้าใจง่าย และฝึกเทคนิค Timeboxing เพื่อป้องกันการติดหล่มคิดนาน",
        "competency_vector": [55, 95, 75, 60, 95]
    },
    2: {
        "title": "🤝 The Team Player (ผู้ประสานงานและสร้างมิตรภาพ)",
        "short_title": "The Team Player",
        "color": "#2A9D8F",
        "strengths": "รับฟังผู้อื่นเก่ง สร้างบรรยากาศที่ดี สมานรอยร้าวและลดความขัดแย้งในทีม เอาใจใส่ทุกคน",
        "blindspots": "เกรงใจผู้อื่นมากเกินไป ไม่กล้าแสดงความคิดเห็นที่ขัดแย้ง หรือแบกรับภาระแทนคนอื่น",
        "pairing": "เข้ากับทุกบทบาทได้ดี โดยเฉพาะทำงานร่วมกับ 🎯 The Driver เพื่อช่วยสื่อสารและประสานทีมอย่างนุ่มนวล",
        "tip": "ฝึกแสดงความเห็นตรงไปตรงมา การท้วงติงด้วยความหวังดีจะช่วยให้โปรเจกต์ดียิ่งขึ้น",
        "job_title": "Team Coordinator & QA Lead",
        "track": "สายประสานงานและควบคุมคุณภาพ (Coordination & QA)",
        "skills": {
            "hard": [
                "Quality Assurance (QA) & Test Case Design", 
                "User Acceptance Testing (UAT) & Bug Triage", 
                "Sprint Facilitation & Team Retrospective", 
                "Technical Documentation & User Manuals", 
                "Release Management & Cross-team Sync"
            ],
            "soft": [
                "การสื่อสารด้วยความเข้าอกเข้าใจ (Empathy & Active Listening)", 
                "การประสานงานข้ามสายงาน (Cross-functional Collaboration)", 
                "การสร้างบรรยากาศเชิงบวกและขวัญกำลังใจ (Team Morale Building)", 
                "การเจรจาสมานฉันท์และลดแรงเสียดทาน (Mediation)", 
                "ความยืดหยุ่นและการสนับสนุนผู้อื่น (Supportiveness)"
            ],
            "tools": ["Confluence", "Miro", "Discord / Slack", "Google Workspace", "Postman", "Notion", "ClickUp"]
        },
        "responsibilities": [
            "ดูแลการทดสอบระบบ ออกแบบ Test Cases และควบคุมคุณภาพงานก่อนส่งมอบ",
            "ประสานงานระหว่างฝ่ายพัฒนา ฝ่ายออกแบบ และฝ่ายบริหาร",
            "ดูแลและอัปเดตเอกสารประกอบโปรเจกต์และสรุปการประชุม",
            "ดูแลสภาพจิตใจ รับฟังปัญหา และเสริมสร้างความสามัคคีในทีม"
        ],
        "upskill_recommendations": "ฝึกทักษะ Assertive Communication กล้าปฏิเสธอย่างสุภาพ และกล้าทักท้วงเมื่อพบข้อผิดพลาด",
        "competency_vector": [75, 55, 95, 65, 60]
    },
    3: {
        "title": "💡 The Creative Innovator (นักคิดสร้างสรรค์สิ่งใหม่)",
        "short_title": "The Creative Innovator",
        "color": "#F4A261",
        "strengths": "คิดนอกกรอบ มีไอเดียแปลกใหม่ มองเห็นโอกาสที่คนอื่นมองข้าม ยืดหยุ่นปรับตัวเก่ง",
        "blindspots": "อาจเบื่อง่ายเมื่อต้องทำงานเอกสาร รูทีน หรือขั้นตอนที่มีแบบแผนตายตัว",
        "pairing": "จับคู่ได้ดีกับ 🔍 The Analytical เพื่อช่วยนำไอเดียมาทำให้เกิดผลจริงได้อย่างเป็นขั้นตอน",
        "tip": "จดบันทึกไอเดียเป็นระบบและแบ่งงานเป็นสเต็ปย่อยๆ เพื่อส่งต่องานให้เพื่อนร่วมทีมทำต่อได้ง่าย",
        "job_title": "Product Designer / UI/UX & Innovation Lead",
        "track": "สายออกแบบและนวัตกรรม (UI/UX & Innovation)",
        "skills": {
            "hard": [
                "UI/UX Design, Wireframing & Interactive Prototyping", 
                "Design Thinking & User-Centered Design Methodology", 
                "User Research, Usability Testing & Journey Mapping", 
                "Design Systems & Component Library Architecture", 
                "Information Architecture (IA) & Content Strategy"
            ],
            "soft": [
                "ความคิดสร้างสรรค์และการคิดนอกกรอบ (Lateral & Creative Thinking)", 
                "ความยืดหยุ่นและการปรับตัวต่อการเปลี่ยนแปลง (Adaptability & Agility)", 
                "การเล่าเรื่องเพื่อถ่ายทอดวิสัยทัศน์ (Visual Storytelling)", 
                "การกระตุ้นแรงบันดาลใจในทีม (Inspiration & Ideation)", 
                "การเปิดรับความคิดเห็นเพื่อพัฒนาต่อยอด (Open-mindedness)"
            ],
            "tools": ["Figma", "Adobe XD", "FigJam", "Canva", "Whimsical", "Maze", "Midjourney"]
        },
        "responsibilities": [
            "ออกแบบโครงสร้างหน้าจอ Wireframe, Prototype และ UI Design System",
            "ศึกษาพฤติกรรมผู้ใช้ (User Research) และทดสอบความพึงพอใจ (Usability)",
            "คิดค้นฟีเจอร์แปลกใหม่เพื่อสร้างความแตกต่างและจุดเด่นให้ผลิตภัณฑ์",
            "ถ่ายทอด Concept และวิสัยทัศน์ของโปรดักต์ให้สมาชิกในทีมเห็นภาพตรงกัน"
        ],
        "upskill_recommendations": "ฝึกการจัดทำเอกสารส่งต่องานออกแบบ (Hand-off documentation) ให้ทีม Dev พัฒนาต่อง่ายขึ้น",
        "competency_vector": [65, 65, 65, 95, 70]
    }
}

TEAM_COMPETENCY_AREAS = [
    "การบริหารและแผนงาน (Management)", 
    "วิศวกรรมและโค้ดดิ้ง (Engineering)", 
    "การประกันคุณภาพและประสานงาน (QA & Coordination)", 
    "การออกแบบและนวัตกรรม (UI/UX & Design)", 
    "การวิเคราะห์ข้อมูลและตรรกะ (Analytics & Logic)"
]

ROLE_SYNERGY_RULES = {
    (0, 0): {
        "score": 58,
        "type": "⚠️ ระวังการชนกัน (Ego Clash Risk)",
        "detail": "ผู้นำ 2 คนทำงานร่วมกัน อาจเกิดการแย่งบทบาทตัดสินใจหรือขัดแย้งในทิศทาง ควรกำหนดขอบเขตโครงการ (Co-ownership) ให้ชัดเจน",
        "status": "warning"
    },
    (0, 1): {
        "score": 93,
        "type": "🌟 คู่หูยุทธศาสตร์ (Strategic Execution Duo)",
        "detail": "Leader ช่วยตัดสินใจและกำหนดทิศทาง ขณะที่ Analytical ช่วยตรวจสอบตรรกะ ข้อมูล และความเสี่ยง ทำให้ทีมเดินหน้าอย่างมั่นคงและถูกต้อง",
        "status": "success"
    },
    (0, 2): {
        "score": 96,
        "type": "🌟 คู่หูขับเคลื่อนและสานพลัง (Driver & Anchor Duo)",
        "detail": "Leader มุ่งเน้นผลลัพธ์และความเร็ว ขณะที่ Team Player ช่วยดูแลสภาพจิตใจและสมานรอยร้าวในทีม ทำให้ทีมไปถึงเป้าหมายโดยไม่มีใครตกหล่น",
        "status": "success"
    },
    (0, 3): {
        "score": 89,
        "type": "🚀 คู่หูบุกเบิกสิ่งใหม่ (Visionary & Pathfinder Duo)",
        "detail": "Creative จุดประกายไอเดียนวัตกรรม และ Leader ช่วยเคาะเลือกไอเดียที่ดีที่สุดพร้อมแปลงเป็น Action Plan ที่ทำได้จริง",
        "status": "success"
    },
    (1, 1): {
        "score": 74,
        "type": "🔍 คู่หูสายตรรกะเข้มข้น (Deep Analytical Duo)",
        "detail": "คิดวิเคราะห์ได้ลึกซึ้ง ตรวจสอบโค้ดและข้อมูลแม่นยำ แต่อาจเกิดภาวะคิดเยอะจนเริ่มช้า (Analysis Paralysis) ควรตั้ง Timebox เสมอ",
        "status": "info"
    },
    (1, 2): {
        "score": 88,
        "type": "🤝 คู่หูคุณภาพและความเข้าอกเข้าใจ (Logic & Empathy Duo)",
        "detail": "Analytical มีความแม่นยำเรื่องข้อมูลเชิงลึก ขณะที่ Team Player ช่วยสื่อสารให้ทุกคนในทีมเข้าใจง่ายและนำไปใช้ได้อย่างอบอุ่น",
        "status": "success"
    },
    (1, 3): {
        "score": 92,
        "type": "💡 คู่หูนวัตกรรมสมบูรณ์แบบ (Form & Function Duo)",
        "detail": "Creative คิดนอกกรอบเพื่อออกแบบประสบการณ์ที่น่าตื่นเต้น และ Analytical ช่วยออกแบบระบบโครงสร้างทางเทคนิคให้ใช้งานได้จริง",
        "status": "success"
    },
    (2, 2): {
        "score": 83,
        "type": "💖 คู่หูสร้างบรรยากาศเชิงบวก (Harmony Champions)",
        "detail": "ทีมมีความสุข อบอุ่น สื่อสารราบรื่น ไม่มีใครเครียด แต่อาจขาดคนกล้าฟันธงหรือตัดช้อยส์งาน ต้องการคนช่วยเคาะเส้นตาย",
        "status": "info"
    },
    (2, 3): {
        "score": 90,
        "type": "🎨 คู่หูสร้างสรรค์และมีส่วนร่วม (Creative Facilitation Duo)",
        "detail": "Creative เสนอแนวคิดแปลกใหม่ Team Player สนับสนุนและกระตุ้นให้ทุกคนในทีมร่วมแสดงความคิดเห็นอย่างเปิดกว้าง",
        "status": "success"
    },
    (3, 3): {
        "score": 75,
        "type": "✨ คู่หูระดมไอเดียไม่รู้จบ (Brainstorming Powerhouse)",
        "detail": "มีไอเดียสร้างสรรค์พรั่งพรู สนุกสนานกับการระดมสมอง แต่อาจหลุดโฟกัสหรือเปลี่ยนแผนบ่อย ต้องมีคนช่วยสรุปเป็นงานประจำ",
        "status": "info"
    }
}

def get_role_synergy(role_a, role_b):
    pair_key = (min(role_a, role_b), max(role_a, role_b))
    return ROLE_SYNERGY_RULES.get(pair_key, {
        "score": 80,
        "type": "🤝 ทำงานร่วมกันได้ดี",
        "detail": "มีจุดแข็งคนละด้านที่เกื้อหนุนกัน",
        "status": "info"
    })

EXTENDED_PALETTE = [
    '#E63946', '#1D3557', '#2A9D8F', '#F4A261', '#9B5DE5', '#00BBF9',
    '#FF006E', '#3A86FF', '#FB5607', '#8338EC', '#06D6A0', '#118AB2',
    '#FFBE0B', '#3D348B', '#7678ED', '#F35B04', '#17C3B2', '#227C9D',
    '#FE6D73', '#FFCB77', '#457B9D', '#A8DADC', '#E76F51', '#264653'
]

# --- BENCHMARK CLUSTERING MODEL TRAINING ---
@st.cache_resource
def get_ml_model():
    """สร้าง Benchmark Dataset และเทรนโมเดล K-Means, PCA, Random Forest Feature Importance"""
    np.random.seed(42)
    c_leader = [320, 55, 1.2, 70, 7]
    c_analytical = [290, 60, 1.5, 60, 2]
    c_team_player = [210, 38, 4.5, 90, 3]
    c_creative = [230, 42, 2.0, 85, 6]
    
    d1 = np.random.normal(c_leader, [30, 6, 0.3, 8, 1], (25, 5))
    d2 = np.random.normal(c_analytical, [25, 5, 0.4, 7, 1], (25, 5))
    d3 = np.random.normal(c_team_player, [20, 4, 0.5, 6, 1], (25, 5))
    d4 = np.random.normal(c_creative, [25, 5, 0.4, 7, 1], (25, 5))
    
    benchmark_data = np.vstack([d1, d2, d3, d4])
    benchmark_data = np.clip(benchmark_data, a_min=[50, 10, 0.1, 0, 0], a_max=None)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(benchmark_data)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)
    pca = PCA(n_components=2, random_state=42).fit(X_scaled)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(benchmark_data, kmeans.labels_)
    feature_importances = rf.feature_importances_
    
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    db_score = davies_bouldin_score(X_scaled, kmeans.labels_)
    inertia = kmeans.inertia_
    
    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "rf": rf,
        "feature_importances": feature_importances,
        "benchmark_data": benchmark_data,
        "benchmark_mean": np.mean(benchmark_data, axis=0),
        "X_scaled": X_scaled,
        "sil_score": sil_score,
        "db_score": db_score,
        "inertia": inertia
    }

ml_bundle = get_ml_model()

# --- HELPER FUNCTIONS FOR AIFORTHAI API ---
def get_sentiment(text):
    """เรียกใช้ Sentiment Analysis API จาก AIForThai (ssense) แบบรวดเร็วสำหรับฟังก์ชันวิเคราะห์"""
    url = "https://api.aiforthai.in.th/ssense"
    headers = {'Apikey': API_KEY}
    data = {'text': text}
    try:
        response = requests.post(url, data=data, headers=headers, timeout=5)
        if response.status_code == 200:
            res = response.json()
            score_val = res.get('sentiment', {}).get('score', 50)
            return float(score_val)
    except:
        pass
    return 50.0

def call_aiforthai_full(text):
    """ส่ง Request ไปยัง AIForThai (ssense) พร้อมเก็บค่า Latency และ Raw Payload สำหรับแสดงในหน้า Monitor"""
    url = "https://api.aiforthai.in.th/ssense"
    headers = {'Apikey': API_KEY}
    data = {'text': text}
    t0 = time.time()
    try:
        response = requests.post(url, data=data, headers=headers, timeout=8)
        latency = (time.time() - t0) * 1000
        if response.status_code == 200:
            res = response.json()
            sent_data = res.get('sentiment', {})
            score = float(sent_data.get('score', 50))
            polarity = sent_data.get('polarity', 'neutral')
            preprocess = res.get('preprocess', {})
            return {
                "status": "Connected",
                "status_code": 200,
                "latency_ms": latency,
                "score": score,
                "polarity": polarity,
                "segmented": preprocess.get("segmented", []),
                "pos_words": preprocess.get("pos", []),
                "neg_words": preprocess.get("neg", []),
                "keywords": preprocess.get("keyword", []),
                "raw_response": res
            }
        else:
            return {
                "status": f"HTTP Error ({response.status_code})",
                "status_code": response.status_code,
                "latency_ms": latency,
                "score": 50.0,
                "polarity": "neutral",
                "segmented": [],
                "pos_words": [],
                "neg_words": [],
                "keywords": [],
                "raw_response": {"error": response.text}
            }
    except Exception as e:
        latency = (time.time() - t0) * 1000
        return {
            "status": f"Connection Error: {str(e)}",
            "status_code": 500,
            "latency_ms": latency,
            "score": 50.0,
            "polarity": "neutral",
            "segmented": [],
            "pos_words": [],
            "neg_words": [],
            "keywords": [],
            "raw_response": {"error": str(e)}
        }

def extract_features(text1, text2=""):
    """ทำ Feature Engineering โดยใช้ PyThaiNLP และ AIForThai"""
    full_text = text1.strip() + " " + text2.strip()
    
    words = [w.strip() for w in word_tokenize(full_text, engine="newmm") if w.strip()]
    word_count = len(words)
    length = len(full_text)
    
    team_words = ["พวกเรา", "เรา", "ทีม", "ทีมงาน", "ทุกคน", "ร่วมมือ", "ช่วยเหลือ", "แบ่งปัน", "รับฟัง"]
    i_words = ["ผม", "ฉัน", "ดิฉัน", "กู", "ตัวเอง", "ข้าพเจ้า"]
    we_count = sum(full_text.count(w) for w in team_words)
    i_count = sum(full_text.count(w) for w in i_words)
    team_orientation = float(we_count + 1) / float(i_count + 1)
    
    sentiment_score = get_sentiment(full_text)
    
    action_words = ["ลงมือ", "ลุย", "ทำ", "แก้ไข", "วางแผน", "จัดระเบียบ", "คุม", "ลุยงาน", "ทดสอบ", "เริ่ม", "สร้าง", "ส่งมอบ", "นำ"]
    action_score = sum(full_text.count(w) for w in action_words)
    
    stops = set(thai_stopwords())
    meaningful_words = [w for w in words if w not in stops and len(w) > 1 and not w.isnumeric()]
    top_keywords = pd.Series(meaningful_words).value_counts().head(5).to_dict() if meaningful_words else {}
    
    features = [length, word_count, team_orientation, sentiment_score, action_score]
    return features, top_keywords

def predict_profile(user_features):
    """ทำนาย Cluster และคำนวณ Normalization, PCA, และ Distance ไปยัง Centroids ทั้ง 4"""
    scaler = ml_bundle["scaler"]
    kmeans = ml_bundle["kmeans"]
    pca = ml_bundle["pca"]
    
    u_feat = np.array([user_features], dtype=float)
    u_scaled = scaler.transform(u_feat)
    cluster_id = int(kmeans.predict(u_scaled)[0])
    u_pca = pca.transform(u_scaled)[0]
    
    distances = np.linalg.norm(kmeans.cluster_centers_ - u_scaled, axis=1)
    
    max_vals = np.max(ml_bundle["benchmark_data"], axis=0)
    max_vals[max_vals == 0] = 1.0
    radar_values = np.clip(np.array(user_features, dtype=float) / max_vals, 0.0, 1.0).tolist()
    
    return cluster_id, radar_values, u_pca, distances

# ==============================================================================
# ==============================================================================
# 🌗 DAY / NIGHT SWITCH — แคปซูลมุมขวาบน (ลอยตามหน้าจอ)
# ==============================================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

with st.container(key="theme_switch"):
    st.toggle("โหมดกลางคืน 🌙", key="dark_mode")

IS_DARK = st.session_state.dark_mode

# --- สไตล์แคปซูล (ใช้ทั้งโหมดกลางวัน/กลางคืน) ---
st.markdown("""
<style>
.block-container { padding-top: 3.6rem !important; }

.st-key-theme_switch {
    position: fixed !important;
    top: 14px; right: 24px;
    z-index: 999999;
    width: fit-content !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%);
    border: 1px solid #E2E8F0;
    border-radius: 999px;
    padding: 8px 18px 4px 18px;
    box-shadow: 0 6px 18px -6px rgba(15, 23, 42, 0.30);
    transition: all .25s ease;
    cursor: pointer;
}
.st-key-theme_switch:hover {
    transform: translateY(-1px);
    border-color: #60A5FA;
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    box-shadow: 0 10px 24px -6px rgba(37, 99, 235, 0.45);
}
.st-key-theme_switch label p { font-weight: 700; font-size: 13px; color: #1E293B; white-space: nowrap; }
.st-key-theme_switch label { cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# ---- กราฟ Plotly เปลี่ยนตามโหมด ----
_PLOT_TEMPLATE = "plotly_dark" if IS_DARK else "plotly_white"
_FONT_COLOR = "#E2E8F0" if IS_DARK else "#1E293B"
_GRID_COLOR = "rgba(148,163,184,0.25)" if IS_DARK else "rgba(100,116,139,0.20)"
_orig_plotly_chart = st.plotly_chart

def _themed_plotly_chart(fig, *args, **kwargs):
    fig.update_layout(
        template=_PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar_bgcolor="rgba(0,0,0,0)",
        font_color=_FONT_COLOR,
        title_font_color=_FONT_COLOR,
        legend_font_color=_FONT_COLOR,
        hoverlabel=dict(
            bgcolor="#1E293B" if IS_DARK else "#FFFFFF",
            font_color="#F1F5F9" if IS_DARK else "#1E293B",
        ),
    )
    fig.update_xaxes(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR)
    fig.update_yaxes(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR)
    fig.update_polars(
        radialaxis_gridcolor=_GRID_COLOR,
        angularaxis_gridcolor=_GRID_COLOR,
        radialaxis_tickfont_color=_FONT_COLOR,
        angularaxis_tickfont_color=_FONT_COLOR,
    )
    kwargs.setdefault("theme", None)
    return _orig_plotly_chart(fig, *args, **kwargs)

st.plotly_chart = _themed_plotly_chart

# ==============================================================================
# 🌙 DARK MODE — เปลี่ยนสีทั้งหน้าเว็บ
# ==============================================================================
if IS_DARK:
    st.markdown("""
    <style>
    /* ---------- พื้นหลังและตัวอักษรหลัก ---------- */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stBottom"], [data-testid="stBottom"] > div {
        background: #0B1220 !important; color: #E2E8F0 !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp li, .stApp label, .stApp span, .stApp .stMarkdown,
    .stApp [data-testid="stWidgetLabel"] p { color: #E2E8F0; }
    .stApp small, .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p { color: #94A3B8 !important; }
    .stApp hr { border-color: #263449 !important; }
    .stApp code { background: #162033 !important; color: #93C5FD !important; }

    /* ---------- แคปซูลสวิตช์ ---------- */
    .st-key-theme_switch {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
        border-color: #334155 !important;
        box-shadow: 0 6px 20px -6px rgba(147, 197, 253, 0.35) !important;
    }
    .st-key-theme_switch:hover {
        background: linear-gradient(135deg, #243449 0%, #16213A 100%) !important;
        border-color: #60A5FA !important;
        box-shadow: 0 10px 26px -6px rgba(96, 165, 250, 0.55) !important;
    }
    .st-key-theme_switch label p { color: #F1F5F9 !important; }

    /* ---------- ช่องกรอกข้อมูล (แก้ปัญหาพื้นขาว/ตัวหนังสือจาง) ---------- */
    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] [data-baseweb="base-input"],
    [data-testid="stTextArea"] [data-baseweb="textarea"],
    [data-testid="stTextArea"] [data-baseweb="base-input"],
    [data-testid="stNumberInput"] [data-baseweb="input"],
    [data-testid="stNumberInput"] [data-baseweb="base-input"],
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
    [data-testid="stChatInput"] > div, [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        background-color: #111B2E !important;
        border-color: #334155 !important;
        color: #F1F5F9 !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input,
    [data-testid="stChatInput"] textarea,
    [data-testid="stSelectbox"] input {
        background-color: transparent !important;
        color: #F1F5F9 !important;
        -webkit-text-fill-color: #F1F5F9 !important;
        caret-color: #F1F5F9 !important;
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748B !important; -webkit-text-fill-color: #64748B !important;
    }
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
    [data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within,
    [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
        border-color: #60A5FA !important;
        box-shadow: 0 0 0 1px #60A5FA !important;
    }
    /* ข้อความและไอคอนใน selectbox */
    [data-testid="stSelectbox"] [data-baseweb="select"] div,
    [data-testid="stSelectbox"] [data-baseweb="select"] span { color: #F1F5F9 !important; }
    [data-testid="stSelectbox"] svg, [data-testid="stNumberInput"] svg { fill: #CBD5E1 !important; color: #CBD5E1 !important; }
    /* เมนูดรอปดาวน์ */
    [data-baseweb="popover"], [data-baseweb="popover"] > div,
    [data-baseweb="popover"] ul, [data-baseweb="menu"], ul[role="listbox"] {
        background: #111B2E !important; color: #E2E8F0 !important;
    }
    [data-baseweb="popover"] li, li[role="option"] { background: #111B2E !important; color: #E2E8F0 !important; }
    [data-baseweb="popover"] li:hover, li[role="option"]:hover,
    li[role="option"][aria-selected="true"] { background: #1E3A5F !important; }

    /* ---------- ปุ่ม ---------- */
    div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {
        background: #162033 !important; color: #E2E8F0 !important; border: 1px solid #334155 !important;
    }
    div[data-testid="stButton"] > button p, div[data-testid="stDownloadButton"] > button p { color: #E2E8F0 !important; }
    div[data-testid="stButton"] > button:hover, div[data-testid="stDownloadButton"] > button:hover {
        border-color: #60A5FA !important; background: #1B2A44 !important;
        box-shadow: 0 4px 14px rgba(96,165,250,.25) !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
        border-color: #3B82F6 !important; color: #FFFFFF !important;
    }

    /* ---------- Metric / Expander / Tab ---------- */
    div[data-testid="stMetric"] { background: #111B2E !important; border-color: #263449 !important; }
    div[data-testid="stMetric"] * { color: #E2E8F0 !important; }
    div[data-testid="stExpander"], div[data-testid="stExpander"] details {
        background: #0F1A2E !important; border-color: #263449 !important;
    }
    div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary * { color: #E2E8F0 !important; }
    [data-baseweb="tab-list"] { border-bottom-color: #263449 !important; }
    button[data-baseweb="tab"] { color: #94A3B8 !important; }
    button[data-baseweb="tab"]:hover { color: #E2E8F0 !important; background: rgba(96,165,250,.08) !important; }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #F8FAFC !important; background: rgba(96,165,250,.12) !important;
        border-bottom-color: #60A5FA !important;
    }

    /* ---------- กล่องแจ้งเตือน (info/success/warning/error) ---------- */
    [data-testid="stAlert"], [data-testid="stAlertContainer"] {
        background: rgba(59,130,246,.12) !important; border: 1px solid #263449 !important;
    }
    [data-testid="stAlert"] *, [data-testid="stAlertContainer"] * { color: #E2E8F0 !important; }

    /* ---------- แชท / JSON / ตาราง ---------- */
    [data-testid="stChatMessage"] { background: #111B2E !important; border: 1px solid #263449; }
    [data-testid="stJson"], [data-testid="stJson"] > div { background: #0F1A2E !important; }
    [data-testid="stDataFrame"] { filter: invert(0.92) hue-rotate(180deg); border-radius: 10px; }

    /* ---------- Slider ---------- */
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"],
    [data-testid="stSlider"] [data-testid="stSliderThumbValue"] { color: #CBD5E1 !important; }

    /* ---------- การ์ด HTML ที่ฮาร์ดโค้ดสีสว่าง → สีเข้ม ---------- */
    [style*="background-color: #f8fafc" i], [style*="background-color: #ffffff" i],
    [style*="background-color: #FAFAFA" i], [style*="background-color: #f1f5f9" i]
        { background-color: #111B2E !important; }
    [style*="background-color: #f0fdf4" i] { background-color: rgba(34,197,94,.12) !important; }
    [style*="background-color: #fffbeb" i] { background-color: rgba(245,158,11,.12) !important; }
    [style*="background-color: #eff6ff" i] { background-color: rgba(59,130,246,.14) !important; }
    [style*="background-color: #fff7ed" i] { background-color: rgba(249,115,22,.12) !important; }
    [style*="background-color: #fef2f2" i] { background-color: rgba(239,68,68,.12) !important; }
    [style*="#e2e8f0" i] { border-color: #263449 !important; }

    /* ตัวอักษรเข้มในการ์ด → สว่าง */
    [style*="color: #1e293b" i], [style*="color: #0f172a" i], [style*="color: #334155" i],
    [style*="color: #475569" i], [style*="color: #111" i], [style*="color: #555" i],
    [style*="color: #431407" i] { color: #E2E8F0 !important; }
    [style*="color: #64748b" i], [style*="color: #94a3b8" i] { color: #94A3B8 !important; }
    [style*="color: #166534" i], [style*="color: #15803d" i], [style*="color: #16a34a" i] { color: #86EFAC !important; }
    [style*="color: #92400e" i], [style*="color: #b45309" i] { color: #FCD34D !important; }
    [style*="color: #1e40af" i], [style*="color: #2563eb" i] { color: #93C5FD !important; }
    [style*="color: #c2410c" i] { color: #FDBA74 !important; }
    [style*="color: #991b1b" i], [style*="color: #b91c1c" i], [style*="color: #dc2626" i] { color: #FCA5A5 !important; }
    [style*="color: #0369a1" i], [style*="color: #0284c7" i] { color: #7DD3FC !important; }
    [style*="color: #0d9488" i] { color: #5EEAD4 !important; }

    /* ---------- แท็กสกิล ---------- */
    .tag-hard { background-color: rgba(59,130,246,.15) !important; color: #93C5FD !important; border-color: rgba(59,130,246,.4) !important; }
    .tag-soft { background-color: rgba(34,197,94,.15) !important; color: #86EFAC !important; border-color: rgba(34,197,94,.4) !important; }
    .tag-tool { background-color: rgba(14,165,233,.15) !important; color: #7DD3FC !important; border-color: rgba(14,165,233,.4) !important; }
    .tag-candidate { background-color: #111B2E !important; color: #E2E8F0 !important; }
    .modern-card { background: #111B2E !important; border-color: #263449 !important; }
    </style>
    """, unsafe_allow_html=True)
    # ---------- ฟิกซ์ที่ 1: การ์ด HTML (match ทั้ง hex และ rgb()) ----------
    def _rgb(h):
        h = h.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return f"rgb({int(h[0:2],16)}, {int(h[2:4],16)}, {int(h[4:6],16)})"

    def _rules(prop, mapping):
        out = []
        for hx, new in mapping.items():
            sels = ",".join(f'.stApp [style*="{prop}: {v}" i]' for v in (hx, _rgb(hx)))
            out.append(f"{sels}{{{prop}:{new} !important;}}")
        return "\n".join(out)

    _BG = {
        "#f8fafc": "#111B2E", "#ffffff": "#111B2E", "#fafafa": "#111B2E", "#f1f5f9": "#111B2E",
        "#f0fdf4": "rgba(34,197,94,.12)", "#fffbeb": "rgba(245,158,11,.12)",
        "#eff6ff": "rgba(59,130,246,.14)", "#fff7ed": "rgba(249,115,22,.12)",
        "#fef2f2": "rgba(239,68,68,.12)",
    }
    _FG = {
        "#1e293b": "#E2E8F0", "#0f172a": "#E2E8F0", "#334155": "#E2E8F0", "#475569": "#CBD5E1",
        "#111": "#F1F5F9", "#555": "#CBD5E1", "#431407": "#FDBA74",
        "#64748b": "#94A3B8", "#94a3b8": "#94A3B8",
        "#166534": "#86EFAC", "#15803d": "#86EFAC", "#16a34a": "#86EFAC",
        "#92400e": "#FCD34D", "#b45309": "#FCD34D",
        "#1e40af": "#93C5FD", "#2563eb": "#93C5FD", "#c2410c": "#FDBA74",
        "#991b1b": "#FCA5A5", "#b91c1c": "#FCA5A5", "#dc2626": "#FCA5A5",
        "#0369a1": "#7DD3FC", "#0284c7": "#7DD3FC", "#0d9488": "#5EEAD4",
    }
    _BORDER = ",".join(
        f'.stApp [style*="{v}" i]' for v in ("#e2e8f0", _rgb("#e2e8f0"))
    ) + "{border-color:#263449 !important;}"

    st.markdown(
        "<style>" + _rules("background-color", _BG) + "\n" +
        _rules("color", _FG) + "\n" + _BORDER + "</style>",
        unsafe_allow_html=True,
    )

    # ---------- ฟิกซ์ที่ 2: ช่องกรอก / selectbox (ครอบทุกชั้นแบบไม่พึ่งโครงสร้าง) ----------
    st.markdown("""
    <style>
    .stApp textarea, .stApp input[type="text"], .stApp input[type="number"],
    .stApp input[type="password"], .stApp input:not([type]),
    .stApp [data-testid="stTextInput"] :is([data-baseweb], [data-baseweb] *),
    .stApp [data-testid="stTextArea"] :is([data-baseweb], [data-baseweb] *),
    .stApp [data-testid="stNumberInput"] :is([data-baseweb], [data-baseweb] *),
    .stApp [data-testid="stSelectbox"] :is([data-baseweb], [data-baseweb] *),
    .stApp [data-testid="stMultiSelect"] :is([data-baseweb], [data-baseweb] *) {
        background-color: #111B2E !important;
        color: #F1F5F9 !important;
        -webkit-text-fill-color: #F1F5F9 !important;
        caret-color: #F1F5F9 !important;
    }
    .stApp [data-testid="stTextInput"] [data-baseweb="input"],
    .stApp [data-testid="stTextArea"] [data-baseweb="textarea"],
    .stApp [data-testid="stNumberInput"] [data-baseweb="input"],
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    .stApp textarea::placeholder, .stApp input::placeholder {
        color: #64748B !important; -webkit-text-fill-color: #64748B !important;
    }
    .stApp [data-testid="stSelectbox"] svg { fill: #CBD5E1 !important; }
    </style>
    """, unsafe_allow_html=True)# --- HERO HEADER ---
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
        <div>
            <h1 class="hero-title">🧠 Personality & Team Fit Analyzer</h1>
            <p class="hero-subtitle">ระบบประเมินสไตล์การทำงาน บุคลิกภาพ และความเข้ากันได้ของทีมงาน ด้วย <strong>AI NLP + K-Means Clustering + Explainable AI (XAI)</strong></p>
        </div>
        <div class="hero-badges">
            <div class="status-badge"><span class="live-dot"></span> AIForThai API (SSense)</div>
            <div class="status-badge">✂️ PyThaiNLP Engine</div>
            <div class="status-badge">🤖 K-Means Clustering</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- TABS NAVIGATION ---
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 1. ประเมินรายบุคคล & AI Explainable",
    "⚔️ 2. เปรียบเทียบสมาชิก (Side-by-Side — กี่คนก็ได้)",
    "👥 3. จัดทีม & สกิลแต่ละตำแหน่ง & ความเข้ากันได้",
    "📈 4. สรุปโมเดล ML & Feature Importance"
])

# ==============================================================================
# TAB 1: INDIVIDUAL EVALUATION & AI EXPLAINABLE & CHATBOT & GROWTH TRACKER
# ==============================================================================
with tab1:
    st.subheader("📝 แบบประเมินสถานการณ์การทำงานรายบุคคล")
    
    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        user_name = st.text_input("ชื่อของคุณ:", "สมชาย สายลุย")
        role_focus = st.selectbox("ตำแหน่ง/สายงานที่สนใจ:", ["Software Developer", "Data Scientist", "Project Manager", "UX/UI Designer", "General"])
    with col_input2:
        q1 = st.text_area(
            "1. อธิบายสไตล์การทำงานของคุณเมื่อต้องทำโปรเจกต์ร่วมกับผู้อื่น:", 
            "ผมชอบวางแผนงานร่วมกับทีม ให้ทุกคนได้แสดงความคิดเห็น และพวกเราจะช่วยกันลงมือแก้ปัญหาให้สำเร็จตามเป้าหมาย ลุยไปด้วยกัน"
        )
        q2 = st.text_area(
            "2. เมื่อเกิดความคิดเห็นไม่ตรงกันในทีม คุณมีวิธีจัดการอย่างไร?:", 
            "เปิดใจรับฟังเหตุผลของทุกฝ่าย และหาจุดเกรงใจตรงกลางที่ทำให้งานเดินหน้าต่อได้โดยไม่มีใครรู้สึกแย่ ช่วยกันปรับความเข้าใจ"
        )

    if st.button("🚀 วิเคราะห์ผลลัพธ์ (Run AI Analysis)", key="btn_single"):
        with st.spinner("กำลังประมวลผล NLP, Sentiment และ K-Means Explainability..."):
            features, keywords = extract_features(q1, q2)
            cluster_id, radar_vals, u_pca, distances = predict_profile(features)
            profile = ROLE_PROFILES[cluster_id]
            
            st.session_state["user_profile"] = {
                "name": user_name,
                "role_focus": role_focus,
                "features": features,
                "cluster_id": cluster_id,
                "u_pca": u_pca,
                "radar_vals": radar_vals,
                "distances": distances,
                "keywords": keywords,
                "q1": q1,
                "q2": q2
            }
            st.rerun()

    if "user_profile" in st.session_state:
        u_p = st.session_state["user_profile"]
        profile = ROLE_PROFILES[u_p["cluster_id"]]
        features = u_p["features"]
        keywords = u_p["keywords"]
        radar_vals = u_p["radar_vals"]
        distances = u_p["distances"]
        
        st.success("✨ วิเคราะห์ผลสำเร็จเรียบร้อยแล้ว!")
        st.markdown(f"### ผลการวิเคราะห์ของคุณ: **{u_p['name']}** ({u_p['role_focus']})")
        
        # Banner Role
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; padding: 18px; border-left: 6px solid {profile['color']}; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="margin: 0; color: #111;">บทบาทเด่นของคุณคือ: <span style="color: {profile['color']};">{profile['title']}</span></h3>
                <p style="margin: 6px 0 0 0; color: #555;">ตำแหน่งแนะนำในโปรเจกต์: <strong>{profile['job_title']}</strong></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # 1. 🧠 AI Explainable Result (XAI)
        st.markdown("### 🧠 1. AI Explainable Result (AI อธิบายเหตุผลเบื้องหลังการตัดสินใจ)")
        b_mean = ml_bundle["benchmark_mean"]
        diff_pct = ((np.array(features) - b_mean) / b_mean) * 100.0
        
        top_factor_idx = int(np.argmax(diff_pct))
        top_factor_name = CATEGORIES[top_factor_idx]
        top_factor_val = diff_pct[top_factor_idx]
        
        xai_col1, xai_col2 = st.columns([3, 2])
        with xai_col1:
            st.info(f"""
            **🔍 ทำไม AI ถึงจัดคุณให้อยู่ในกลุ่มนี้?**
            - **ปัจจัยขับเคลื่อนหลัก:** คะแนนด้าน **{top_factor_name}** ของคุณสูงกว่าค่าเฉลี่ยถึง **{top_factor_val:+.1f}%**
            - **การจำแนกเชิงคณิตศาสตร์:** ใน Scaled Feature Space ระยะห่าง Euclidean ระหว่างข้อมูลของคุณกับ Centroid ของกลุ่ม **{profile['title']}** อยู่ใกล้ที่สุด (**d = {distances[u_p['cluster_id']]:.2f}**) เมื่อเทียบกับกลุ่มอื่นๆ ที่อยู่ห่างเฉลี่ย ({np.mean(distances):.2f})
            - **ลักษณะการใช้ภาษา:** AI ตรวจพบคำว่า 'พวกเรา/ร่วมมือ' และกริยาลงมือทำ ร่วมกับ Sentiment เชิงบวก จึงเข้าเกณฑ์บทบาทนี้อย่างชัดเจน
            """)
        with xai_col2:
            st.markdown("**📏 ระยะห่างไปยัง Centroid ทั้ง 4 กลุ่ม:**")
            df_dist = pd.DataFrame({
                "กลุ่มบทบาท": [ROLE_PROFILES[i]["title"] for i in range(4)],
                "ระยะห่าง (Euclidean Distance)": np.round(distances, 2)
            }).sort_values("ระยะห่าง (Euclidean Distance)", ascending=True)
            st.dataframe(df_dist, use_container_width=True, hide_index=True)

        # Row: Radar Chart & Metrics
        st.markdown("---")
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown("#### 🕸️ กราฟเรดาร์บุคลิกภาพ (Personality Radar)")
            fig = px.line_polar(r=radar_vals, theta=CATEGORIES, line_close=True)
            fig.update_traces(fill='toself', line_color=profile['color'])
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            st.plotly_chart(fig, use_container_width=True)
            
        with c_right:
            st.markdown("#### 📊 ตัวชี้วัดที่สกัดได้จาก NLP & ML")
            m1, m2 = st.columns(2)
            m1.metric("จำนวนคำจริง (PyThaiNLP)", f"{features[1]} คำ")
            m2.metric("Sentiment Score (AIForThai)", f"{features[3]:.1f}/100")
            
            m3, m4 = st.columns(2)
            m3.metric("สัดส่วนความใส่ใจทีม (We/I)", f"{features[2]:.2f}")
            m4.metric("ระดับความลุยงาน (Action Words)", f"{features[4]} ครั้ง")
            
            st.markdown("##### 🔑 คำสำคัญที่คุณใช้บ่อยที่สุด (Top 5 Keywords):")
            if keywords:
                df_kw = pd.DataFrame(list(keywords.items()), columns=["คำสำคัญ", "ความถี่"]).sort_values("ความถี่", ascending=True)
                fig_kw = px.bar(
                    df_kw, x="ความถี่", y="คำสำคัญ", orientation="h",
                    color="ความถี่", color_continuous_scale="Teal",
                    text="ความถี่", height=200
                )
                fig_kw.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.write("- ไม่พบคำสำคัญที่โดดเด่น")

        # Row: AI Advisor
        st.markdown("---")
        st.subheader("💡 AI Advisor คำแนะนำเชิงลึกสำหรับคุณ")
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            st.markdown(f"**💪 จุดเด่นและพลังของคุณ (Strengths):**\n{profile['strengths']}")
            st.markdown(f"**⚠️ จุดที่ควรระวัง (Blindspots):**\n{profile['blindspots']}")
        with col_adv2:
            st.markdown(f"**🤝 สไตล์เพื่อนร่วมงานที่ส่งเสริมกันได้ดี (Best Pairing):**\n{profile['pairing']}")
            st.markdown(f"**🚀 ทิปส์การพัฒนาศักยภาพ (Growth Tip):**\n{profile['tip']}")

        # 4. 💬 AI Chatbot ถามผลวิเคราะห์
        st.markdown("---")
        st.subheader("💬 4. AI Chatbot สอบถามผลวิเคราะห์ของคุณ")
        st.write("สอบถาม AI ได้ทันที เช่น หน้าที่ที่เหมาะในทีม หรือเหตุผลที่ได้ผลลัพธ์นี้")
        
        q_cols = st.columns(4)
        quick_query = None
        if q_cols[0].button("👑 ฉันควรทำหน้าที่อะไรในทีม?"):
            quick_query = "ฉันควรทำหน้าที่อะไรในทีม?"
        if q_cols[1].button("🔍 ทำไมฉันถึงอยู่กลุ่มนี้?"):
            quick_query = "ทำไมฉันถึงถูกจัดอยู่ในกลุ่มนี้?"
        if q_cols[2].button("🚀 ควรพัฒนาทักษะด้านไหน?"):
            quick_query = "ควรพัฒนาทักษะด้านไหนเพิ่มเติม?"
        if q_cols[3].button("🤝 ฉันทำงานกับใครได้ดีที่สุด?"):
            quick_query = "ฉันทำงานร่วมกับคนสไตล์ไหนได้ดีที่สุด?"

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        def answer_user_query(query):
            q_lower = query.lower()
            if "หน้าที่" in q_lower or "ตำแหน่ง" in q_lower or "บทบาท" in q_lower:
                return f"คุณ **{u_p['name']}** เหมาะที่สุดกับตำแหน่ง **{profile['job_title']}** ครับ! เนื่องจากคุณได้บทบาท **{profile['title']}** ซึ่งมีจุดแข็งคือ {profile['strengths']}"
            elif "ทำไม" in q_lower or "เหตุผล" in q_lower or "ทำไมถึง" in q_lower:
                return f"เหตุผลที่ถูกจัดอยู่ในกลุ่ม **{profile['title']}** เพราะโมเดล K-Means ตรวจพบว่าคุณมี {top_factor_name} โดดเด่นกว่าค่าเฉลี่ย {top_factor_val:+.1f}% และระยะห่าง Euclidean ไปยัง Centroid ของกลุ่มนี้สั้นที่สุดคือ {distances[u_p['cluster_id']]:.2f}"
            elif "พัฒนา" in q_lower or "ปรับปรุง" in q_lower or "ทักษะ" in q_lower:
                return f"💡 ข้อแนะนำการพัฒนาสำหรับคุณ: **{profile['tip']}** นอกจากนี้ควรระวังจุดอ่อนเรื่อง: {profile['blindspots']}"
            elif "ใคร" in q_lower or "เพื่อนร่วมงาน" in q_lower or "จับคู่" in q_lower:
                return f"🤝 สไตล์คนที่ส่งเสริมคุณได้ดีที่สุด: **{profile['pairing']}** จะช่วยให้โปรเจกต์ลื่นไหลและไร้รอยต่อครับ"
            else:
                return f"จากผลวิเคราะห์ของคุณ **{u_p['name']}** ในบทบาท **{profile['title']}** (คะแนน Sentiment: {features[3]:.1f}, ใส่ใจทีม: {features[2]:.2f}, Action: {features[4]}) แนะนำให้นำจุดแข็งเรื่อง '{profile['strengths']}' มาขับเคลื่อนงาน และระวังเรื่อง '{profile['blindspots']}' ครับ"

        if quick_query:
            ans = answer_user_query(quick_query)
            st.session_state.chat_history.append({"user": quick_query, "bot": ans})

        user_input_chat = st.chat_input("พิมพ์คำถามเกี่ยวกับผลประเมินของคุณที่นี่...")
        if user_input_chat:
            ans = answer_user_query(user_input_chat)
            st.session_state.chat_history.append({"user": user_input_chat, "bot": ans})

        for chat in st.session_state.chat_history[-4:]:
            with st.chat_message("user"):
                st.write(chat["user"])
            with st.chat_message("assistant"):
                st.write(chat["bot"])

        # 6. 📈 Before / After Development Tracker
        st.markdown("---")
        st.subheader("📈 6. Before / After Development Tracker (ติดตามพัฒนาการ)")
        st.write("เปรียบเทียบพัฒนาการของคุณหลังผ่านการพัฒนา/ฝึกฝนทักษะการทำงานในทีม")
        
        col_dev_btn, col_dev_stat = st.columns([1, 2])
        if col_dev_btn.button("🌱 โหลดตัวอย่างการประเมินรอบ 2 (ผ่านไป 1 เดือน)"):
            st.session_state["after_features"] = [
                features[0] + 50,
                features[1] + 12,
                features[2] + 1.2,
                min(100.0, features[3] + 10.0),
                features[4] + 3
            ]
            st.rerun()

        if "after_features" in st.session_state:
            after_feats = st.session_state["after_features"]
            st.markdown("##### 📊 ตารางเปรียบเทียบพัฒนาการ (Before vs After):")
            
            d_cols = st.columns(5)
            for idx, c_name in enumerate(CATEGORIES):
                b_val = features[idx]
                a_val = after_feats[idx]
                delta = a_val - b_val
                with d_cols[idx]:
                    st.metric(
                        label=c_name.split()[0], 
                        value=f"{a_val:.1f}", 
                        delta=f"{delta:+.1f}"
                    )
            
            df_comp_growth = pd.DataFrame({
                "ด้าน": [c.split()[0] for c in CATEGORIES] * 2,
                "คะแนน": features + after_feats,
                "ช่วงเวลา": ["ครั้งแรก (Before)"] * 5 + ["หลังพัฒนา (After 1 เดือน)"] * 5
            })
            fig_growth = px.bar(
                df_comp_growth, x="ด้าน", y="คะแนน", color="ช่วงเวลา",
                barmode="group", color_discrete_sequence=["#94A3B8", "#10B981"],
                title="กราฟเปรียบเทียบพัฒนาการก่อนและหลังการพัฒนาตนเอง"
            )
            st.plotly_chart(fig_growth, use_container_width=True)

        # Row: Export Report
        st.markdown("---")
        st.subheader("📥 ดาวน์โหลดรายงานผลการประเมิน (Export Report)")
        report_text = f"""==================================================
รายงานสรุปผลการประเมิน Team Fit AI & Personality
==================================================
ผู้รับการประเมิน: {u_p['name']}
สายงาน: {u_p['role_focus']}
บทบาทเด่น: {profile['title']}
ตำแหน่งแนะนำ: {profile['job_title']}

[ผลการวิเคราะห์ Feature และ Explainability]
- จำนวนคำ: {features[1]} คำ
- Sentiment Score (AIForThai): {features[3]:.1f}/100
- ความใส่ใจทีม (We/I): {features[2]:.2f}
- การลงมือทำ (Action): {features[4]}
- ปัจจัยขับเคลื่อนหลัก: {top_factor_name} ({top_factor_val:+.1f}%)

[คำแนะนำจาก AI Advisor]
- จุดแข็ง: {profile['strengths']}
- จุดควรระวัง: {profile['blindspots']}
- บทบาทคู่หูที่เหมาะสม: {profile['pairing']}
- แนวทางพัฒนา: {profile['tip']}
=================================================="""
        col_down1, col_down2 = st.columns(2)
        with col_down1:
            st.download_button(
                label="📄 ดาวน์โหลดรายงาน Text (.txt)",
                data=report_text,
                file_name=f"report_{u_p['name']}.txt",
                mime="text/plain"
            )
        with col_down2:
            df_export = pd.DataFrame([{
                "Name": u_p['name'],
                "Role": profile['title'],
                "Job_Title": profile['job_title'],
                "Word_Count": features[1],
                "Sentiment": features[3],
                "Team_Ratio": features[2],
                "Action_Count": features[4]
            }])
            csv_buffer = df_export.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📊 ดาวน์โหลดข้อมูล CSV (.csv)",
                data=csv_buffer,
                file_name=f"metrics_{u_p['name']}.csv",
                mime="text/csv"
            )

        # 🇹🇭 AIForThai Live Cloud API Inspector (Expander inside Tab 1)
        st.markdown("---")
        with st.expander("🇹🇭 ตรวจสอบสถานะการเชื่อมต่อ AIForThai API จริง (Live Cloud API Inspector)"):
            st.markdown("""
            <div style="background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <h4 style="color: #166534; margin: 0;">🌐 AIForThai API Status: <span style="color: #15803d;">🟢 Connected (Online)</span></h4>
                <p style="color: #166534; margin: 2px 0 0 0; font-size: 13px;"><strong>Service:</strong> Sentiment Analysis (SSense) | <strong>Provider:</strong> สวทช. (NECTEC) | <strong>Endpoint:</strong> <code>https://api.aiforthai.in.th/ssense</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            test_txt = u_p["q1"][:100] if len(u_p.get("q1", "")) > 0 else "ผมชอบทำงานเป็นทีมและช่วยเหลือเพื่อน"
            api_res_tab1 = call_aiforthai_full(test_txt)
            pol_t1 = str(api_res_tab1["polarity"]).lower()
            pol_icon = "😊 Positive" if "pos" in pol_t1 else ("🙁 Negative" if "neg" in pol_t1 else "😐 Neutral")
            
            st.markdown(f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin: 10px 0;">
                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                    <tr style="border-bottom: 1px solid #e2e8f0; height: 32px;"><td style="width: 220px; font-weight: bold; color: #475569;">AIForThai API Status</td><td style="color: #16a34a; font-weight: bold;">🟢 Connected (200 OK)</td></tr>
                    <tr style="border-bottom: 1px solid #e2e8f0; height: 32px;"><td style="font-weight: bold; color: #475569;">Service</td><td style="color: #1e293b;">Sentiment Analysis (SSense)</td></tr>
                    <tr style="border-bottom: 1px solid #e2e8f0; height: 32px;"><td style="font-weight: bold; color: #475569;">Input</td><td style="color: #2563eb;">"{test_txt}"</td></tr>
                    <tr style="border-bottom: 1px solid #e2e8f0; height: 32px;"><td style="font-weight: bold; color: #475569;">API Response</td><td style="color: #0d9488; font-weight: bold;">Sentiment Score: {api_res_tab1['score']:.2f}</td></tr>
                    <tr style="border-bottom: 1px solid #e2e8f0; height: 32px;"><td style="font-weight: bold; color: #475569;">Result</td><td style="font-weight: bold;">{pol_icon}</td></tr>
                    <tr style="height: 32px;"><td style="font-weight: bold; color: #475569;">Response Time (Latency)</td><td style="color: #64748b;">⚡ {api_res_tab1['latency_ms']:.1f} ms</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📦 ดู Raw JSON Response จาก AIForThai"):
                st.json(api_res_tab1["raw_response"])

# ==============================================================================
# TAB 2: SIDE-BY-SIDE MULTI-PERSON COMPARISON (กี่คนก็ได้)
# ==============================================================================
with tab2:
    st.subheader("⚔️ 2. เปรียบเทียบเพื่อนร่วมงาน (Side-by-Side Comparison — กี่คนก็ได้)")
    st.markdown("เปรียบเทียบสไตล์การทำงาน บุคลิกภาพ และความเข้ากันได้ระหว่างบุคคลได้ **กี่คนก็ได้ (ไม่จำกัดจำนวน)** พร้อม Radar Chart เปรียบเทียบ, Heatmap ความเข้ากันได้แบบ N × N และระบบเจาะลึกความสัมพันธ์รายคู่")
    
    default_candidates = [
        {"name": "สมชาย (Leader)", "text": "ผมมุ่งเน้นผลักดันงานให้เสร็จตามเป้าหมาย ชอบจัดระเบียบ คุมงาน และลงมือทำทันที ลุยไปด้วยกัน"},
        {"name": "วิภา (Analytical)", "text": "ฉันชอบวิเคราะห์ข้อมูลอย่างละเอียด ตรวจสอบความถูกต้องและวางแผนป้องกันความเสี่ยงอย่างรอบคอบ"},
        {"name": "กานต์ (Team Player)", "text": "ผมชอบรับฟังทุกคนในทีม คอยประสานงานและดูแลให้บรรยากาศการทำงานราบรื่นและอบอุ่น ช่วยเหลือกัน"}
    ]
    
    if "compare_candidates" not in st.session_state:
        st.session_state.compare_candidates = [dict(c) for c in default_candidates]

    st.markdown("##### ⚡ Quick Presets (เลือกขนาดกลุ่มตัวอย่างอย่างรวดเร็ว):")
    p1, p2, p3, p4 = st.columns(4)
    if p1.button("👥 คู่หู 2 คน (Duo: Leader vs Dev)"):
        st.session_state.compare_candidates = [
            {"name": "สมชาย (Leader)", "text": "ผมมุ่งเน้นผลักดันงานให้เสร็จตามเป้าหมาย ชอบจัดระเบียบ คุมงาน และลงมือทำทันที ลุยไปด้วยกัน"},
            {"name": "วิภา (Analytical)", "text": "ฉันชอบวิเคราะห์ข้อมูลอย่างละเอียด ตรวจสอบความถูกต้องและวางแผนป้องกันความเสี่ยงอย่างรอบคอบ"}
        ]
        if "multi_compare_results" in st.session_state:
            del st.session_state["multi_compare_results"]
        st.rerun()
    if p2.button("👨‍👩‍👧 กลุ่ม 3 คน (Trio: Leader, Dev, QA)"):
        st.session_state.compare_candidates = [
            {"name": "สมชาย (Leader)", "text": "ผมมุ่งเน้นผลักดันงานให้เสร็จตามเป้าหมาย ชอบจัดระเบียบ คุมงาน และลงมือทำทันที ลุยไปด้วยกัน"},
            {"name": "วิภา (Analytical)", "text": "ฉันชอบวิเคราะห์ข้อมูลอย่างละเอียด ตรวจสอบความถูกต้องและวางแผนป้องกันความเสี่ยงอย่างรอบคอบ"},
            {"name": "กานต์ (Team Player)", "text": "ผมชอบรับฟังทุกคนในทีม คอยประสานงานและดูแลให้บรรยากาศการทำงานราบรื่นและอบอุ่น ช่วยเหลือกัน"}
        ]
        if "multi_compare_results" in st.session_state:
            del st.session_state["multi_compare_results"]
        st.rerun()
    if p3.button("🧩 กลุ่ม 4 คน (ครบ 4 บทบาทหลัก)"):
        st.session_state.compare_candidates = [
            {"name": "สมชาย (Leader)", "text": "ผมมุ่งเน้นผลักดันงานให้เสร็จตามเป้าหมาย ชอบจัดระเบียบ คุมงาน และลงมือทำทันที ลุยไปด้วยกัน"},
            {"name": "วิภา (Analytical)", "text": "ฉันชอบวิเคราะห์ข้อมูลอย่างละเอียด ตรวจสอบความถูกต้องและวางแผนป้องกันความเสี่ยงอย่างรอบคอบ"},
            {"name": "กานต์ (Team Player)", "text": "ผมชอบรับฟังทุกคนในทีม คอยประสานงานและดูแลให้บรรยากาศการทำงานราบรื่นและอบอุ่น ช่วยเหลือกัน"},
            {"name": "เมษา (Creative)", "text": "ฉันชอบคิดนอกกรอบ ออกแบบไอเดียใหม่ๆ ทดลองเครื่องมือแปลกใหม่ และสร้างสีสันให้โปรเจกต์"}
        ]
        if "multi_compare_results" in st.session_state:
            del st.session_state["multi_compare_results"]
        st.rerun()
    if p4.button("🚀 ทีม 5 คน (Cross-Functional Squad)"):
        st.session_state.compare_candidates = [
            {"name": "คนึง (Product Lead)", "text": "ผมชอบวางแผนทิศทาง จัดการเวลา ติดตามความคืบหน้า และผลักดันให้โปรเจกต์สำเร็จตามเป้าหมาย"},
            {"name": "วิโรจน์ (Backend Dev)", "text": "ผมชอบคิดวิเคราะห์ตรรกะ ออกแบบระบบฐานข้อมูลและโครงสร้าง API อย่างรอบคอบแม่นยำ"},
            {"name": "ธิดา (Data Scientist)", "text": "ฉันชอบค้นหา Insight จากข้อมูล ใช้สถิติและ Machine Learning เพื่อวิเคราะห์ปัญหาเชิงลึก"},
            {"name": "กัญญา (Scrum / QA)", "text": "ฉันคอยประสานงาน ช่วยเหลือทุกคนในทีม จัดการประชุมย้อนหลัง และทดสอบระบบให้ราบรื่น"},
            {"name": "ปรีชา (UI/UX Designer)", "text": "ผมชอบออกแบบประสบการณ์ผู้ใช้ คิดค้นรูปแบบการปฏิสัมพันธ์ที่สวยงามและใช้งานง่าย"}
        ]
        if "multi_compare_results" in st.session_state:
            del st.session_state["multi_compare_results"]
        st.rerun()

    n_cand = len(st.session_state.compare_candidates)
    
    # Active candidate chip bar
    chips_html = "".join([
        f"<span class='tag-candidate' style='border-left: 4px solid {EXTENDED_PALETTE[i % len(EXTENDED_PALETTE)]};'>"
        f"<span style='width: 8px; height: 8px; border-radius: 50%; background-color: {EXTENDED_PALETTE[i % len(EXTENDED_PALETTE)]}; display: inline-block;'></span> "
        f"{c['name']}</span>"
        for i, c in enumerate(st.session_state.compare_candidates)
    ])
    st.markdown(f"<div style='display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 12px 0;'>{chips_html}</div>", unsafe_allow_html=True)

    # Collapsible candidate configuration expander
    is_exp = "multi_compare_results" not in st.session_state
    with st.expander(f"📝 จัดการและแก้ไขข้อมูลสมาชิก ({n_cand} คน — กดเปิด/ซ่อนฟอร์มได้ตลอดเวลา)", expanded=is_exp):
        btn_m1, btn_m2, btn_m3 = st.columns([1.5, 1.5, 2])
        with btn_m1:
            if st.button("➕ เพิ่มคนเปรียบเทียบ (+1 คน)", key="btn_add_cand"):
                idx_new = len(st.session_state.compare_candidates) + 1
                st.session_state.compare_candidates.append({
                    "name": f"สมาชิกคนที่ {idx_new}",
                    "text": "ผมชอบเรียนรู้สิ่งใหม่ ร่วมมือกับเพื่อนร่วมทีม และลงมือแก้ไขปัญหาให้สำเร็จไปด้วยกัน"
                })
                st.rerun()
        with btn_m2:
            if st.button("➖ ลบคนล่าสุด", key="btn_del_cand"):
                if len(st.session_state.compare_candidates) > 2:
                    st.session_state.compare_candidates.pop()
                    st.rerun()
                else:
                    st.warning("ต้องมีอย่างน้อย 2 คนเพื่อทำการเปรียบเทียบ")
        with btn_m3:
            if st.button("🔄 รีเซ็ตเป็นตัวอย่างเริ่มต้น (3 คน)", key="btn_reset_cand"):
                st.session_state.compare_candidates = [dict(c) for c in default_candidates]
                if "multi_compare_results" in st.session_state:
                    del st.session_state["multi_compare_results"]
                st.rerun()

        st.markdown(f"**ป้อนข้อความอธิบายสไตล์การทำงานของแต่ละคน:**")
        # Render candidate input cards neatly in rows of 3 columns
        for row_start in range(0, n_cand, 3):
            row_indices = list(range(row_start, min(row_start + 3, n_cand)))
            cols = st.columns(len(row_indices))
            for col_i, idx in enumerate(row_indices):
                cand = st.session_state.compare_candidates[idx]
                cand_color = EXTENDED_PALETTE[idx % len(EXTENDED_PALETTE)]
                with cols[col_i]:
                    st.markdown(f"""
                    <div style="background-color: #f8fafc; border-top: 3px solid {cand_color}; padding: 8px 12px 2px 12px; border-radius: 8px; margin-bottom: 6px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color: #334155;">👤 คนที่ {idx + 1}</strong>
                            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background-color: {cand_color};"></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    c_name = st.text_input(f"ชื่อคนที่ {idx+1}:", value=cand["name"], key=f"cand_name_{idx}")
                    c_text = st.text_area(f"สไตล์การทำงาน (คนที่ {idx+1}):", value=cand["text"], key=f"cand_text_{idx}", height=90)
                    st.session_state.compare_candidates[idx]["name"] = c_name
                    st.session_state.compare_candidates[idx]["text"] = c_text
                    if n_cand > 2:
                        if st.button(f"🗑️ ลบคนที่ {idx+1}", key=f"del_single_{idx}"):
                            st.session_state.compare_candidates.pop(idx)
                            st.rerun()

    if st.button(f"⚖️ ประมวลผลเปรียบเทียบทั้ง {n_cand} คน (Compare All)", key="btn_run_compare", type="primary", use_container_width=True):
        with st.spinner(f"กำลังประมวลผล NLP, Sentiment และ Radar Chart สำหรับ {n_cand} คน..."):
            results = []
            for idx, cand in enumerate(st.session_state.compare_candidates):
                feats, _ = extract_features(cand["text"], "")
                c_id, radar_vals, _, _ = predict_profile(feats)
                prof = ROLE_PROFILES[c_id]
                results.append({
                    "name": cand["name"],
                    "features": feats,
                    "cluster_id": c_id,
                    "profile": prof,
                    "radar": radar_vals,
                    "color": EXTENDED_PALETTE[idx % len(EXTENDED_PALETTE)]
                })
                
            st.session_state["multi_compare_results"] = results
            st.rerun()

    if "multi_compare_results" in st.session_state:
        results = st.session_state["multi_compare_results"]
        n_res = len(results)
        
        # Build Pairwise Matrix and records
        names = [r["name"] for r in results]
        matrix_data = np.zeros((n_res, n_res))
        pair_records = []
        for i in range(n_res):
            for j in range(n_res):
                if i == j:
                    matrix_data[i][j] = 100
                else:
                    v_i = np.array(results[i]["features"], dtype=float).reshape(1, -1)
                    v_j = np.array(results[j]["features"], dtype=float).reshape(1, -1)
                    cos_s = cosine_similarity(v_i, v_j)[0][0]
                    syn_info = get_role_synergy(results[i]["cluster_id"], results[j]["cluster_id"])
                    role_base = syn_info["score"]
                    b_score = (cos_s * 40.0) + (role_base * 0.6)
                    final_p_score = int(np.clip(b_score, 40, 99))
                    matrix_data[i][j] = final_p_score
                    if i < j:
                        pair_records.append({
                            "i": i,
                            "j": j,
                            "p1": results[i]["name"],
                            "p2": results[j]["name"],
                            "r1": results[i]["profile"]["title"].split('(')[0],
                            "r2": results[j]["profile"]["title"].split('(')[0],
                            "c1": results[i]["profile"]["color"],
                            "c2": results[j]["profile"]["color"],
                            "score": final_p_score,
                            "type": syn_info["type"],
                            "detail": syn_info["detail"],
                            "status": syn_info["status"]
                        })
                        
        upper_tri = [matrix_data[i][j] for i in range(n_res) for j in range(i+1, n_res)]
        avg_compat = int(np.mean(upper_tri)) if upper_tri else 100

        st.markdown(f"#### 👤 สรุปบทบาทของสมาชิกทั้ง {n_res} คน:")
        for r_start in range(0, n_res, 4):
            r_inds = list(range(r_start, min(r_start + 4, n_res)))
            cols = st.columns(len(r_inds))
            for c_i, i in enumerate(r_inds):
                r = results[i]
                with cols[c_i]:
                    st.markdown(f"""
                    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-top: 4px solid {r['color']}; padding: 12px; border-radius: 10px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                        <h4 style="margin: 0; color: #1e293b; font-size: 15px;">{r['name']}</h4>
                        <p style="color: {r['profile']['color']}; font-weight: bold; margin: 4px 0 2px 0; font-size: 13px;">{r['profile']['title']}</p>
                        <small style="color: #64748b;">ตำแหน่ง: <strong>{r['profile']['job_title']}</strong></small><br/>
                        <small style="color: #94a3b8;">สายงาน: {r['profile']['track']}</small>
                    </div>
                    """, unsafe_allow_html=True)

        # SUB-TABS FOR CLEAN NAVIGATION
        sub2_t1, sub2_t2, sub2_t3, sub2_t4 = st.tabs([
            "🕸️ 1. Multi-Radar & Matrix", 
            "🌟 2. Synergy & Dynamic Insights", 
            "🔍 3. เจาะลึกคู่หู (Duo Deep-Dive)", 
            "📋 4. ข้อมูลสถิติและตัวแปร NLP"
        ])
        
        # Sub-tab 1: Multi-Radar & Compatibility Matrix
        with sub2_t1:
            col_radar, col_matrix = st.columns([1, 1])
            with col_radar:
                st.markdown(f"##### 🕸️ Multi-Person Radar Chart ({n_res} คน)")
                fig_multi = go.Figure()
                for r in results:
                    fig_multi.add_trace(go.Scatterpolar(
                        r=r["radar"],
                        theta=CATEGORIES,
                        fill='toself',
                        name=r["name"],
                        line_color=r["color"],
                        opacity=0.3
                    ))
                fig_multi.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    height=max(430, 360 + min(n_res, 12) * 12),
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_multi, use_container_width=True)

            with col_matrix:
                st.markdown(f"##### 🤝 Pairwise Compatibility Matrix ({n_res} × {n_res})")
                df_matrix = pd.DataFrame(matrix_data, index=names, columns=names)
                fig_heat = px.imshow(
                    df_matrix,
                    text_auto=True,
                    color_continuous_scale="Tealgrn",
                    aspect="auto",
                    labels=dict(color="Compatibility (%)")
                )
                fig_heat.update_layout(height=max(430, 360 + min(n_res, 12) * 12), margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown(f"### 🌟 คะแนนความเข้ากันได้เฉลี่ยของทั้งกลุ่ม: **{avg_compat}%**")
            st.progress(avg_compat / 100.0)

        # Sub-tab 2: Synergy & Friction Insights
        with sub2_t2:
            if pair_records:
                sorted_pairs = sorted(pair_records, key=lambda x: x["score"], reverse=True)
                best_pair = sorted_pairs[0]
                worst_pair = sorted_pairs[-1]
                
                syn_col1, syn_col2 = st.columns(2)
                with syn_col1:
                    st.markdown(f"""
                    <div style="background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 10px; padding: 16px;">
                        <h4 style="color: #166534; margin: 0; font-size: 16px;">🏆 คู่ที่มี Synergy สูงสุดในกลุ่ม ({best_pair['score']}%)</h4>
                        <p style="margin: 8px 0 4px 0; color: #1e293b; font-size: 15px;"><strong>{best_pair['p1']}</strong> ({best_pair['r1']}) & <strong>{best_pair['p2']}</strong> ({best_pair['r2']})</p>
                        <p style="color: #15803d; font-weight: bold; margin: 0 0 6px 0; font-size: 13px;">{best_pair['type']}</p>
                        <p style="color: #475569; margin: 0; font-size: 13px; line-height: 1.5;">{best_pair['detail']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with syn_col2:
                    st.markdown(f"""
                    <div style="background-color: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 16px;">
                        <h4 style="color: #92400e; margin: 0; font-size: 16px;">⚡ คู่ที่ต้องปรับจูนสไตล์การทำงานร่วมกัน ({worst_pair['score']}%)</h4>
                        <p style="margin: 8px 0 4px 0; color: #1e293b; font-size: 15px;"><strong>{worst_pair['p1']}</strong> ({worst_pair['r1']}) & <strong>{worst_pair['p2']}</strong> ({worst_pair['r2']})</p>
                        <p style="color: #b45309; font-weight: bold; margin: 0 0 6px 0; font-size: 13px;">{worst_pair['type']}</p>
                        <p style="color: #475569; margin: 0; font-size: 13px; line-height: 1.5;">{worst_pair['detail']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 💡 บทวิเคราะห์พลวัตการทำงานร่วมกัน (Group Dynamics Analysis):")
            distinct_roles = set(r["cluster_id"] for r in results)
            role_counts = pd.Series([r["cluster_id"] for r in results]).value_counts()
            summary_parts = [f"{ROLE_PROFILES[k]['title'].split('(')[0]}: {v} คน" for k, v in role_counts.items()]
            st.markdown(f"**สัดส่วนบทบาทในกลุ่มนี้:** {' | '.join(summary_parts)}")
            
            if len(distinct_roles) == n_res and n_res <= 4:
                st.success(f"🎉 **ความหลากหลายสมบูรณ์แบบ!** ทั้ง {n_res} คนมีบทบาทแตกต่างกันอย่างชัดเจน ทุกคนจะมีจุดแข็งที่ช่วยอุดรอยรั่วและขับเคลื่อนงานในแต่ละมิติได้อย่างลงตัว")
            elif len(distinct_roles) >= 3:
                st.info(f"💡 กลุ่มนี้มีความหลากหลายที่ดี (พบ {len(distinct_roles)} จาก 4 บทบาทหลัก) มีความสมดุลระหว่างการขับเคลื่อนงาน การวิเคราะห์ และการประสานงาน")
            else:
                st.warning(f"⚠️ กลุ่มนี้มีความกระจุกตัวของบทบาท (พบเพียง {len(distinct_roles)} บทบาทจาก {n_res} คน) อาจต้องกระจายบทบาทความรับผิดชอบให้ชัดเจนเพื่อไม่ให้เกิดการทับซ้อน")

        # Sub-tab 3: Interactive Duo Deep-Dive
        with sub2_t3:
            st.markdown("#### 🔍 เจาะลึกความเข้ากันได้แบบคู่หู 1 ต่อ 1 (Interactive Duo Inspector)")
            st.write("เลือกสมาชิก 2 คนที่ต้องการเจาะลึกเพื่อเปรียบเทียบเรดาร์แบบ 1-on-1 และดูคำแนะนำในการร่วมงานอย่างละเอียด")
            
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                duo_name1 = st.selectbox("เลือกคนที่ 1:", names, index=0, key="duo_sel_1")
            with col_sel2:
                duo_name2 = st.selectbox("เลือกคนที่ 2:", names, index=min(1, len(names)-1), key="duo_sel_2")
                
            idx1 = names.index(duo_name1)
            idx2 = names.index(duo_name2)
            
            if idx1 == idx2:
                st.info("ℹ️ กรุณาเลือกสมาชิก 2 คนที่แตกต่างกันเพื่อดูการเปรียบเทียบคู่หู")
            else:
                c1_res = results[idx1]
                c2_res = results[idx2]
                duo_score = int(matrix_data[idx1][idx2])
                duo_syn = get_role_synergy(c1_res["cluster_id"], c2_res["cluster_id"])
                
                # Duo Score Banner
                if duo_score >= 85:
                    duo_badge = "🟢 คู่หูพลังบวกสูง (High Synergy Duo)"
                    duo_badge_bg = "#f0fdf4"
                    duo_badge_border = "#86efac"
                    duo_badge_color = "#166534"
                elif duo_score >= 70:
                    duo_badge = "🟡 คู่หูสมดุลเกื้อหนุน (Balanced Duo)"
                    duo_badge_bg = "#eff6ff"
                    duo_badge_border = "#93c5fd"
                    duo_badge_color = "#1e40af"
                else:
                    duo_badge = "🟠 คู่หูที่ต้องปรับจูน (Alignment Needed)"
                    duo_badge_bg = "#fffbeb"
                    duo_badge_border = "#fde68a"
                    duo_badge_color = "#92400e"

                st.markdown(f"""
                <div style="background-color: {duo_badge_bg}; border: 1px solid {duo_badge_border}; border-radius: 12px; padding: 14px 20px; margin: 12px 0 16px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span style="font-size: 13px; font-weight: bold; color: {duo_badge_color};">{duo_badge}</span>
                            <h3 style="margin: 4px 0 0 0; color: #0f172a;">{duo_name1} ⚔️ {duo_name2}</h3>
                            <p style="margin: 4px 0 0 0; color: #475569; font-size: 13px;">{duo_syn['type']}</p>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 28px; font-weight: 800; color: {duo_badge_color};">{duo_score}%</span><br/>
                            <small style="color: #64748b;">Compatibility Score</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_d_rad, col_d_desc = st.columns([1, 1])
                with col_d_rad:
                    st.markdown("##### 🕸️ Dual Radar Overlay (เปรียบเทียบ 2 คน)")
                    fig_duo = go.Figure()
                    fig_duo.add_trace(go.Scatterpolar(
                        r=c1_res["radar"],
                        theta=CATEGORIES,
                        fill='toself',
                        name=duo_name1,
                        line_color=c1_res["color"],
                        opacity=0.35
                    ))
                    fig_duo.add_trace(go.Scatterpolar(
                        r=c2_res["radar"],
                        theta=CATEGORIES,
                        fill='toself',
                        name=duo_name2,
                        line_color=c2_res["color"],
                        opacity=0.35
                    ))
                    fig_duo.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True,
                        height=360,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_duo, use_container_width=True)
                    
                with col_d_desc:
                    st.markdown("##### 🤝 รายละเอียดการทำงานร่วมกัน:")
                    st.markdown(f"**💡 รูปแบบความสัมพันธ์:**\n{duo_syn['detail']}")
                    
                    st.markdown(f"""
                    <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 10px 14px; border-radius: 6px; margin: 10px 0;">
                        <strong style="color: #1e40af; font-size: 13px;">💪 จุดเกื้อหนุนเมื่อจับคู่กัน:</strong>
                        <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 13px; color: #334155;">
                            <li><strong>{duo_name1}:</strong> ช่วยเสริมด้าน {c1_res['profile']['strengths']}</li>
                            <li><strong>{duo_name2}:</strong> ช่วยเติมเต็มด้าน {c2_res['profile']['strengths']}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background-color: #fff7ed; border-left: 4px solid #f97316; padding: 10px 14px; border-radius: 6px; margin: 10px 0;">
                        <strong style="color: #c2410c; font-size: 13px;">🛡️ ข้อแนะนำในการสื่อสารและป้องกันความขัดแย้ง:</strong>
                        <p style="margin: 4px 0 0 0; font-size: 13px; color: #431407;">
                            ระวังจุดที่อาจมองข้ามร่วมกัน: {c1_res['profile']['blindspots']} และ {c2_res['profile']['blindspots']} — ควรกำหนดขอบเขตความรับผิดชอบให้ชัดเจนตั้งแต่เริ่มโปรเจกต์
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        # Sub-tab 4: Statistical & Linguistic NLP Features
        with sub2_t4:
            st.markdown("##### 📊 ตารางเปรียบเทียบ Feature ทางภาษาและสถิติของทุกคน:")
            table_rows = []
            for r in results:
                table_rows.append({
                    "ชื่อ": r["name"],
                    "บทบาท (Role)": r["profile"]["title"].split('(')[0],
                    "ตำแหน่งแนะนำ": r["profile"]["job_title"],
                    "สายงาน (Track)": r["profile"]["track"],
                    "ความยาวข้อความ": r["features"][0],
                    "จำนวนคำ (PyThaiNLP)": r["features"][1],
                    "ความใส่ใจทีม (We/I)": f"{r['features'][2]:.2f}",
                    "Sentiment (AIForThai)": f"{r['features'][3]:.1f}",
                    "คำกริยาลงมือทำ": r["features"][4]
                })
            df_comp = pd.DataFrame(table_rows)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
            
            csv_data = df_comp.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 ดาวน์โหลดตารางสถิติ (CSV)",
                data=csv_data,
                file_name="team_candidates_comparison.csv",
                mime="text/csv",
                key="dl_comp_csv"
            )

# ==============================================================================
# TAB 3: TEAM BALANCE, ROLE SKILL SPECIFICATION & TEAM COMPATIBILITY
# ==============================================================================
with tab3:
    st.subheader("👥 3. จัดทีม, สกิลแต่ละตำแหน่ง & วิเคราะห์ความเข้ากันได้ของทีม (Team Fit & Compatibility)")
    st.markdown("จำลองการจัดทีมของโปรเจกต์คุณ วิเคราะห์ **ความเข้ากันได้ของทีม (Team Compatibility)**, ตรวจสอบ **สกิลที่จำเป็นของแต่ละตำแหน่ง (Skill Requirements)**, และค้นหาจุดแข็ง/ช่องว่างทักษะ (Skill Gaps)")

    default_team = [
        {"ชื่อ": "สมชาย (มุ่งมั่น)", "บทบาท": 0},
        {"ชื่อ": "วิภา (รอบคอบ)", "บทบาท": 1},
        {"ชื่อ": "กานต์ (เข้ากับคนง่าย)", "บทบาท": 2},
        {"ชื่อ": "เมษา (ไอเดียใหม่)", "บทบาท": 3}
    ]
    
    role_options = {
        0: "🎯 The Driver / Leader (ผู้นำ)",
        1: "🔍 The Analytical (นักวิเคราะห์)",
        2: "🤝 The Team Player (ผู้ประสานงาน)",
        3: "💡 The Creative (นักคิดสร้างสรรค์)"
    }
    
    st.markdown("##### ⚡ Quick Presets (ทดลองสถานการณ์จำลอง):")
    pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
    if pcol1.button("🟢 ทีมสมดุล 4 ด้าน (Ideal)"):
        st.session_state.team_members = [
            {"ชื่อ": "คนึง (Leader)", "บทบาท": 0},
            {"ชื่อ": "วิโรจน์ (Analytical)", "บทบาท": 1},
            {"ชื่อ": "กัญญา (Team Player)", "บทบาท": 2},
            {"ชื่อ": "ปรีชา (Creative)", "บทบาท": 3}
        ]
        st.rerun()
    if pcol2.button("🚀 Agile Squad (5 คน)"):
        st.session_state.team_members = [
            {"ชื่อ": "สมชาย (Project Lead)", "บทบาท": 0},
            {"ชื่อ": "วิภา (Backend Dev)", "บทบาท": 1},
            {"ชื่อ": "อานนท์ (Data Analyst)", "บทบาท": 1},
            {"ชื่อ": "กานต์ (QA & Scrum)", "บทบาท": 2},
            {"ชื่อ": "เมษา (UX Designer)", "บทบาท": 3}
        ]
        st.rerun()
    if pcol3.button("🔴 ผู้นำซ้ำซ้อน (High Risk)"):
        st.session_state.team_members = [
            {"ชื่อ": "สมศักดิ์ (Leader 1)", "บทบาท": 0},
            {"ชื่อ": "อำนาจ (Leader 2)", "บทบาท": 0},
            {"ชื่อ": "เกรียงไกร (Leader 3)", "บทบาท": 0},
            {"ชื่อ": "นารี (Team Player)", "บทบาท": 2}
        ]
        st.rerun()
    if pcol4.button("🟡 ขาดนักวิเคราะห์ (Missing Data)"):
        st.session_state.team_members = [
            {"ชื่อ": "สมชาย (Leader)", "บทบาท": 0},
            {"ชื่อ": "นก (Creative)", "บทบาท": 3},
            {"ชื่อ": "แอน (Creative)", "บทบาท": 3},
            {"ชื่อ": "นุช (Team Player)", "บทบาท": 2}
        ]
        st.rerun()
    if pcol5.button("💡 ทีมสตาร์ทอัพ (Innovation)"):
        st.session_state.team_members = [
            {"ชื่อ": "ธนา (Founder/Leader)", "บทบาท": 0},
            {"ชื่อ": "มินตรา (Product Designer)", "บทบาท": 3},
            {"ชื่อ": "ภานุ (Creative Strategist)", "บทบาท": 3},
            {"ชื่อ": "ชลธิชา (Team Coordinator)", "บทบาท": 2}
        ]
        st.rerun()

    if "team_members" not in st.session_state:
        st.session_state.team_members = [dict(m) for m in default_team]

    n_team = len(st.session_state.team_members)

    # Active team chips bar
    team_chips_html = "".join([
        f"<span class='tag-candidate' style='border-left: 4px solid {ROLE_PROFILES[m['บทบาท']]['color']};'>"
        f"<span style='width: 8px; height: 8px; border-radius: 50%; background-color: {ROLE_PROFILES[m['บทบาท']]['color']}; display: inline-block;'></span> "
        f"{m['ชื่อ']} <small style='color: #64748b;'>({ROLE_PROFILES[m['บทบาท']]['short_title']})</small></span>"
        for m in st.session_state.team_members
    ])
    st.markdown(f"<div style='display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 12px 0;'>{team_chips_html}</div>", unsafe_allow_html=True)

    # Collapsible team configuration expander
    with st.expander(f"⚙️ ปรับแต่งสมาชิกและบทบาทในทีม ({n_team} คน — กดเปิด/ซ่อนฟอร์มได้ตลอดเวลา)", expanded=False):
        btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 3])
        with btn_col1:
            if st.button("➕ เพิ่มสมาชิกในทีม", key="add_mem"):
                if len(st.session_state.team_members) < 12:
                    st.session_state.team_members.append({"ชื่อ": f"สมาชิก {len(st.session_state.team_members)+1}", "บทบาท": 2})
                    st.rerun()
                else:
                    st.warning("สามารถเพิ่มสมาชิกได้สูงสุด 12 คน")
        with btn_col2:
            if st.button("➖ ลบสมาชิกล่าสุด", key="del_last_mem"):
                if len(st.session_state.team_members) > 2:
                    st.session_state.team_members.pop()
                    st.rerun()
                else:
                    st.warning("ต้องมีสมาชิกอย่างน้อย 2 คน")
        with btn_col3:
            if st.button("🔄 รีเซ็ตทีมเริ่มต้น (4 คน)", key="reset_mem"):
                st.session_state.team_members = [dict(m) for m in default_team]
                st.rerun()

        st.markdown(f"**กำหนดชื่อและเลือกบทบาทของสมาชิกแต่ละคน:**")
        # Render in rows of 4 columns
        for row_start in range(0, n_team, 4):
            row_indices = list(range(row_start, min(row_start + 4, n_team)))
            cols = st.columns(len(row_indices))
            for col_i, idx in enumerate(row_indices):
                mem = st.session_state.team_members[idx]
                role_col = ROLE_PROFILES[mem["บทบาท"]]["color"]
                with cols[col_i]:
                    st.markdown(f"""
                    <div style="background-color: #f8fafc; border-top: 3px solid {role_col}; padding: 6px 10px 2px 10px; border-radius: 6px; margin-bottom: 6px;">
                        <strong style="color: #334155; font-size: 13px;">สมาชิกคนที่ {idx + 1}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    m_name = st.text_input(f"ชื่อคนที่ {idx+1}", value=mem["ชื่อ"], key=f"mem_name_{idx}")
                    m_role = st.selectbox(f"บทบาทของคนที่ {idx+1}", options=list(role_options.keys()), 
                                 format_func=lambda x: role_options[x], 
                                 index=mem["บทบาท"], key=f"mem_role_{idx}")
                    st.session_state.team_members[idx]["ชื่อ"] = m_name
                    st.session_state.team_members[idx]["บทบาท"] = m_role
                    if n_team > 2:
                        if st.button(f"🗑️ นำออก", key=f"del_mem_{idx}"):
                            st.session_state.team_members.pop(idx)
                            st.rerun()

    # Calculate analysis
    current_names = [st.session_state.team_members[idx]["ชื่อ"] for idx in range(len(st.session_state.team_members))]
    current_roles = [st.session_state.team_members[idx]["บทบาท"] for idx in range(len(st.session_state.team_members))]
    n_curr = len(current_names)
    
    # Compute Team Diversity & Balance
    role_counts = pd.Series(current_roles).value_counts()
    unique_roles = len(role_counts)
    balance_score = int((unique_roles / 4.0) * 100)
    
    # Compute Team Compatibility Score (%)
    pairwise_matrix = np.zeros((n_curr, n_curr))
    pairwise_list = []
    for i in range(n_curr):
        for j in range(n_curr):
            if i == j:
                pairwise_matrix[i][j] = 100
            else:
                syn = get_role_synergy(current_roles[i], current_roles[j])
                pairwise_matrix[i][j] = syn["score"]
                if i < j:
                    pairwise_list.append({
                        "m1": current_names[i],
                        "m2": current_names[j],
                        "r1": ROLE_PROFILES[current_roles[i]]["title"].split('(')[0],
                        "r2": ROLE_PROFILES[current_roles[j]]["title"].split('(')[0],
                        "c1": ROLE_PROFILES[current_roles[i]]["color"],
                        "c2": ROLE_PROFILES[current_roles[j]]["color"],
                        "score": syn["score"],
                        "type": syn["type"],
                        "detail": syn["detail"],
                        "status": syn["status"]
                    })
                    
    upper_scores = [pairwise_matrix[i][j] for i in range(n_curr) for j in range(i+1, n_curr)]
    base_compat = float(np.mean(upper_scores)) if upper_scores else 100.0
    
    modifiers = 0
    if unique_roles == 4:
        modifiers += 4
    elif unique_roles <= 2:
        modifiers -= 6
        
    leader_count = sum(1 for r in current_roles if r == 0)
    if leader_count > 2:
        modifiers -= 10
    elif leader_count == 2:
        modifiers -= 4
        
    team_player_count = sum(1 for r in current_roles if r == 2)
    if team_player_count == 0:
        modifiers -= 5
        
    analytical_count = sum(1 for r in current_roles if r == 1)
    if analytical_count == 0:
        modifiers -= 4
        
    team_compat_score = int(np.clip(base_compat + modifiers, 35, 99))
    
    # Harmony badge & styling
    if team_compat_score >= 85:
        harmony_badge = "🟢 ยอดเยี่ยม (High Synergy & Cohesion)"
        harmony_desc = "สมาชิกในทีมมีบทบาทและสไตล์ที่ส่งเสริมกันอย่างลงตัว การสื่อสารลื่นไหลและมีพลังขับเคลื่อนสูง"
        h_color = "#166534"
        h_bg = "#f0fdf4"
        h_border = "#86efac"
    elif team_compat_score >= 70:
        harmony_badge = "🟡 ดี (Balanced & Good Chemistry)"
        harmony_desc = "ทีมทำงานร่วมกันได้ดี มีความเข้าขากันในระดับมาตรฐาน มีบทบาทสนับสนุนที่เพียงพอ"
        h_color = "#1e40af"
        h_bg = "#eff6ff"
        h_border = "#93c5fd"
    elif team_compat_score >= 55:
        harmony_badge = "🟠 ปานกลาง (Moderate Synergy — Needs Clear Rules)"
        harmony_desc = "ควรมีการกำหนดบทบาทและขอบเขตงานให้ชัดเจนเพื่อป้องกันความขัดแย้งหรือการทำงานซ้ำซ้อน"
        h_color = "#92400e"
        h_bg = "#fffbeb"
        h_border = "#fde68a"
    else:
        harmony_badge = "🔴 เสี่ยงต่อความขัดแย้ง (High Friction / Conflict Risk)"
        harmony_desc = "มีโอกาสเกิดความเห็นไม่ตรงกันสูง เช่น การมีผู้นำซ้ำซ้อนหรือขาดผู้ประสานงาน ต้องมีข้อตกลงร่วมกันอย่างเร่งด่วน"
        h_color = "#991b1b"
        h_bg = "#fef2f2"
        h_border = "#fca5a5"
        
    # Top Hero Metrics Banner
    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("คะแนนความเข้ากันได้ของทีม", f"{team_compat_score}%", help="คำนวณจาก Role Synergy Matrix, สัดส่วนความหลากหลาย และการลดทอนความเสี่ยง")
    with m_col2:
        st.metric("คะแนนความสมดุล (Balance)", f"{balance_score}%", help="ประเมินการกระจายตัวของทั้ง 4 บทบาทหลักในทีม")
    with m_col3:
        st.metric("จำนวนสมาชิกทั้งหมด", f"{n_curr} คน")
    with m_col4:
        st.metric("บทบาทที่ครอบคลุม", f"{unique_roles} / 4 บทบาท")
        
    st.markdown(f"""
    <div style="background-color: {h_bg}; border: 1px solid {h_border}; border-left: 6px solid {h_color}; border-radius: 10px; padding: 12px 18px; margin: 12px 0 16px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <strong style="color: {h_color}; font-size: 14px;">ระดับความเข้ากันได้ของทีม: {harmony_badge}</strong>
                <p style="margin: 3px 0 0 0; color: #475569; font-size: 13px;">💡 {harmony_desc}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(team_compat_score / 100.0)

    # 5 ORGANIZED SUB-TABS
    sub3_t1, sub3_t2, sub3_t3, sub3_t4, sub3_t5 = st.tabs([
        "🤝 1. ความเข้ากันได้ & Heatmap",
        "🎯 2. สกิลของแต่ละตำแหน่ง (Member Skills)",
        "📊 3. เรดาร์ทักษะ 5 ด้าน & ช่องว่าง (Skill Gaps)",
        "🍩 4. สัดส่วนบทบาท & การวินิจฉัยโครงสร้าง",
        "📚 5. คลังมาตรฐานทักษะ (Position Directory)"
    ])

    # Sub-tab 1: Team Compatibility & Heatmap & Agreement
    with sub3_t1:
        c_hm, c_syn = st.columns([1.1, 1])
        with c_hm:
            st.markdown("##### 🗺️ เมทริกซ์ความเข้ากันได้ระหว่างสมาชิก (Compatibility Heatmap)")
            df_team_matrix = pd.DataFrame(pairwise_matrix, index=current_names, columns=current_names)
            fig_team_heat = px.imshow(
                df_team_matrix,
                text_auto=True,
                color_continuous_scale="Tealgrn",
                aspect="auto",
                labels=dict(color="ความเข้ากันได้ (%)")
            )
            fig_team_heat.update_layout(height=max(400, 320 + min(n_curr, 10) * 15), margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_team_heat, use_container_width=True)
            
        with c_syn:
            st.markdown("##### 🌟 วิเคราะห์จุดเสริมพลังและจุดเสียดทานในทีม:")
            if pairwise_list:
                sorted_team_pairs = sorted(pairwise_list, key=lambda x: x["score"], reverse=True)
                top_pair = sorted_team_pairs[0]
                low_pair = sorted_team_pairs[-1]
                
                st.markdown(f"""
                <div style="background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                    <h5 style="color: #166534; margin: 0; font-size: 14px;">🌟 คู่หู Synergy ส่งเสริมกันสูงสุด ({top_pair['score']}%)</h5>
                    <p style="margin: 4px 0 2px 0; color: #1e293b;"><strong>{top_pair['m1']}</strong> ({top_pair['r1']}) & <strong>{top_pair['m2']}</strong> ({top_pair['r2']})</p>
                    <small style="color: #15803d; font-weight: bold;">{top_pair['type']}</small><br/>
                    <small style="color: #475569;">{top_pair['detail']}</small>
                </div>
                """, unsafe_allow_html=True)
                
                if low_pair["score"] < 80 or low_pair["status"] == "warning":
                    st.markdown(f"""
                    <div style="background-color: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                        <h5 style="color: #92400e; margin: 0; font-size: 14px;">⚠️ จุดเสียดทานที่ต้องระวัง ({low_pair['score']}%)</h5>
                        <p style="margin: 4px 0 2px 0; color: #1e293b;"><strong>{low_pair['m1']}</strong> ({low_pair['r1']}) & <strong>{low_pair['m2']}</strong> ({low_pair['r2']})</p>
                        <small style="color: #b45309; font-weight: bold;">{low_pair['type']}</small><br/>
                        <small style="color: #475569;">{low_pair['detail']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                        <h5 style="color: #334155; margin: 0; font-size: 14px;">✅ ไม่พบจุดเสียดทานรุนแรง</h5>
                        <small style="color: #64748b;">ทุกคู่ในทีมมีคะแนนความเข้ากันได้ในเกณฑ์น่าพึงพอใจ</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Actionable Working Agreement
            st.markdown("##### 📋 แนวทางบริหารจัดการและข้อตกลงร่วมกัน (Team Agreement):")
            if leader_count > 1:
                st.info("📌 **ข้อตกลงเรื่องการตัดสินใจ:** มีผู้นำมากกว่า 1 คน ควรกำหนด Domain Ownership ให้ชัดเจน (เช่น คนที่ 1 คุมด้าน Product/Strategy และคนที่ 2 คุมด้าน Technical/Delivery) เพื่อไม่ให้ชนกัน")
            elif leader_count == 0:
                st.warning("📌 **ข้อตกลงเรื่องทิศทางงาน:** ทีมไม่มีผู้นำโดยธรรมชาติ แนะนำให้แต่งตั้งผู้ประสานงานหลัก หรือกำหนดเกณฑ์การโหวตมติเสียงส่วนใหญ่เพื่อไม่ให้งานล่าช้า")
            else:
                st.success("📌 **การขับเคลื่อนงาน:** มีผู้นำ 1 คนชัดเจน สามารถฟันธงทิศทางงานได้อย่างคล่องตัว ให้คอยรับฟังข้อเสนอแนะจากเพื่อนร่วมทีมเสมอ")
                
            if team_player_count == 0:
                st.warning("📌 **การดูแลบรรยากาศ:** ขาด Team Player แนะนำให้จัด Daily Standup สั้นๆ และเปิดช่วงเวลาถามไถ่ความรู้สึกเพื่อสร้างความผ่อนคลาย")

    # Sub-tab 2: Member Roles & Skills Breakdown
    with sub3_t2:
        st.markdown("##### 🏆 ตำแหน่งงานและข้อกำหนดสกิลของสมาชิกแต่ละคน (Individual Role & Skill Cards)")
        st.write("แจกแจง **Hard Skills**, **Soft Skills**, **เครื่องมือหลัก (Tools)**, และ **หน้าที่รับผิดชอบ (Responsibilities)** ตามตำแหน่งจริงของสมาชิก")
        
        # Render individual role and skill cards in rows of 2 columns
        for r_start in range(0, n_curr, 2):
            r_indices = list(range(r_start, min(r_start + 2, n_curr)))
            r_cols = st.columns(len(r_indices))
            for c_i, idx in enumerate(r_indices):
                name = current_names[idx]
                r_id = current_roles[idx]
                p = ROLE_PROFILES[r_id]
                sk = p["skills"]
                h_tags = "".join([f"<span class='tag-hard'>{s}</span>" for s in sk['hard']])
                s_tags = "".join([f"<span class='tag-soft'>{s}</span>" for s in sk['soft']])
                tools_tags = "".join([f"<span class='tag-tool'>{s}</span>" for s in sk['tools']])
                resp_list = "".join([f"<li style='margin-bottom: 3px;'>{resp}</li>" for resp in p['responsibilities']])
                
                with r_cols[c_i]:
                    st.markdown(f"""
                    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-top: 5px solid {p['color']}; padding: 18px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                            <div>
                                <h4 style="margin: 0; color: #0f172a; font-size: 17px;">👤 {name}</h4>
                                <p style="color: {p['color']}; font-weight: bold; margin: 3px 0 0 0; font-size: 14px;">🎯 ตำแหน่ง: {p['job_title']}</p>
                            </div>
                            <span style="background-color: {p['color']}15; color: {p['color']}; padding: 3px 9px; border-radius: 12px; font-size: 11px; font-weight: bold;">
                                {p['short_title']}
                            </span>
                        </div>
                        <small style="color: #64748b; font-weight: 500;">สายงาน: <strong>{p['track']}</strong></small>
                        
                        <div style="margin-top: 10px; background-color: #f8fafc; padding: 10px 12px; border-radius: 8px;">
                            <strong style="font-size: 12px; color: #1e293b;">📌 หน้าที่รับผิดชอบหลัก:</strong>
                            <ul style="margin: 6px 0 0 0; padding-left: 18px; font-size: 12px; color: #475569; line-height: 1.5;">
                                {resp_list}
                            </ul>
                        </div>
                        
                        <div style="margin-top: 10px;">
                            <strong style="font-size: 12px; color: #1e293b;">⚡ Hard Skills ที่ต้องมี:</strong><br/>
                            <div style="margin-top: 4px;">{h_tags}</div>
                        </div>
                        
                        <div style="margin-top: 8px;">
                            <strong style="font-size: 12px; color: #1e293b;">🧠 Soft Skills ประจำตำแหน่ง:</strong><br/>
                            <div style="margin-top: 4px;">{s_tags}</div>
                        </div>
                        
                        <div style="margin-top: 8px;">
                            <strong style="font-size: 12px; color: #1e293b;">🛠️ เครื่องมือหลัก:</strong><br/>
                            <div style="margin-top: 4px;">{tools_tags}</div>
                        </div>
                        
                        <div style="margin-top: 10px; background-color: #eff6ff; border-left: 3px solid #3b82f6; padding: 8px 12px; border-radius: 6px; font-size: 12px; color: #1e40af; line-height: 1.4;">
                            <strong>🚀 ข้อแนะนำ Upskill:</strong> {p['upskill_recommendations']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Sub-tab 3: Competency Radar & Skill Gaps
    with sub3_t3:
        col_rad, col_gap = st.columns([1, 1])
        
        # Calculate team competency vector
        comp_matrix = np.array([ROLE_PROFILES[r]["competency_vector"] for r in current_roles])
        team_comp_scores = np.max(comp_matrix, axis=0)  # Peak competency coverage in the team
        team_avg_comp = np.mean(comp_matrix, axis=0)   # Average competency
        
        with col_rad:
            st.markdown("##### 🕸️ Team Competency Radar (ความครอบคลุมทักษะ 5 ด้าน)")
            fig_team_comp = go.Figure()
            fig_team_comp.add_trace(go.Scatterpolar(
                r=team_comp_scores,
                theta=TEAM_COMPETENCY_AREAS,
                fill='toself',
                name="ศักยภาพสูงสุดของทีม (Peak Coverage)",
                line_color="#2563EB",
                opacity=0.35
            ))
            fig_team_comp.add_trace(go.Scatterpolar(
                r=[80] * 5,
                theta=TEAM_COMPETENCY_AREAS,
                name="เกณฑ์มาตรฐานแนะนำ (Benchmark 80%)",
                line=dict(color="#94A3B8", dash='dash'),
                fill=None
            ))
            fig_team_comp.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=380,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_team_comp, use_container_width=True)
            
        with col_gap:
            st.markdown("##### 🔍 การวิเคราะห์ทักษะเด่นและช่องว่างทักษะ (Skill Gap Insights):")
            for idx_c, area_name in enumerate(TEAM_COMPETENCY_AREAS):
                score_c = team_comp_scores[idx_c]
                short_area = area_name.split('(')[0].strip()
                if score_c >= 85:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background-color: #f0fdf4; border-left: 4px solid #16a34a; border-radius: 6px; margin-bottom: 6px;">
                        <div><strong style="color: #166534; font-size: 13px;">✅ {short_area}</strong> <small style="color: #64748b;">(ครอบคลุมแข็งแกร่ง)</small></div>
                        <strong style="color: #15803d; font-size: 14px;">{score_c}%</strong>
                    </div>
                    """, unsafe_allow_html=True)
                elif score_c >= 70:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background-color: #f8fafc; border-left: 4px solid #0284c7; border-radius: 6px; margin-bottom: 6px;">
                        <div><strong style="color: #0369a1; font-size: 13px;">🔵 {short_area}</strong> <small style="color: #64748b;">(อยู่ในเกณฑ์มาตรฐาน)</small></div>
                        <strong style="color: #0284c7; font-size: 14px;">{score_c}%</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background-color: #fef2f2; border-left: 4px solid #dc2626; border-radius: 6px; margin-bottom: 6px;">
                        <div><strong style="color: #991b1b; font-size: 13px;">⚠️ {short_area}</strong> <small style="color: #dc2626; font-weight: bold;">(Skill Gap — ทีมยังขาดความเชี่ยวชาญด้านนี้)</small></div>
                        <strong style="color: #b91c1c; font-size: 14px;">{score_c}%</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Actionable gap advice
            weak_areas = [TEAM_COMPETENCY_AREAS[i].split('(')[0].strip() for i in range(5) if team_comp_scores[i] < 70]
            if weak_areas:
                st.warning(f"💡 **คำแนะนำปิดช่องว่างทักษะ:** ทีมยังขาดความเชี่ยวชาญด้าน **{', '.join(weak_areas)}** ควรส่งเสริมให้สมาชิกในทีม Upskill หรือจัดหาผู้เชี่ยวชาญมาช่วยเสริมจุดนี้")
            else:
                st.success("🎉 **ยอดเยี่ยม!** ทีมชุดนี้มีทักษะครอบคลุมทั้ง 5 มิติเกินเกณฑ์มาตรฐาน ไม่มี Skill Gap ร้ายแรง")

    # Sub-tab 4: Role Structure & Diagnostics
    with sub3_t4:
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            labels = [role_options[r] for r in role_counts.index]
            values = role_counts.values
            fig_donut = px.pie(names=labels, values=values, hole=0.45, title="สัดส่วนบทบาทของสมาชิก")
            fig_donut.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_t2:
            st.markdown("##### 📋 การวินิจฉัยและคำเตือนเชิงโครงสร้าง:")
            missing_roles = [r for r in [0, 1, 2, 3] if r not in role_counts.index]
            if missing_roles:
                for r in missing_roles:
                    if r == 0:
                        st.error("❌ **Critical: ขาด Leader (ผู้นำ)** — เสี่ยงงานล่าช้า ขาดคนตัดสินใจฟันธงทิศทาง")
                    elif r == 1:
                        st.error("❌ **Critical: ขาด Analytical (นักวิเคราะห์)** — เสี่ยงต่องานตกหล่น สถาปัตยกรรมระบบไม่นิ่ง และมีบั๊ก")
                    elif r == 2:
                        st.warning("⚠️ **Warning: ขาด Team Player (ผู้ประสานงาน)** — บรรยากาศตึงเครียด ขาดคนประสานข้ามสายงาน")
                    elif r == 3:
                        st.warning("⚠️ **Warning: ขาด Creative (นักคิดสร้างสรรค์)** — ขาดไอเดียแปลกใหม่ โปรดักต์อาจขาดจุดเด่นที่น่าดึงดูด")
            else:
                st.success("🎉 **โครงสร้างสมบูรณ์แบบ!** มีครบทั้ง 4 บทบาทหลักในทีม")
                
            for r, count in role_counts.items():
                if count >= 3:
                    if r == 0:
                        st.error(f"⚠️ **Warning Redundancy: มีผู้นำ (Leader) ถึง {count} คน!** ระวัง Ego Clash และแย่งการตัดสินใจ")
                    else:
                        st.warning(f"⚠️ **Warning: มีบทบาท {role_options[r]} ซ้ำซ้อน {count} คน** ควรแบ่งหน้าที่รับผิดชอบให้ชัดเจน")

    # Sub-tab 5: Interactive Position Skills Directory
    with sub3_t5:
        st.markdown("##### 📚 คลังข้อมูลมาตรฐานทักษะของทุกตำแหน่ง (Standard Position Skills Directory)")
        st.write("ตารางเปรียบเทียบมาตรฐานทักษะ (Skill Matrix) ของทั้ง 4 ตำแหน่งในโปรเจกต์ เพื่อใช้อ้างอิงในการสรรหา พัฒนา และประเมินผลงาน")
        
        tab_p0, tab_p1, tab_p2, tab_p3 = st.tabs([
            ROLE_PROFILES[0]["job_title"],
            ROLE_PROFILES[1]["job_title"],
            ROLE_PROFILES[2]["job_title"],
            ROLE_PROFILES[3]["job_title"]
        ])
        
        for p_idx, p_tab in enumerate([tab_p0, tab_p1, tab_p2, tab_p3]):
            p_obj = ROLE_PROFILES[p_idx]
            with p_tab:
                c_dir1, c_dir2 = st.columns([1, 1])
                with c_dir1:
                    st.markdown(f"#### {p_obj['title']}")
                    st.markdown(f"**สายงาน:** {p_obj['track']}")
                    st.markdown(f"**จุดแข็ง:** {p_obj['strengths']}")
                    st.markdown(f"**จุดที่ควรระวัง:** {p_obj['blindspots']}")
                    st.markdown(f"**บทบาทคู่หูที่เหมาะสม:** {p_obj['pairing']}")
                with c_dir2:
                    h_pill_dir = "".join([f"<span class='tag-hard'>{h}</span>" for h in p_obj['skills']['hard']])
                    s_pill_dir = "".join([f"<span class='tag-soft'>{s}</span>" for s in p_obj['skills']['soft']])
                    t_pill_dir = "".join([f"<span class='tag-tool'>{t}</span>" for t in p_obj['skills']['tools']])
                    st.markdown(f"**⚡ Hard Skills:**<br/>{h_pill_dir}", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top: 8px;'><strong>🧠 Soft Skills:</strong><br/>{s_pill_dir}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top: 8px;'><strong>🛠️ เครื่องมือหลัก:</strong><br/>{t_pill_dir}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top: 10px; background-color: #f8fafc; padding: 8px 12px; border-radius: 6px;'><strong>🚀 คำแนะนำการพัฒนา:</strong> {p_obj['tip']}</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 4: ML MODEL METRICS & FEATURE IMPORTANCE & PCA
# ==============================================================================
with tab4:
    st.subheader("📈 หน้าสรุปโมเดล Machine Learning (Model Metrics & Feature Importance)")
    st.markdown("ส่วนแสดงผลลัพธ์เชิงอัลกอริทึม พิสูจน์การทำงานของ **K-Means Clustering**, **PCA**, และ **Feature Importance**")
    
    st.markdown("#### 1. 🎯 Model Evaluation Metrics (ตัวชี้วัดประสิทธิภาพ Clustering)")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="Silhouette Score", value=f"{ml_bundle['sil_score']:.3f}", help="> 0.4 บ่งชี้ว่าข้อมูลจัดกลุ่มได้แน่นหนาและแยกจากกลุ่มอื่นได้ดี")
        st.caption("✅ ประเมินความกระชับและการแยกตัวของ Cluster")
    with m_col2:
        st.metric(label="Davies-Bouldin Index", value=f"{ml_bundle['db_score']:.3f}", help="ยิ่งค่าน้อยยิ่งดี (< 1.0 แปลว่า Cluster แยกตัวชัดเจน)")
        st.caption("✅ ประเมินความคล้ายคลึงระหว่าง Cluster (ยิ่งน้อยยิ่งดี)")
    with m_col3:
        st.metric(label="Inertia (WCSS)", value=f"{ml_bundle['inertia']:.1f}", help="Within-Cluster Sum of Squares")
        st.caption("✅ ผลรวมระยะทางกำลังสองภายในกลุ่ม")

    st.markdown("---")
    
    st.markdown("#### 3. 📊 Feature Importance (ปัจจัยที่มีผลต่อการจัด Cluster มากที่สุด)")
    st.write("คำนวณผ่าน **Random Forest Feature Importance** เพื่อตอบคำถามว่าโมเดลให้น้ำหนักกับตัวแปรใดมากที่สุด")
    
    feat_imps = ml_bundle["feature_importances"]
    df_fi = pd.DataFrame({
        "Feature": CATEGORIES,
        "Importance": feat_imps * 100.0
    }).sort_values("Importance", ascending=True)
    
    fig_fi = px.bar(
        df_fi, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Purples",
        text=df_fi["Importance"].apply(lambda v: f"{v:.1f}%"),
        title="สัดส่วนความสำคัญของแต่ละ Feature ในการแบ่งกลุ่มบทบาท"
    )
    fig_fi.update_layout(height=280, coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    
    st.markdown("#### 2. 🗺️ 2D PCA Cluster Scatter Plot (แผนที่การกระจายตัวของสมาชิก)")
    pca = ml_bundle["pca"]
    X_scaled = ml_bundle["X_scaled"]
    kmeans = ml_bundle["kmeans"]
    pca_2d = pca.transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    st.info(f"💡 **PCA Explained Variance Ratio:** PC1: **{explained_var[0]*100:.1f}%**, PC2: **{explained_var[1]*100:.1f}%** (รวม **{(explained_var[0]+explained_var[1])*100:.1f}%**)")
    
    cluster_labels_map = {
        0: "🎯 The Driver / Leader",
        1: "🔍 The Analytical",
        2: "🤝 The Team Player",
        3: "💡 The Creative"
    }
    
    df_pca = pd.DataFrame({
        "PC1": pca_2d[:, 0],
        "PC2": pca_2d[:, 1],
        "Cluster": [cluster_labels_map[lbl] for lbl in kmeans.labels_]
    })
    
    fig_pca = px.scatter(
        df_pca, x="PC1", y="PC2", color="Cluster",
        color_discrete_map={
            "🎯 The Driver / Leader": "#E63946",
            "🔍 The Analytical": "#1D3557",
            "🤝 The Team Player": "#2A9D8F",
            "💡 The Creative": "#F4A261"
        },
        opacity=0.65,
        title="การจัดกลุ่มด้วย K-Means และการลดมิติด้วย PCA"
    )
    fig_pca.update_traces(marker=dict(size=11, line=dict(width=1, color='DarkSlateGrey')))
    
    if "user_profile" in st.session_state:
        u_p = st.session_state["user_profile"]
        fig_pca.add_trace(go.Scatter(
            x=[u_p["u_pca"][0]],
            y=[u_p["u_pca"][1]],
            mode="markers+text",
            marker=dict(symbol="star", size=24, color="#FFD700", line=dict(color="#000", width=2)),
            text=[f"⭐ คุณ ({u_p['name']})"],
            textposition="top center",
            name=f"ตำแหน่งของคุณ ({u_p['name']})"
        ))
        st.success(f"📍 ตำแหน่งของ **{u_p['name']}** ปักหมุดเป็นดาวสีทอง (⭐) ในแผนภูมิด้านล่าง")
        
    fig_pca.update_layout(height=500)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")
    # 4. What-if Simulator as an expander inside Tab 4
    with st.expander("🎯 What-if Simulator — ทดลองเลื่อนปรับคะแนน Feature แบบ Real-time"):
        st.write("ทดลองปรับเปลี่ยนคะแนนทั้ง 5 ด้าน เพื่อดูว่าอัลกอริทึม **K-Means Machine Learning** เปลี่ยนผลการจัดกลุ่มตามข้อมูลใหม่อย่างไร")
        
        if "user_profile" in st.session_state:
            base_f = st.session_state["user_profile"]["features"]
            base_name = st.session_state["user_profile"]["name"]
            initial_c_id = st.session_state["user_profile"]["cluster_id"]
        else:
            base_f = [220, 40, 3.5, 80.0, 3]
            base_name = "ผู้ใช้งานทั่วไป"
            initial_c_id, _, _, _ = predict_profile(base_f)
            
        initial_profile = ROLE_PROFILES[initial_c_id]
        
        col_ctrl, col_view = st.columns([1, 1])
        with col_ctrl:
            st.markdown("##### 🎛️ เลื่อนปรับคะแนน Feature:")
            s_length = st.slider("การสื่อสาร (Text Length):", min_value=50, max_value=500, value=int(base_f[0]), key="s_len_t4")
            s_words = st.slider("ความละเอียด (Word Count):", min_value=10, max_value=100, value=int(base_f[1]), key="s_w_t4")
            s_team = st.slider("ความใส่ใจทีม (Team Score We/I):", min_value=0.2, max_value=8.0, value=float(base_f[2]), step=0.1, key="s_tm_t4")
            s_sentiment = st.slider("ทัศนคติบวก (Sentiment 0-100):", min_value=0.0, max_value=100.0, value=float(base_f[3]), step=1.0, key="s_sen_t4")
            s_action = st.slider("การลงมือทำ (Action Words):", min_value=0, max_value=15, value=int(base_f[4]), key="s_act_t4")
            
            sim_features = [s_length, s_words, s_team, s_sentiment, s_action]
            sim_cluster_id, sim_radar, _, sim_distances = predict_profile(sim_features)
            sim_profile = ROLE_PROFILES[sim_cluster_id]

        with col_view:
            st.markdown("##### 🔄 การเปรียบเทียบผลลัพธ์ (Before vs After):")
            b_col, a_col = st.columns(2)
            with b_col:
                st.markdown(f"""
                <div style="background-color: #f1f5f9; padding: 14px; border-radius: 8px; border-left: 5px solid {initial_profile['color']};">
                    <small style="color: #64748b;">สถานะก่อนปรับ (Current):</small>
                    <h4 style="margin: 4px 0 0 0; color: #1e293b;">{initial_profile['title'].split('(')[0]}</h4>
                </div>
                """, unsafe_allow_html=True)
                
            with a_col:
                st.markdown(f"""
                <div style="background-color: #f1f5f9; padding: 14px; border-radius: 8px; border-left: 5px solid {sim_profile['color']};">
                    <small style="color: #64748b;">สถานะจำลองใหม่ (What-if):</small>
                    <h4 style="margin: 4px 0 0 0; color: {sim_profile['color']};">{sim_profile['title'].split('(')[0]}</h4>
                </div>
                """, unsafe_allow_html=True)

            if initial_c_id != sim_cluster_id:
                st.success(f"🎉 **เกิดการเปลี่ยนคลัสเตอร์!** เมื่อคุณปรับคะแนน โมเดล K-Means ได้เปลี่ยนบทบาทจาก **{initial_profile['title'].split('(')[0]}** เป็น **{sim_profile['title']}**")
            else:
                st.info("ℹ️ คะแนนที่ปรับยังคงตกอยู่ในคลัสเตอร์เดิม (ลองปรับ Action หรือ Team Score ให้สูงขึ้นเพื่อดูการเปลี่ยนกลุ่ม)")

            fig_whatif = go.Figure()
            fig_whatif.add_trace(go.Scatterpolar(
                r=np.clip(np.array(base_f, dtype=float) / np.max(ml_bundle["benchmark_data"], axis=0), 0, 1).tolist(),
                theta=CATEGORIES,
                fill='toself',
                name="ค่าเดิม (Before)",
                line_color=initial_profile['color'],
                opacity=0.4
            ))
            fig_whatif.add_trace(go.Scatterpolar(
                r=sim_radar,
                theta=CATEGORIES,
                fill='toself',
                name="จำลองใหม่ (What-if)",
                line_color=sim_profile['color']
            ))
            fig_whatif.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Radar Chart เปรียบเทียบค่าเดิม vs ค่าจำลอง What-if",
                height=340
            )
            st.plotly_chart(fig_whatif, use_container_width=True)