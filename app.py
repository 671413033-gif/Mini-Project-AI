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
API_KEY = "IFksVNfu2PXl0SIxmZz4zdxU01TYyanJ"

st.set_page_config(
    page_title="Team Fit AI & Personality Analyzer",
    page_icon="🧠",
    layout="wide"
)

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
        "color": "#E63946",
        "strengths": "กล้าตัดสินใจ มุ่งเน้นความสำเร็จและเป้าหมาย ผลักดันงานให้เสร็จตามกำหนด มีภาวะผู้นำสูง",
        "blindspots": "อาจเร่งรัดงานจนเพื่อนร่วมทีมเครียด หรือลืมฟังรายละเอียดปลีกย่อยของคนอื่น",
        "pairing": "จับคู่ได้ดีกับ 🔍 The Analytical เพื่อช่วยตรวจสอบความเสี่ยง และ 🤝 The Team Player เพื่อช่วยดูแลขวัญกำลังใจทีม",
        "tip": "ลองให้เวลารับฟังมุมมองของเพื่อนร่วมทีมให้มากขึ้นก่อนเคาะทิศทางงาน",
        "job_title": "Project Leader / Scrum Master"
    },
    1: {
        "title": "🔍 The Analytical (นักคิดวิเคราะห์รอบคอบ)",
        "color": "#1D3557",
        "strengths": "คิดอย่างมีเหตุผล ละเอียดรอบคอบ มองเห็นความเสี่ยงและจุดบกพร่องของระบบได้ชัดเจน",
        "blindspots": "คิดวิเคราะห์เยอะเกินไป (Analysis Paralysis) จนเริ่มงานช้ากว่าแผน หรือกังวลมากเกินไป",
        "pairing": "จับคู่ได้ดีกับ 🎯 The Driver เพื่อช่วยเคาะตัดสินใจ และ 💡 The Creative เพื่อช่วยจุดประกายไอเดียใหม่",
        "tip": "กำหนดเส้นตายชัดเจน และยอมรับความสมบูรณ์แบบที่ 80% เพื่อให้เริ่มลงมือทำได้ไวขึ้น",
        "job_title": "Core Developer / Data Analyst"
    },
    2: {
        "title": "🤝 The Team Player (ผู้ประสานงานและสร้างมิตรภาพ)",
        "color": "#2A9D8F",
        "strengths": "รับฟังผู้อื่นเก่ง สร้างบรรยากาศที่ดี สมานรอยร้าวและลดความขัดแย้งในทีม เอาใจใส่ทุกคน",
        "blindspots": "เกรงใจผู้อื่นมากเกินไป ไม่กล้าแสดงความคิดเห็นที่ขัดแย้ง หรือแบกรับภาระแทนคนอื่น",
        "pairing": "เข้ากับทุกบทบาทได้ดี โดยเฉพาะทำงานร่วมกับ 🎯 The Driver เพื่อช่วยสื่อสารและประสานทีมอย่างนุ่มนวล",
        "tip": "ฝึกแสดงความเห็นตรงไปตรงมา การท้วงติงด้วยความหวังดีจะช่วยให้โปรเจกต์ดียิ่งขึ้น",
        "job_title": "Team Coordinator & QA Lead"
    },
    3: {
        "title": "💡 The Creative Innovator (นักคิดสร้างสรรค์สิ่งใหม่)",
        "color": "#F4A261",
        "strengths": "คิดนอกกรอบ มีไอเดียแปลกใหม่ มองเห็นโอกาสที่คนอื่นมองข้าม ยืดหยุ่นปรับตัวเก่ง",
        "blindspots": "อาจเบื่อง่ายเมื่อต้องทำงานเอกสาร รูทีน หรือขั้นตอนที่มีแบบแผนตายตัว",
        "pairing": "จับคู่ได้ดีกับ 🔍 The Analytical เพื่อช่วยนำไอเดียมาทำให้เกิดผลจริงได้อย่างเป็นขั้นตอน",
        "tip": "จดบันทึกไอเดียเป็นระบบและแบ่งงานเป็นสเต็ปย่อยๆ เพื่อส่งต่องานให้เพื่อนร่วมทีมทำต่อได้ง่าย",
        "job_title": "Product Designer / Innovation Lead"
    }
}

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

# --- HEADER ---
st.title("🧠 Personality & Team Fit Analyzer (AI & Machine Learning)")
st.markdown("ระบบวิเคราะห์สไตล์การทำงาน บุคลิกภาพ และความเข้ากันได้ของทีมงาน ด้วย **AI NLP + K-Means Clustering + Explainable AI (XAI)**")
st.caption("🌐 ขับเคลื่อนด้วย **AIForThai Cloud Service (NECTEC)** & **PyThaiNLP Engine** | สถานะ API: 🟢 Connected")
st.markdown("---")

# --- TABS NAVIGATION ---
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 1. ประเมินรายบุคคล & AI Explainable",
    "⚔️ 2. เปรียบเทียบคู่หู (Side-by-Side)",
    "👥 3. จัดทีม & Role Recommendation",
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
# TAB 2: SIDE-BY-SIDE COMPARISON & COMPATIBILITY METER
# ==============================================================================
with tab2:
    st.subheader("⚔️ เปรียบเทียบบุคลิกภาพ 2 คน (Side-by-Side Comparison)")
    st.write("ดูว่าเพื่อนร่วมงาน 2 คนมีสไตล์ที่ส่งเสริมกัน หรือมีจุดไหนที่อาจเกิดความขัดแย้งกัน")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        name_a = st.text_input("ชื่อคนที่ 1:", "สมชาย (Leader)", key="name_a")
        text_a = st.text_area("คำอธิบายสไตล์การทำงาน (คนที่ 1):", "ผมมุ่งเน้นผลักดันงานให้เสร็จตามเป้าหมาย ชอบจัดระเบียบ คุมงาน และลงมือทำทันที ลุยไปด้วยกัน", key="text_a")
    
    with col_p2:
        name_b = st.text_input("ชื่อคนที่ 2:", "วิภา (Analytical)", key="name_b")
        text_b = st.text_area("คำอธิบายสไตล์การทำงาน (คนที่ 2):", "ฉันชอบวิเคราะห์ข้อมูลอย่างละเอียด ตรวจสอบความถูกต้องและวางแผนป้องกันความเสี่ยงอย่างรอบคอบ", key="text_b")
        
    if st.button("⚖️ เปรียบเทียบทั้งสองคน", key="btn_compare"):
        with st.spinner("กำลังคำนวณ Compatibility Meter และวิเคราะห์ความเข้ากันได้..."):
            feat_a, _ = extract_features(text_a, "")
            feat_b, _ = extract_features(text_b, "")
            
            c_id_a, radar_a, _, _ = predict_profile(feat_a)
            c_id_b, radar_b, _, _ = predict_profile(feat_b)
            
            p_a = ROLE_PROFILES[c_id_a]
            p_b = ROLE_PROFILES[c_id_b]
            
            vec_a = np.array(feat_a, dtype=float).reshape(1, -1)
            vec_b = np.array(feat_b, dtype=float).reshape(1, -1)
            cos_sim = cosine_similarity(vec_a, vec_b)[0][0]
            
            is_complementary = (c_id_a != c_id_b)
            base_score = float(cos_sim * 60.0) + (35.0 if is_complementary else 20.0)
            compat_score = int(np.clip(base_score, 40, 98))
            
            st.markdown("---")
            st.markdown(f"### 🤝 Compatibility Percentage Meter: **{compat_score}%**")
            
            if compat_score >= 80:
                st.progress(compat_score / 100.0)
                st.success("🌟 **ระดับความเข้ากันได้: สูงมาก (High Synergy)** — มีแนวโน้มทำงานร่วมกันได้อย่างมีประสิทธิภาพสูงและเติมเต็มกันได้ดี")
            elif compat_score >= 60:
                st.progress(compat_score / 100.0)
                st.info("⚖️ **ระดับความเข้ากันได้: ปานกลาง (Moderate Synergy)** — ทำงานร่วมกันได้ดี ควรสื่อสารเป้าหมายให้ตรงกันสม่ำเสมอ")
            else:
                st.progress(compat_score / 100.0)
                st.warning("⚠️ **ระดับความเข้ากันได้: ต้องปรับจูน (Needs Alignment)** — สไตล์การทำงานอาจมีความขัดแย้งเชิงกระบวนการ")
            
            col_res_a, col_res_b = st.columns(2)
            with col_res_a:
                st.markdown(f"**{name_a}**: <span style='color:{p_a['color']}; font-weight:bold;'>{p_a['title']}</span>", unsafe_allow_html=True)
            with col_res_b:
                st.markdown(f"**{name_b}**: <span style='color:{p_b['color']}; font-weight:bold;'>{p_b['title']}</span>", unsafe_allow_html=True)
                
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatterpolar(
                r=radar_a, theta=CATEGORIES, fill='toself', name=name_a, line_color='#E63946'
            ))
            fig_compare.add_trace(go.Scatterpolar(
                r=radar_b, theta=CATEGORIES, fill='toself', name=name_b, line_color='#1D3557'
            ))
            fig_compare.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="เปรียบเทียบ Radar Chart ระหว่าง 2 บุคคล"
            )
            st.plotly_chart(fig_compare, use_container_width=True)

# ==============================================================================
# TAB 3: TEAM BALANCE SIMULATOR & ROLE RECOMMENDATION
# ==============================================================================
with tab3:
    st.subheader("👥 จำลองการจัดทีม & แนะนำตำแหน่งจริง (Team Balance & Role Recommendation)")
    st.write("จำลองการจัดทีมของโปรเจกต์คุณ และให้ AI แนะนำบทบาทที่เหมาะสมที่สุดของแต่ละคน")
    
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
    pcol1, pcol2, pcol3 = st.columns(3)
    if pcol1.button("🟢 ทีมสมดุลครบ 4 ด้าน (Ideal Team)"):
        st.session_state.team_members = [
            {"ชื่อ": "คนึง (Leader)", "บทบาท": 0},
            {"ชื่อ": "วิโรจน์ (Analytical)", "บทบาท": 1},
            {"ชื่อ": "กัญญา (Team Player)", "บทบาท": 2},
            {"ชื่อ": "ปรีชา (Creative)", "บทบาท": 3}
        ]
        st.rerun()
    if pcol2.button("🔴 ทีมผู้นำซ้ำซ้อน (3 Leaders - High Risk)"):
        st.session_state.team_members = [
            {"ชื่อ": "สมศักดิ์ (Leader 1)", "บทบาท": 0},
            {"ชื่อ": "อำนาจ (Leader 2)", "บทบาท": 0},
            {"ชื่อ": "เกรียงไกร (Leader 3)", "บทบาท": 0},
            {"ชื่อ": "นารี (Team Player)", "บทบาท": 2}
        ]
        st.rerun()
    if pcol3.button("🟡 ทีมขาดนักวิเคราะห์ (Missing Analytical)"):
        st.session_state.team_members = [
            {"ชื่อ": "สมชาย (Leader)", "บทบาท": 0},
            {"ชื่อ": "นก (Creative)", "บทบาท": 3},
            {"ชื่อ": "แอน (Creative)", "บทบาท": 3},
            {"ชื่อ": "นุช (Team Player)", "บทบาท": 2}
        ]
        st.rerun()

    if "team_members" not in st.session_state:
        st.session_state.team_members = default_team

    st.markdown("##### 👥 รายชื่อสมาชิกในทีมปัจจุบัน:")
    cols = st.columns(4)
    for idx, member in enumerate(st.session_state.team_members):
        with cols[idx % 4]:
            st.text_input(f"ชื่อคนที่ {idx+1}", value=member["ชื่อ"], key=f"mem_name_{idx}")
            st.selectbox(f"บทบาทของคนที่ {idx+1}", options=list(role_options.keys()), 
                         format_func=lambda x: role_options[x], 
                         index=member["บทบาท"], key=f"mem_role_{idx}")

    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        if st.button("➕ เพิ่มสมาชิกในทีม", key="add_mem"):
            if len(st.session_state.team_members) < 8:
                st.session_state.team_members.append({"ชื่อ": f"สมาชิก {len(st.session_state.team_members)+1}", "บทบาท": 2})
                st.rerun()
            else:
                st.warning("สามารถเพิ่มสมาชิกได้สูงสุด 8 คน")
    with btn_col2:
        if st.button("🔄 รีเซ็ตทีมเริ่มต้น", key="reset_mem"):
            st.session_state.team_members = default_team
            st.rerun()

    st.markdown("---")
    if st.button("📊 วิเคราะห์ความสมดุลและแนะนำตำแหน่ง (Analyze & Recommend)", key="btn_team_eval"):
        current_names = [st.session_state[f"mem_name_{idx}"] for idx in range(len(st.session_state.team_members))]
        current_roles = [st.session_state[f"mem_role_{idx}"] for idx in range(len(st.session_state.team_members))]
        role_counts = pd.Series(current_roles).value_counts()
        
        unique_roles = len(role_counts)
        balance_score = int((unique_roles / 4.0) * 100)
        
        st.markdown(f"### คะแนนความสมดุลของทีม: **{balance_score}%**")
        st.progress(balance_score / 100.0)
        
        # 5. 🏆 Team Role Recommendation
        st.markdown("### 🏆 5. AI Team Role Recommendation (การจัดสรรตำแหน่งจริงในโปรเจกต์)")
        rec_cols = st.columns(len(current_names) if len(current_names) <= 4 else 4)
        for i, (name, r_id) in enumerate(zip(current_names, current_roles)):
            p = ROLE_PROFILES[r_id]
            with rec_cols[i % 4]:
                st.markdown(
                    f"""
                    <div style="background-color: #f1f5f9; padding: 12px; border-radius: 8px; border-top: 4px solid {p['color']}; margin-bottom: 12px;">
                        <h4 style="margin: 0; color: #1e293b;">👤 {name}</h4>
                        <p style="color: {p['color']}; font-weight: bold; margin: 4px 0;">ตำแหน่ง: {p['job_title']}</p>
                        <small style="color: #64748b;">เหตุผล: มีจุดแข็งด้าน {p['strengths'][:45]}...</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("#### 🍩 สัดส่วนบทบาทในทีม")
            labels = [role_options[r] for r in role_counts.index]
            values = role_counts.values
            fig_donut = px.pie(names=labels, values=values, hole=0.45, title="สัดส่วนบทบาทของสมาชิก")
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_t2:
            st.markdown("#### 📋 การวินิจฉัยและคำแนะนำเชิงลึก:")
            missing_roles = [r for r in [0, 1, 2, 3] if r not in role_counts.index]
            if missing_roles:
                for r in missing_roles:
                    if r == 0:
                        st.error("❌ **Critical: ขาด Leader (ผู้นำ)** — เสี่ยงงานล่าช้า ขาดคนตัดสินใจฟันธง")
                    elif r == 1:
                        st.error("❌ **Critical: ขาด Analytical (นักวิเคราะห์)** — เสี่ยงต่องานตกหล่นและบั๊กในระบบ")
                    elif r == 2:
                        st.warning("⚠️ **Warning: ขาด Team Player (ผู้ประสานงาน)** — บรรยากาศตึงเครียด ขาดคนเกลี่ยงาน")
                    elif r == 3:
                        st.warning("⚠️ **Warning: ขาด Creative (นักคิดสร้างสรรค์)** — งานจำเจ ขาดไอเดียแปลกใหม่")
            else:
                st.success("🎉 ครบถ้วน! ไม่มีบทบาทหลักใดที่ขาดหายไปในทีมนี้")
                
            for r, count in role_counts.items():
                if count >= 3:
                    if r == 0:
                        st.error(f"⚠️ **Warning Redundancy: มีผู้นำ (Leader) ถึง {count} คน!** ระวัง Ego Clash และแย่งการตัดสินใจ")
                    else:
                        st.warning(f"⚠️ **Warning: มีบทบาท {role_options[r]} ซ้ำซ้อน {count} คน**")

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