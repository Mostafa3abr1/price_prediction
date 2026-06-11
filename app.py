#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sal7ly — Price Prediction App  (MiniLM + Scikit-Learn models)
inDrive-style: Customer writes problem → AI predicts category + price → adjusts → submits
"""

import json, math, re
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="صَلِّحلي",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');

html, body, [class*="css"] { font-family:'Cairo',sans-serif !important; }
.stApp { background: linear-gradient(160deg,#090912 0%,#0f0f20 60%,#0a1520 100%) !important; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1rem !important; max-width:700px; }

.hero { text-align:center; padding:1.8rem 1rem 1.2rem; }
.hero h1 {
    font-size:2.6rem; font-weight:900; margin:.3rem 0 .4rem;
    background:linear-gradient(135deg,#6C63FF,#EC4899,#10B981);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p { color:#475569; font-size:1rem; }

.step-label {
    color:#6C63FF; font-size:.8rem; font-weight:700;
    letter-spacing:.1em; text-transform:uppercase; margin-bottom:.3rem;
}

.price-card {
    background:linear-gradient(135deg,rgba(16,185,129,.12),rgba(108,99,255,.08));
    border:1.5px solid rgba(16,185,129,.4);
    border-radius:22px; padding:2rem 1.5rem 1.5rem;
    text-align:center; margin:1rem 0;
    animation: glow 2.5s ease-in-out infinite;
}
@keyframes glow {
    0%,100%{box-shadow:0 0 20px rgba(16,185,129,.1);}
    50%    {box-shadow:0 0 45px rgba(16,185,129,.25);}
}
.price-egp  { font-size:4.5rem; font-weight:900; color:#10B981; line-height:1; }
.price-unit { color:#475569; font-size:.9rem; margin-top:.3rem; }
.price-rng  { color:#818CF8; font-size:.95rem; margin-top:.5rem; font-weight:600; }
.det-row    { display:flex; justify-content:center; gap:.7rem; margin-top:.9rem; flex-wrap:wrap; }
.det-pill   { border-radius:30px; padding:.25rem .9rem; font-size:.82rem; font-weight:700; }

.acc-badge {
    display:inline-block; background:rgba(16,185,129,.15);
    color:#10B981; border:1px solid rgba(16,185,129,.3);
    border-radius:20px; padding:.15rem .7rem; font-size:.75rem;
    font-weight:700; margin-top:.4rem;
}

.stTextArea textarea {
    background:rgba(255,255,255,.05) !important;
    border:1.5px solid rgba(108,99,255,.3) !important;
    border-radius:14px !important; color:#E2E8F0 !important;
    font-family:'Cairo',sans-serif !important; font-size:1.05rem !important;
    direction:rtl !important;
}
.stTextArea textarea:focus {
    border-color:#6C63FF !important;
    box-shadow:0 0 0 3px rgba(108,99,255,.2) !important;
}
.stButton>button {
    background:linear-gradient(135deg,#6C63FF,#8B85FF) !important;
    color:#fff !important; border:none !important; border-radius:14px !important;
    font-family:'Cairo',sans-serif !important;
    font-size:1.15rem !important; font-weight:700 !important;
    padding:.8rem !important; width:100%; transition:all .2s !important;
}
.stButton>button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 25px rgba(108,99,255,.45) !important;
}
div[data-testid="metric-container"] {
    background:rgba(255,255,255,.04);
    border:1px solid rgba(255,255,255,.08);
    border-radius:12px; padding:.7rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
MODELS_DIR  = ROOT_DIR / "models"

CAT_META = {
    "سباكة":  {"icon":"🚿","color":"#38BDF8"},
    "كهرباء": {"icon":"⚡","color":"#FBBF24"},
    "نجارة":  {"icon":"🪚","color":"#A78BFA"},
    "نقاشة":  {"icon":"🎨","color":"#F472B6"},
}
DIFF_META = {
    "easy":  {"ar":"سهل",   "color":"#10B981"},
    "medium":{"ar":"متوسط", "color":"#F59E0B"},
    "hard":  {"ar":"صعب",   "color":"#EF4444"},
}

# ─────────────────────────────────────────────────────────────
# CLEAN ARABIC TEXT
# ─────────────────────────────────────────────────────────────
DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
TATWEEL_RE = re.compile(r"\u0640+")

def clean_arabic_text(text):
    text = str(text)
    text = DIACRITICS_RE.sub("", text)
    text = TATWEEL_RE.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub(r"[^ء-يa-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models():
    # 1. Load MiniLM
    minilm_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 2. Load classifiers
    category_model = joblib.load(MODELS_DIR / "best_category_minilm_linear_svc.joblib")
    difficulty_model = joblib.load(MODELS_DIR / "best_difficulty_minilm_linear_svc.joblib")

    # 3. Load price models
    price_reg_model = joblib.load(MODELS_DIR / "best_price_regression_minilm_random_forest.joblib")
    price_band_bundle = joblib.load(MODELS_DIR / "best_enhanced_price_band_classifier_random_forest_enhanced.joblib")

    return minilm_model, category_model, difficulty_model, price_reg_model, price_band_bundle

# ─────────────────────────────────────────────────────────────
# EXAMPLE REQUESTS
# ─────────────────────────────────────────────────────────────
EXAMPLES = [
    "الحنفية في الحمام بتقطر مية طول الوقت عايز حد يصلحها",
    "اللمبات في الأوضة بتوقع كتير ومفيش كهرباء في بعض الأماكن",
    "باب الأوضة الخشب مش بيقفل كويس والمفصلات بتصرصر",
    "الصالة كلها محتاجة دهان جديد الحيطان فيها شقوق",
    "فيه تسريب مية في السقف من الشقة اللي فوق",
    "المضخة بتاعة الخزان وقفت عن الشغل",
    "الدولاب الخشب في الأوضة فيه درج مكسور وباب معوج",
    "عايز ورق حائط في غرفة النوم المتوسطة الحجم",
]

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
with st.spinner("جاري تحميل النماذج..."):
    minilm_model, category_model, difficulty_model, price_reg_model, price_band_bundle = load_all_models()

# Helper for live prediction
def classify_text_live(text):
    cleaned = clean_arabic_text(text)
    X = minilm_model.encode([cleaned], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    cat_pred = str(category_model.predict(X)[0])
    diff_pred = str(difficulty_model.predict(X)[0])
    return cat_pred, diff_pred

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "text_val"    not in st.session_state: st.session_state.text_val    = ""
if "show_result" not in st.session_state: st.session_state.show_result = False
if "pred_min"    not in st.session_state: st.session_state.pred_min    = 0
if "pred_max"    not in st.session_state: st.session_state.pred_max    = 0
if "pred_avg"    not in st.session_state: st.session_state.pred_avg    = 0
if "det_cat"     not in st.session_state: st.session_state.det_cat     = ""
if "det_diff"    not in st.session_state: st.session_state.det_diff    = ""
if "pred_band"   not in st.session_state: st.session_state.pred_band   = ""

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div style="font-size:2.8rem;">🔧</div>
    <h1>صَلِّحلي</h1>
    <p>اكتب مشكلتك وهنديك سعر مناسب فوراً</p>
    <span class="acc-badge">✦ دقة عالية · MiniLM & RF · v5</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="step-label">اكتب وصف مشكلتك</div>', unsafe_allow_html=True)

with st.expander("💡 أمثلة جاهزة", expanded=False):
    for ex in EXAMPLES:
        if st.button(f"◈ {ex}", key=f"ex_{ex[:15]}"):
            st.session_state.text_val    = ex
            st.session_state.show_result = False
            st.rerun()

text = st.text_area(
    "وصف المشكلة",
    value=st.session_state.text_val,
    placeholder="مثال: الحنفية في الحمام بتقطر مية...",
    height=130,
    label_visibility="collapsed",
)

# Live preview
if text.strip():
    try:
        cat_preview, diff_preview = classify_text_live(text)
        cm = CAT_META[cat_preview]
        dm = DIFF_META[diff_preview]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:.7rem;margin:.4rem 0;flex-wrap:wrap;">
            <span style="color:#475569;font-size:.82rem;">الـ AI اكتشف:</span>
            <span style="background:rgba(255,255,255,.06);color:{cm['color']};
                         border:1px solid {cm['color']}44;border-radius:20px;
                         padding:.2rem .9rem;font-size:.85rem;font-weight:700;">
                {cm['icon']} {cat_preview}
            </span>
            <span style="background:rgba(255,255,255,.04);color:{dm['color']};
                         border:1px solid {dm['color']}33;border-radius:20px;
                         padding:.2rem .8rem;font-size:.82rem;font-weight:700;">
                {dm['ar']}
            </span>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        pass

st.markdown("<br>", unsafe_allow_html=True)
calc_btn = st.button("✦ احسب السعر المناسب", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# CALCULATE
# ─────────────────────────────────────────────────────────────
if calc_btn:
    if not text.strip():
        st.warning("⚠️ اكتب وصف مشكلتك الأول!")
    else:
        with st.spinner("جاري التحليل..."):
            cleaned = clean_arabic_text(text)
            X = minilm_model.encode([cleaned], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            
            # Predict category & difficulty
            cat = str(category_model.predict(X)[0])
            diff = str(difficulty_model.predict(X)[0])
            
            # Predict price range
            pred_prices = price_reg_model.predict(X)
            pred_prices = np.asarray(pred_prices).reshape(1, -1)[0]
            
            # Step and price order checks
            pmin = int(round(max(pred_prices[0], 0) / 25) * 25)
            pavg = int(round(max(pred_prices[1], 0) / 25) * 25)
            pmax = int(round(max(pred_prices[2], 0) / 25) * 25)
            
            # Enforce min <= avg <= max
            prices_sorted = sorted([pmin, pavg, pmax])
            pmin, pavg, pmax = prices_sorted[0], prices_sorted[1], prices_sorted[2]
            
            # Ensure UI can move + and -
            if pmax <= pavg:
                pmax = pavg + 25
            if pmin >= pavg:
                pmin = max(0, pavg - 25)
                
            # Predict price band using enhanced classifier
            context_df = pd.DataFrame([{"category": cat, "difficulty": diff}])
            context_df = context_df[["category", "difficulty"]].astype(str)
            
            # Get encoders from bundle
            pb_model = price_band_bundle["model"]
            pb_label_encoder = price_band_bundle["label_encoder"]
            pb_context_encoder = price_band_bundle["context_encoder"]
            
            X_context = pb_context_encoder.transform(context_df)
            X_enhanced = hstack([
                csr_matrix(X),
                X_context
            ]).tocsr()
            
            pb_encoded = pb_model.predict(X_enhanced)
            predicted_band = str(pb_label_encoder.inverse_transform(pb_encoded)[0])
            
        st.session_state.pred_min    = pmin
        st.session_state.pred_max    = pmax
        st.session_state.pred_avg    = pavg
        st.session_state.det_cat     = cat
        st.session_state.det_diff    = diff
        st.session_state.pred_band   = predicted_band
        st.session_state.show_result = True
        st.rerun()

# ─────────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────────
if st.session_state.show_result:
    pmin = st.session_state.pred_min
    pmax = st.session_state.pred_max
    pavg = st.session_state.pred_avg
    cat  = st.session_state.det_cat
    diff = st.session_state.det_diff
    pband = st.session_state.pred_band
    cm   = CAT_META[cat]
    dm   = DIFF_META[diff]
    
    # Map price band to Arabic for user-friendliness
    band_ar = {"low": "منخفض", "medium": "متوسط", "high": "مرتفع", "very_high": "مرتفع جداً"}.get(pband, pband)

    st.markdown("---")
    st.markdown('<div class="step-label">السعر المقترح</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="price-card">
        <div style="color:#475569;font-size:.85rem;margin-bottom:.3rem;">السعر المناسب لخدمتك</div>
        <div class="price-egp">{pavg:,.0f}</div>
        <div class="price-unit">جنيه مصري</div>
        <div class="price-rng">نطاق: {pmin:,.0f} — {pmax:,.0f} جنيه</div>
        <div class="det-row">
            <span class="det-pill" style="background:rgba(255,255,255,.06);
                color:{cm['color']};border:1px solid {cm['color']}44;">
                {cm['icon']} {cat}
            </span>
            <span class="det-pill" style="background:rgba(255,255,255,.04);
                color:{dm['color']};border:1px solid {dm['color']}33;">
                {dm['ar']}
            </span>
            <span class="det-pill" style="background:rgba(255,255,255,.04);
                color:#EC4899;border:1px solid rgba(236,72,153,.3);">
                🏷️ فئة السعر: {band_ar}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Adjust slider ──
    st.markdown('<div class="step-label" style="margin-top:1rem;">عدّل السعر</div>',
                unsafe_allow_html=True)
    st.caption("زيّد أو نقّص حسب رأيك قبل ما ترفع الطلب")

    lo_s = max(25, pmin - pmin // 2)
    hi_s = pmax + pmax // 2
    
    # Round slider boundaries to 25
    lo_s = int(round(lo_s / 25) * 25)
    hi_s = int(round(hi_s / 25) * 25)

    final_price = st.slider(
        "السعر",
        min_value=int(lo_s), max_value=int(hi_s),
        value=int(pavg), step=25,
        format="%d جنيه", label_visibility="collapsed",
    )

    diff_val = final_price - pavg
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("التقدير الأصلي", f"{pavg:,} EGP")
    with c2: st.metric("سعرك النهائي",   f"{final_price:,} EGP",
                       delta=f"{diff_val:+,} EGP" if diff_val != 0 else None)
    with c3: st.metric("النطاق المقترح", f"{pmin:,}–{pmax:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Submit ──
    if st.button(f"✦ رفع الطلب بسعر {final_price:,} جنيه", use_container_width=True):
        st.balloons()
        st.success(f"""
        ✅ **تم رفع طلبك بنجاح!**

        سيتم عرض طلبك على الفنيين المتاحين بسعر **{final_price:,} جنيه**
        وسيتواصل معك الفني في أقرب وقت.
        """)

    st.markdown("""
    <div style="background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.2);
                border-radius:12px;padding:.7rem 1rem;margin-top:.8rem;
                color:#64748B;font-size:.8rem;text-align:center;">
        ⚠️ هذا تقدير استرشادي — السعر الفعلي قد يختلف حسب الفني والموقع والتفاصيل
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;color:#64748B;font-size:.75rem;margin-top:3rem;padding-bottom:1rem;">
    Sal7ly · MiniLM & Random Forest · R²=0.88 · MAE=152 EGP · مشروع التخرج 2026
</div>
""", unsafe_allow_html=True)
