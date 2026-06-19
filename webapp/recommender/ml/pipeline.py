"""
Fungsi & konstanta inti dari notebook 'Algoritma_FINAL_Hybrid_LearningToRank.ipynb'.

Modul ini dipakai bersama oleh:
  - prepare.py  (membangkitkan artifact: SBERT, hybrid, LambdaMART, evaluasi)
  - engine.py   (melayani rekomendasi di web)

Semua logika dipertahankan sama persis dengan notebook agar hasil konsisten.
"""
import re

import numpy as np
import pandas as pd

# ============================================================
# Hyperparameter (sel "0. Import & Konfigurasi" pada notebook)
# ============================================================
SEED = 42

TFIDF_TITLE, TFIDF_SKILLS, TFIDF_DESC = 2000, 2000, 1000
W_TITLE, W_SKILLS, W_DESC = 0.50, 0.25, 0.25

SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SBERT_BATCH = 64

ALPHA, BETA = 0.6, 0.4

POOL = 200
K_LIST = [5, 10, 20, 30]
TEST_FRAC = 0.30

MAX_EXP = 10.0

# Bobot relevansi holistik (sel 7)
W_CAT, W_SK, W_EX, W_ED, W_LO = 0.40, 0.25, 0.15, 0.10, 0.10

# Nama 10 fitur untuk LambdaMART (urutan harus tetap)
FEAT = ["sim_hybrid", "sim_tfidf", "sim_sbert", "loc_match",
        "edu_fit", "edu_av", "exp_fit", "exp_av", "sal_fit", "sal_av"]


# ============================================================
# 1. Praproses teks
# ============================================================
def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def combine_job_text(row):
    return clean_text(" ".join([str(row.get("job_title", "") or ""),
                                str(row.get("skills", "") or ""),
                                str(row.get("job_description", "") or "")]))


# ============================================================
# 2. Fitur struktural
# ============================================================
def experience_years(v):
    t = clean_text(v)
    if not t:
        return np.nan
    if "fresh" in t:
        return 0.0
    n = re.findall(r"\d+(?:[.,]\d+)?", t)
    return min(float(n[0].replace(",", ".")), MAX_EXP) if n else np.nan


def education_value(v):
    t = str(v).upper() if not pd.isna(v) else ""
    if not t or "TIDAK TERSPESIFIKASI" in t:
        return np.nan
    m = {"SMA": 0.0, "SMU": 0.0, "SMK": 0.0, "STM": 0.0, "D3": 0.25, "D4": 0.40,
         "S1": 0.50, "SARJANA": 0.50, "S2": 0.75, "MAGISTER": 0.75, "S3": 1.0, "DOKTOR": 1.0}
    f = [s for l, s in m.items() if l in t]
    return min(f) if f else np.nan


def location_value(v):
    t = clean_text(v)
    a = {"jakarta raya": "jakarta", "dki jakarta": "jakarta"}
    return a.get(t, t)


# ============================================================
# 3. Kategori judul & himpunan skill (untuk label relevansi)
# ============================================================
TITLE_CATEGORIES = {
    "Data Analytics": ["data analyst", "business intelligence", "bi analyst", "reporting analyst", "analytics"],
    "Data Science and AI": ["data scientist", "machine learning", "ml engineer", "artificial intelligence", "ai engineer"],
    "Data Engineering": ["data engineer", "etl developer", "database engineer", "database administrator", "big data"],
    "Software Development": ["software", "developer", "programmer", "backend", "front end", "frontend", "full stack", "mobile developer", "web developer"],
    "IT Infrastructure and Security": ["network", "system administrator", "infrastructure", "cloud", "cyber security", "cybersecurity", "information security", "devops"],
    "UI UX and Product": ["ui ux", "ux designer", "ui designer", "product designer", "product manager", "product owner"],
    "Sales and Business Development": ["sales", "account executive", "business development", "relationship manager", "telesales"],
    "Finance and Accounting": ["finance", "accounting", "accountant", "tax", "audit", "treasury", "financial analyst"],
    "Human Resources": ["human resource", "human capital", "recruiter", "talent acquisition", "people development", "payroll"],
    "Marketing and Communication": ["marketing", "digital marketing", "brand", "social media", "content writer", "public relation", "communication"],
    "Administration and Operations": ["administration", "administrative", "operator", "operations", "office staff", "customer service", "procurement", "warehouse"],
    "Engineering": ["engineer", "engineering", "mechanical", "electrical", "civil", "quality assurance", "quality control", "technician"],
}


def title_category(t):
    t = clean_text(t)
    for cat, kws in TITLE_CATEGORIES.items():
        if any(k in t for k in kws):
            return cat
    return "Other"


def skill_token_set(v):
    t = clean_text(v)
    stop = {"and", "or", "with", "using", "skill", "skills", "ability",
            "knowledge", "experience", "good", "strong", "basic"}
    return {x for x in t.split() if len(x) > 1 and x not in stop}


# ============================================================
# 7. Relevansi holistik
# ============================================================
def jaccard(a, b):
    if not a or not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def vec_fit(arr, qi, pool):
    """Kecocokan fitur numerik (1 - |selisih|), 0 jika kandidat NaN."""
    a = arr[qi]
    b = arr[pool]
    if np.isnan(a):
        return np.zeros(len(pool), dtype=np.float32)
    return np.where(~np.isnan(b), 1.0 - np.abs(b - a), 0.0).astype(np.float32)


def vec_fit_av(arr, qi, pool):
    """Versi yang juga mengembalikan 'availability mask' (1 jika kandidat punya nilai)."""
    a = arr[qi]
    b = arr[pool]
    if np.isnan(a):
        return (np.zeros(len(pool), dtype=np.float32),
                np.zeros(len(pool), dtype=np.float32))
    return (np.where(~np.isnan(b), 1.0 - np.abs(b - a), 0.0).astype(np.float32),
            (~np.isnan(b)).astype(np.float32))


def to_graded(rel):
    return np.digitize(rel, np.array([0.15, 0.30, 0.45, 0.60]))


# ============================================================
# 10. Metrik evaluasi
# ============================================================
def dcg(g, k):
    disc = 1 / np.log2(np.arange(2, k + 2))
    return float(np.sum(g[:k] * disc))


def ndcg_graded(order, graded, k):
    g = 2.0 ** graded[order[:k]] - 1
    ideal = 2.0 ** np.sort(graded)[::-1][:k] - 1
    iz = dcg(ideal, k)
    return dcg(g, k) / iz if iz else 0.0


def precision_at(order, high, k):
    return float(high[order[:k]].mean())


def ap_at(order, high, k):
    rel = high[order[:k]]
    if rel.sum() == 0:
        return 0.0
    cum = np.cumsum(rel)
    pos = cum / np.arange(1, k + 1)
    return float((pos * rel).sum() / min(int(high.sum()), k))
