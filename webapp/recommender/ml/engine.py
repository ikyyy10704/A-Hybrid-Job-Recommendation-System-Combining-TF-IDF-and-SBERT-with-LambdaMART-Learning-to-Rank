"""
Engine serving rekomendasi.

Memuat artifact (TF-IDF, SBERT, LambdaMART) satu kali ke memori, lalu melayani
rekomendasi untuk profil pencari kerja baru. Logika sama dengan fungsi
``recommend_for`` pada sel 13 notebook, namun jarak Euclidean dihitung lewat
dekomposisi (tanpa materialisasi matriks hybrid dense).
"""
import threading

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from sklearn.preprocessing import normalize

from . import pipeline as P

_engine = None
_lock = threading.Lock()


class RecommenderEngine:
    REQUIRED = ["vectorizer_title.joblib", "vectorizer_skills.joblib", "vectorizer_desc.joblib",
                "tfidf_norm.joblib", "sbert_norm.joblib", "ranker.joblib", "meta.joblib"]

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._loaded = False
        self.eval_results = None
        self._loc_dist = None

    # -- ketersediaan artifact -------------------------------------------------
    def missing_artifacts(self):
        return [f for f in self.REQUIRED if not (self.model_dir / f).exists()]

    def is_ready(self):
        return len(self.missing_artifacts()) == 0

    # -- pemuatan (lazy) -------------------------------------------------------
    def load(self):
        if self._loaded:
            return
        md = self.model_dir
        self.vec_title = joblib.load(md / "vectorizer_title.joblib")
        self.vec_skills = joblib.load(md / "vectorizer_skills.joblib")
        self.vec_desc = joblib.load(md / "vectorizer_desc.joblib")
        self.tfidf_norm = joblib.load(md / "tfidf_norm.joblib")
        self.sbert_norm = joblib.load(md / "sbert_norm.joblib")
        self.ranker = joblib.load(md / "ranker.joblib")
        meta = joblib.load(md / "meta.joblib")
        self.smin, self.smax = meta["smin"], meta["smax"]
        if (md / "eval_results.joblib").exists():
            self.eval_results = joblib.load(md / "eval_results.joblib")

        # data.csv -> dataframe untuk tampilan + fitur struktural
        df = pd.read_csv(md / "data.csv", encoding="utf-8", on_bad_lines="skip")
        for c in ["job_title", "skills", "job_description", "location", "education_level",
                  "experience_level", "salary", "company_industry", "career_level"]:
            if c not in df.columns:
                df[c] = np.nan
        text_cols = ["job_title", "skills", "job_description", "location",
                     "education_level", "experience_level", "company_industry", "career_level"]
        df[text_cols] = df[text_cols].fillna("")
        # Urutan sama dengan prepare.py / notebook: dedup -> hybrid_text -> filter.
        df = df.drop_duplicates(subset=["job_title", "company_industry", "location"],
                                keep="first").reset_index(drop=True)
        df["hybrid_text"] = df.apply(P.combine_job_text, axis=1)
        df = df[df["hybrid_text"].str.len() > 0].reset_index(drop=True)

        assert len(df) == self.sbert_norm.shape[0] == self.tfidf_norm.shape[0], \
            "Jumlah baris data.csv tidak cocok dengan artifact. Jalankan ulang prepare_artifacts."

        df["title_category"] = df["job_title"].map(P.title_category)
        self.df = df

        # array struktural (sama seperti notebook)
        self.exp_arr = (df["experience_level"].map(P.experience_years) / P.MAX_EXP).to_numpy()
        self.edu_arr = df["education_level"].map(P.education_value).to_numpy()
        self.loc_arr = df["location"].map(P.location_value).to_numpy()
        sal_numeric = pd.to_numeric(df["salary"], errors="coerce")
        self._sal_arr = ((sal_numeric - self.smin) / max(self.smax - self.smin, 1)).clip(0, 1).to_numpy()

        # Hsq (dekomposisi); baris TF-IDF & SBERT ter-normalisasi L2
        A2, B2 = P.ALPHA ** 2, P.BETA ** 2
        tf_sq = np.asarray(self.tfidf_norm.multiply(self.tfidf_norm).sum(axis=1)).ravel().astype("float32")
        sb_sq = np.einsum("ij,ij->i", self.sbert_norm, self.sbert_norm).astype("float32")
        self.Hsq = (A2 * tf_sq + B2 * sb_sq).astype("float32")
        self._A2, self._B2 = A2, B2

        # SBERT model untuk encode query (lazy juga, agar startup ringan)
        self._sbert_model = None
        self._loaded = True

    @property
    def sbert_model(self):
        if self._sbert_model is None:
            from sentence_transformers import SentenceTransformer
            self._sbert_model = SentenceTransformer(P.SBERT_MODEL)
        return self._sbert_model

    # -- rekomendasi -----------------------------------------------------------
    def recommend(self, title, skills, description="", experience="", education="",
                  location="", salary=np.nan, top_k=10):
        self.load()
        A2, B2 = self._A2, self._B2

        # 1) representasi teks query
        t = self.vec_title.transform([P.clean_text(title)]).multiply(P.W_TITLE)
        s = self.vec_skills.transform([P.clean_text(skills)]).multiply(P.W_SKILLS)
        d = self.vec_desc.transform([P.clean_text(description)]).multiply(P.W_DESC)
        tfidf_u = normalize(sparse_hstack([t, s, d]).tocsr(), norm="l2").astype("float32")  # (1, D) sparse
        sbert_txt = P.clean_text(" ".join([title, skills, description]))
        sbert_u = normalize(self.sbert_model.encode([sbert_txt], convert_to_numpy=True),
                            norm="l2").astype("float32").ravel()

        # 2) retrieval Top-POOL (jarak Euclidean lewat dekomposisi)
        g = (A2 * np.asarray((self.tfidf_norm @ tfidf_u.T).todense()).ravel()
             + B2 * (self.sbert_norm @ sbert_u)).astype("float32")
        u_sq = float(A2 * (tfidf_u.multiply(tfidf_u)).sum() + B2 * float(sbert_u @ sbert_u))
        d2 = self.Hsq + u_sq - 2.0 * g
        pool = np.argpartition(d2, P.POOL)[:P.POOL]
        pool = pool[np.argsort(d2[pool])]

        sim_hy = (1.0 / (1.0 + np.sqrt(np.clip(d2[pool], 0, None)))).astype(np.float32)
        sim_tf = np.asarray((self.tfidf_norm[pool] @ tfidf_u.T).todense()).ravel().astype(np.float32)
        sim_sb = (self.sbert_norm[pool] @ sbert_u).astype(np.float32)

        # 3) fitur struktural query
        qe = P.experience_years(experience)
        q_exp = qe / P.MAX_EXP if not np.isnan(qe) else np.nan
        q_edu = P.education_value(education)
        q_loc = P.location_value(location)
        q_sal = ((salary - self.smin) / max(self.smax - self.smin, 1)) if (salary == salary and salary > 0) else np.nan

        def fit_scalar(qval, arr):
            if np.isnan(qval):
                return np.zeros(len(pool), np.float32), np.zeros(len(pool), np.float32)
            b = arr[pool]
            return (np.where(~np.isnan(b), 1.0 - np.abs(b - qval), 0.0).astype(np.float32),
                    (~np.isnan(b)).astype(np.float32))

        lo = np.array([1.0 if (q_loc and self.loc_arr[int(j)] and q_loc == self.loc_arr[int(j)]) else 0.0
                       for j in pool], np.float32)
        ef, ea = fit_scalar(q_edu, self.edu_arr)
        xf, xa = fit_scalar(q_exp, self.exp_arr)
        sf, sa = fit_scalar(q_sal, self._sal_arr)

        X = np.stack([sim_hy, sim_tf, sim_sb, lo, ef, ea, xf, xa, sf, sa], axis=1).astype(np.float32)

        # 4) perangkingan LambdaMART
        scores = self.ranker.predict(X)
        order = np.argsort(-scores)
        top = pool[order[:top_k]]
        sorted_scores = np.sort(scores)[::-1][:top_k]

        results = []
        for rank, (j, sc) in enumerate(zip(top, sorted_scores), start=1):
            row = self.df.iloc[int(j)]
            results.append({
                "idx": int(j),
                "rank": rank,
                "job_title": str(row["job_title"]),
                "location": str(row["location"]) or "-",
                "category": str(row["title_category"]),
                "skills": str(row.get("skills", "") or "")[:160],
                "company_industry": str(row.get("company_industry", "") or "-"),
                "skor_ltr": round(float(sc), 3),
            })
        return results

    # -- sebaran lokasi untuk peta ---------------------------------------------
    def location_distribution(self):
        """Agregasi jumlah lowongan per titik koordinat (untuk peta Indonesia)."""
        if self._loc_dist is not None:
            return self._loc_dist
        self.load()
        from collections import defaultdict
        from . import geo
        counts, coords = defaultdict(int), {}
        total = unplotted = 0
        vc = self.df["location"].fillna("").map(P.location_value).value_counts()
        for loc, cnt in vc.items():
            cnt = int(cnt)
            total += cnt
            hit = geo.resolve(loc)
            if hit is None:
                unplotted += cnt
                continue
            name, lat, lng = hit
            counts[name] += cnt
            coords[name] = (lat, lng)
        points = [{"name": n, "lat": coords[n][0], "lng": coords[n][1], "count": c}
                  for n, c in counts.items()]
        points.sort(key=lambda p: -p["count"])
        self._loc_dist = {"points": points, "total": total,
                          "plotted": total - unplotted, "unplotted": unplotted,
                          "n_titik": len(points)}
        return self._loc_dist

    # -- detail satu lowongan --------------------------------------------------
    def get_job(self, idx):
        """Kembalikan detail lengkap satu lowongan berdasarkan indeks baris df."""
        self.load()
        if idx < 0 or idx >= len(self.df):
            return None
        row = self.df.iloc[int(idx)]

        def val(col, dash="-"):
            v = row.get(col, "")
            s = "" if v is None else str(v).strip()
            if not s or s.lower() in ("nan", "none", "tidak terspesifikasi"):
                return dash
            return s

        # gaji
        try:
            sal = float(row.get("salary"))
        except (TypeError, ValueError):
            sal = float("nan")
        if sal == sal and sal > 0:
            cur = val("salary_currency", "")
            salary = (f"{cur} " if cur and cur != "-" else "") + f"{sal:,.0f}".replace(",", ".")
        else:
            salary = "Tidak disebutkan"

        skills_raw = val("skills", "")
        skills = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw and skills_raw != "-" else []
        benefits_raw = val("job_benefits", "")
        benefits = [b.strip() for b in benefits_raw.split(",") if b.strip()] if benefits_raw and benefits_raw != "-" else []

        return {
            "idx": int(idx),
            "job_title": val("job_title"),
            "location": val("location"),
            "category": str(row.get("title_category", "")),
            "company_industry": val("company_industry"),
            "company_size": val("company_size"),
            "employment_type": val("employment_type"),
            "career_level": val("career_level"),
            "experience_level": val("experience_level"),
            "education_level": val("education_level"),
            "job_function": val("job_function"),
            "salary": salary,
            "skills": skills,
            "benefits": benefits,
            "description": val("job_description", ""),
        }


def get_engine():
    """Singleton engine; aman dipanggil dari banyak request."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                from django.conf import settings
                _engine = RecommenderEngine(settings.MODEL_DIR)
    return _engine
