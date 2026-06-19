"""
Pembangkit artifact (sekali jalan) untuk web rekomendasi.

Mereplikasi tahapan berat notebook 'Algoritma_FINAL_Hybrid_LearningToRank.ipynb':
  1. Praproses + fitur struktural
  2. TF-IDF per-field berbobot
  3. Embedding SBERT (768-dim)        <- berat (encode seluruh data)
  4. Relevansi holistik + retrieval 200 kandidat + ekstraksi 10 fitur
  5. Pelatihan LambdaMART              <- berat
  6. Evaluasi (multi-K, ablation, Wilcoxon, feature importance, contoh I/O)

Artifact disimpan ke folder Model/ agar bisa dimuat cepat oleh engine.py (serving).

Jalankan via:  python manage.py prepare_artifacts
"""
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import lightgbm as lgb

from . import pipeline as P


def _log(msg, verbose=True):
    if verbose:
        print(msg, flush=True)


def run(model_dir, verbose=True, use_sbert_cache=True):
    """Bangun semua artifact dan simpan ke ``model_dir`` (pathlib.Path)."""
    import torch

    np.random.seed(P.SEED)
    torch.manual_seed(P.SEED)
    rng = np.random.default_rng(P.SEED)

    model_dir.mkdir(parents=True, exist_ok=True)
    csv_path = model_dir / "data.csv"

    # ------------------------------------------------------------------
    # 1. Muat data + praproses (notebook sel 1)
    # ------------------------------------------------------------------
    _log("=" * 60, verbose)
    _log("[1/6] Memuat & memproses data ...", verbose)
    raw_df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    req = ["job_title", "skills", "job_description", "location", "education_level",
           "experience_level", "salary", "company_industry", "career_level"]
    for c in req:
        if c not in raw_df.columns:
            raw_df[c] = np.nan
    df = raw_df.copy()
    text_columns = ["job_title", "skills", "job_description", "location",
                    "education_level", "experience_level", "company_industry", "career_level"]
    df[text_columns] = df[text_columns].fillna("")
    # Hapus duplikat (notebook sel 1). Pada Model/data.csv ini no-op karena
    # data sudah ter-dedup, tetapi dipertahankan agar setia bila sumber data mentah.
    df = df.drop_duplicates(subset=["job_title", "company_industry", "location"],
                            keep="first").reset_index(drop=True)
    df["hybrid_text"] = df.apply(P.combine_job_text, axis=1)
    df["title_clean"] = df["job_title"].apply(P.clean_text)
    df["skills_clean"] = df["skills"].apply(P.clean_text)
    df["desc_clean"] = df["job_description"].apply(P.clean_text)
    df = df[df["hybrid_text"].str.len() > 0].reset_index(drop=True)
    _log(f"      Data eksperimen: {len(df):,} lowongan", verbose)

    # ------------------------------------------------------------------
    # 2. Fitur struktural (notebook sel 2)
    # ------------------------------------------------------------------
    df["exp_norm"] = df["experience_level"].map(P.experience_years) / P.MAX_EXP
    df["edu_norm"] = df["education_level"].map(P.education_value)
    df["loc_norm"] = df["location"].map(P.location_value)
    df["salary_numeric"] = pd.to_numeric(df["salary"], errors="coerce")
    vs = df.loc[df["salary_numeric"] > 0, "salary_numeric"]
    smin = float(vs.min()) if len(vs) else 0.0
    smax = float(vs.max()) if len(vs) else 1.0
    df["salary_norm"] = ((df["salary_numeric"] - smin) / max(smax - smin, 1)).clip(0, 1)

    exp_arr = df["exp_norm"].to_numpy()
    edu_arr = df["edu_norm"].to_numpy()
    sal_arr = df["salary_norm"].to_numpy()
    loc_arr = df["loc_norm"].to_numpy()
    availability = df[["exp_norm", "edu_norm", "salary_norm"]].notna().mean().mul(100).round(1).to_dict()

    # 3. Kategori & skill (untuk label relevansi)
    df["title_category"] = df["job_title"].map(P.title_category)
    cat_arr = df["title_category"].to_numpy()
    skills = df["skills"].map(P.skill_token_set).tolist()
    category_counts = df["title_category"].value_counts().head(8).to_dict()

    # ------------------------------------------------------------------
    # 4. TF-IDF per-field berbobot (notebook sel 4)
    # ------------------------------------------------------------------
    _log("[2/6] Membangun TF-IDF ...", verbose)
    t0 = time.time()
    vec_title = TfidfVectorizer(max_features=P.TFIDF_TITLE, ngram_range=(1, 3), min_df=2, max_df=0.8, sublinear_tf=True, norm="l2")
    vec_skills = TfidfVectorizer(max_features=P.TFIDF_SKILLS, ngram_range=(1, 3), min_df=2, max_df=0.8, sublinear_tf=True, norm="l2")
    vec_desc = TfidfVectorizer(max_features=P.TFIDF_DESC, ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True, norm="l2")
    m_title = vec_title.fit_transform(df["title_clean"]).multiply(P.W_TITLE)
    m_skills = vec_skills.fit_transform(df["skills_clean"]).multiply(P.W_SKILLS)
    m_desc = vec_desc.fit_transform(df["desc_clean"]).multiply(P.W_DESC)
    # Disimpan SPARSE (hemat memori); baris ter-normalisasi L2 -> dot product = cosine.
    tfidf_norm = normalize(sparse_hstack([m_title, m_skills, m_desc]).tocsr(), norm="l2").astype("float32")
    _log(f"      TF-IDF selesai: {tfidf_norm.shape} dalam {time.time()-t0:.1f}s", verbose)

    # ------------------------------------------------------------------
    # 5. SBERT (notebook sel 5)  -- berat
    # ------------------------------------------------------------------
    sbert_path = model_dir / "sbert_norm.joblib"
    if use_sbert_cache and sbert_path.exists():
        _log("[3/6] Memuat SBERT dari cache ...", verbose)
        sbert_norm = joblib.load(sbert_path)
        if sbert_norm.shape[0] != len(df):
            _log("      Cache tidak cocok jumlah baris -> hitung ulang.", verbose)
            sbert_norm = None
    else:
        sbert_norm = None

    if sbert_norm is None:
        from sentence_transformers import SentenceTransformer
        _log(f"[3/6] Encoding SBERT '{P.SBERT_MODEL}' (bisa lama di CPU) ...", verbose)
        sbert_model = SentenceTransformer(P.SBERT_MODEL)
        t0 = time.time()
        sbert_dense = sbert_model.encode(
            df["hybrid_text"].tolist(), batch_size=P.SBERT_BATCH,
            show_progress_bar=verbose, convert_to_numpy=True).astype("float32")
        sbert_norm = normalize(sbert_dense, norm="l2").astype("float32")
        _log(f"      SBERT selesai dalam {(time.time()-t0)/60:.2f} menit | shape {sbert_norm.shape}", verbose)
        joblib.dump(sbert_norm, sbert_path)

    assert sbert_norm.shape[0] == len(df)

    # ------------------------------------------------------------------
    # 6. Hybrid (dekomposisi) -- tidak perlu materialisasi matriks dense.
    #    V = [ALPHA*tfidf_norm, BETA*sbert_norm]
    #    ||V_i||^2 = ALPHA^2*||tfidf_i||^2 + BETA^2*||sbert_i||^2
    #    V_i . V_j = ALPHA^2*(tfidf_i.tfidf_j) + BETA^2*(sbert_i.sbert_j)
    # ------------------------------------------------------------------
    A2, B2 = P.ALPHA ** 2, P.BETA ** 2
    tf_sq = np.asarray(tfidf_norm.multiply(tfidf_norm).sum(axis=1)).ravel().astype("float32")
    sb_sq = np.einsum("ij,ij->i", sbert_norm, sbert_norm).astype("float32")
    Hsq = (A2 * tf_sq + B2 * sb_sq).astype("float32")

    # ------------------------------------------------------------------
    # 7-8. Relevansi holistik + retrieval + ekstraksi fitur (sel 7-8)
    # ------------------------------------------------------------------
    def holistic_relevance(qi, pool):
        cat = (cat_arr[pool] == cat_arr[qi]).astype(np.float32)
        qs = skills[qi]
        sk = np.array([P.jaccard(qs, skills[int(j)]) for j in pool], dtype=np.float32)
        ex = P.vec_fit(exp_arr, qi, pool)
        ed = P.vec_fit(edu_arr, qi, pool)
        lo = np.array([1.0 if (loc_arr[qi] and loc_arr[int(j)] and loc_arr[qi] == loc_arr[int(j)]) else 0.0
                       for j in pool], dtype=np.float32)
        return P.W_CAT * cat + P.W_SK * sk + P.W_EX * ex + P.W_ED * ed + P.W_LO * lo

    vc = df["title_category"].value_counts()
    elig = df.index[(df["title_category"] != "Other") & (df["title_category"].map(vc) > max(P.K_LIST))].to_numpy()
    rng.shuffle(elig)
    _log("[4/6] Ekstraksi fitur untuk %s query ..." % f"{len(elig):,}", verbose)

    all_pool, all_X, all_graded = [], [], []
    CH = 256
    t0 = time.time()
    for s in range(0, len(elig), CH):
        qc = elig[s:s + CH]
        # G = V @ Vqc^T  (dekomposisi)
        Gtf = (tfidf_norm @ tfidf_norm[qc].T).toarray()
        Gsb = sbert_norm @ sbert_norm[qc].T
        G = (A2 * Gtf + B2 * Gsb).astype(np.float32)
        for k, qi in enumerate(qc):
            col = Hsq + Hsq[qi] - 2.0 * G[:, k]
            col[qi] = np.inf
            cand = np.argpartition(col, P.POOL)[:P.POOL]
            cand = cand[np.argsort(col[cand])]
            sim_hy = (1.0 / (1.0 + np.sqrt(np.clip(col[cand], 0, None)))).astype(np.float32)
            sim_tf = np.asarray((tfidf_norm[cand] @ tfidf_norm[qi].T).todense()).ravel().astype(np.float32)
            sim_sb = (sbert_norm[cand] @ sbert_norm[qi]).astype(np.float32)
            lo = np.array([1.0 if (loc_arr[qi] and loc_arr[int(j)] and loc_arr[qi] == loc_arr[int(j)]) else 0.0
                           for j in cand], dtype=np.float32)
            ef, ea = P.vec_fit_av(edu_arr, qi, cand)
            xf, xa = P.vec_fit_av(exp_arr, qi, cand)
            sf, sa = P.vec_fit_av(sal_arr, qi, cand)
            X = np.stack([sim_hy, sim_tf, sim_sb, lo, ef, ea, xf, xa, sf, sa], axis=1).astype(np.float32)
            all_pool.append(cand)
            all_X.append(X)
            all_graded.append(P.to_graded(holistic_relevance(qi, cand)))
        if (s // CH) % 10 == 0:
            _log(f"      {s+len(qc):,}/{len(elig):,} | {time.time()-t0:.0f}s", verbose)
    _log(f"      Ekstraksi selesai: {time.time()-t0:.0f}s | total query: {len(all_X):,}", verbose)

    # ------------------------------------------------------------------
    # 9. Split + pelatihan LambdaMART (sel 9)  -- berat
    # ------------------------------------------------------------------
    _log("[5/6] Melatih LambdaMART ...", verbose)
    n = len(all_X)
    n_test = int(n * P.TEST_FRAC)
    idx = np.arange(n)
    rng.shuffle(idx)
    test_set = set(idx[:n_test].tolist())
    train_pos = [i for i in range(n) if i not in test_set]
    test_pos = [i for i in range(n) if i in test_set]
    Xtr = np.vstack([all_X[i] for i in train_pos])
    ytr = np.concatenate([all_graded[i] for i in train_pos])
    grp = [len(all_graded[i]) for i in train_pos]
    _log(f"      Train: {len(train_pos):,} query ({len(ytr):,} baris) | Test: {len(test_pos):,} query", verbose)
    t0 = time.time()
    ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=500, learning_rate=0.05,
                            num_leaves=63, min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
                            random_state=P.SEED, verbose=-1)
    ranker.fit(Xtr, ytr, group=grp)
    _log(f"      LambdaMART dilatih: {time.time()-t0:.0f}s", verbose)

    # ------------------------------------------------------------------
    # 10. Evaluasi multi-K (sel 10)
    # ------------------------------------------------------------------
    _log("[6/6] Evaluasi & menyimpan artifact ...", verbose)
    ndcg_at = {k: [] for k in P.K_LIST}
    prec_at = {k: [] for k in P.K_LIST}
    map_at = {k: [] for k in P.K_LIST}
    for i in test_pos:
        X = all_X[i]
        graded = all_graded[i]
        order = np.argsort(-ranker.predict(X))
        high = (graded >= 2).astype(int)
        for k in P.K_LIST:
            ndcg_at[k].append(P.ndcg_graded(order, graded, k))
            prec_at[k].append(P.precision_at(order, high, k))
            map_at[k].append(P.ap_at(order, high, k))
    metrics = {k: {"ndcg": round(float(np.mean(ndcg_at[k])) * 100, 2),
                   "precision": round(float(np.mean(prec_at[k])) * 100, 2),
                   "map": round(float(np.mean(map_at[k])) * 100, 2)} for k in P.K_LIST}

    # 10b. Ablation + Wilcoxon
    methods_abl = {
        "TF-IDF saja":        lambda X: X[:, 1],
        "SBERT saja":         lambda X: X[:, 2],
        "Hybrid (tanpa LTR)": lambda X: X[:, 0],
        "Hybrid + fusi tetap": lambda X: X.mean(axis=1),
        "Hybrid + LambdaMART": lambda X: ranker.predict(X),
    }
    abl = {m: {"nd": {k: [] for k in P.K_LIST}, "pr": {k: [] for k in P.K_LIST},
               "mp": {k: [] for k in P.K_LIST}} for m in methods_abl}
    for i in test_pos:
        X = all_X[i]
        graded = all_graded[i]
        high = (graded >= 2).astype(int)
        for m, fn in methods_abl.items():
            order = np.argsort(-fn(X))
            for k in P.K_LIST:
                abl[m]["nd"][k].append(P.ndcg_graded(order, graded, k))
                abl[m]["pr"][k].append(P.precision_at(order, high, k))
                abl[m]["mp"][k].append(P.ap_at(order, high, k))
    ablation = [{
        "metode": m,
        "ndcg10": round(float(np.mean(abl[m]["nd"][10])) * 100, 2),
        "prec10": round(float(np.mean(abl[m]["pr"][10])) * 100, 2),
        "map10": round(float(np.mean(abl[m]["mp"][10])) * 100, 2),
        "ndcg5": round(float(np.mean(abl[m]["nd"][5])) * 100, 2),
        "ndcg20": round(float(np.mean(abl[m]["nd"][20])) * 100, 2),
    } for m in methods_abl]

    prop = abl["Hybrid + LambdaMART"]["nd"][10]
    wilcoxon_res = []
    for m in methods_abl:
        if m == "Hybrid + LambdaMART":
            continue
        try:
            stat, pv = wilcoxon(prop, abl[m]["nd"][10])
        except ValueError:
            pv = float("nan")
        winner = "LambdaMART unggul" if np.mean(prop) > np.mean(abl[m]["nd"][10]) else "baseline unggul"
        wilcoxon_res.append({"baseline": m, "p": f"{pv:.3g}", "winner": winner})

    _imp = [(f, int(v)) for f, v in zip(P.FEAT, ranker.feature_importances_)]
    _imax = max((v for _, v in _imp), default=1) or 1
    feat_importance = sorted(
        [{"feature": f, "importance": v, "pct": round(v / _imax * 100, 1)} for f, v in _imp],
        key=lambda z: z["importance"], reverse=True)

    # 11. Contoh masukan & keluaran (sel 11)
    t = test_pos[0]
    q_idx = int(elig[t])
    pool = all_pool[t]
    X = all_X[t]
    graded = all_graded[t]
    scores = ranker.predict(X)
    order = np.argsort(-scores)
    topk = pool[order[:10]]
    example = {
        "input": {
            "job_title": str(df.at[q_idx, "job_title"]),
            "skills": str(df.at[q_idx, "skills"])[:120],
            "location": str(df.at[q_idx, "location"]) or "-",
            "category": str(df.at[q_idx, "title_category"]),
        },
        "output": [{
            "rank": r + 1,
            "job_title": str(df.at[j, "job_title"]),
            "location": str(df.at[j, "location"]) or "-",
            "category": str(df.at[j, "title_category"]),
            "relevansi": int(graded[order[r]]),
            "skor_ltr": round(float(np.sort(scores)[::-1][r]), 3),
        } for r, j in enumerate(topk)],
    }

    eval_results = {
        "data_size": int(len(df)),
        "n_queries": int(n),
        "train_size": int(len(train_pos)),
        "test_size": int(len(test_pos)),
        "k_list": P.K_LIST,
        "metrics": metrics,
        "ablation": ablation,
        "wilcoxon": wilcoxon_res,
        "feature_importance": feat_importance,
        "example": example,
        "category_counts": {str(k): int(v) for k, v in category_counts.items()},
        "availability": {str(k): float(v) for k, v in availability.items()},
        "hyperparams": {
            "alpha": P.ALPHA, "beta": P.BETA, "pool": P.POOL,
            "weights": {"title": P.W_TITLE, "skills": P.W_SKILLS, "desc": P.W_DESC},
            "sbert_model": P.SBERT_MODEL,
            "tfidf_dim": int(tfidf_norm.shape[1]), "sbert_dim": int(sbert_norm.shape[1]),
        },
    }

    # ------------------------------------------------------------------
    # Simpan artifact
    # ------------------------------------------------------------------
    joblib.dump(vec_title, model_dir / "vectorizer_title.joblib")
    joblib.dump(vec_skills, model_dir / "vectorizer_skills.joblib")
    joblib.dump(vec_desc, model_dir / "vectorizer_desc.joblib")
    joblib.dump(tfidf_norm, model_dir / "tfidf_norm.joblib")
    joblib.dump(ranker, model_dir / "ranker.joblib")
    joblib.dump({"smin": smin, "smax": smax, "n_rows": int(len(df))}, model_dir / "meta.joblib")
    joblib.dump(eval_results, model_dir / "eval_results.joblib")
    _log("      Artifact tersimpan di: %s" % model_dir, verbose)
    _log("=" * 60, verbose)
    _log("SELESAI. Jalankan: python manage.py runserver", verbose)
    return eval_results
