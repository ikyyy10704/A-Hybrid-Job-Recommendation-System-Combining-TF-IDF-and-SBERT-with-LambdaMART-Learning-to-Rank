# Web Rekomendasi Lowongan — Hybrid TF-IDF–SBERT + LambdaMART

Implementasi web (Django) dari notebook
`Algoritma_FINAL_Hybrid_LearningToRank.ipynb`.

## Arsitektur

```
Implementasi/
├── Model/                 # data.csv + artifact model (dibangkitkan)
├── Dataset/               # JobPosting.csv (sumber mentah)
└── webapp/
    ├── manage.py
    ├── config/            # proyek Django (settings, urls, wsgi)
    └── recommender/       # aplikasi
        ├── ml/
        │   ├── pipeline.py   # fungsi & konstanta inti (sama dgn notebook)
        │   ├── prepare.py    # bangkitkan artifact (SBERT + LambdaMART + evaluasi)
        │   └── engine.py     # serving: muat artifact, recommend()
        ├── management/commands/prepare_artifacts.py
        ├── templates/recommender/   # index.html, dashboard.html, base.html
        └── views.py
```

## Cara menjalankan

```bash
cd webapp

# 1) (sekali) siapkan database autentikasi + tabel profil
python manage.py migrate

# 2) (sekali) bangkitkan artifact model: encode SBERT seluruh data + latih LambdaMART.
#    BERAT di CPU (puluhan menit). Hasil disimpan ke ../Model/.
python manage.py prepare_artifacts

# 3) jalankan server
python manage.py runserver
```

Buka <http://127.0.0.1:8000/>. Aplikasi memerlukan login:

- **Daftar** akun di `/register/` lalu **masuk** di `/login/`.
- Halaman **Profil Saya** (`/profile/`) menyimpan data pencari kerja
  (posisi, skill, pengalaman, pendidikan, lokasi) yang otomatis mengisi form rekomendasi.
- Halaman **Rekomendasi** (`/`) dan **Dashboard Evaluasi** (`/dashboard/`) butuh login.
- Tema UI: terang/clean, sidebar + kartu metrik + grafik (CSS di
  `recommender/static/recommender/app.css`).

> Akun demo (dari verifikasi): username `demo`, password `rahasia123`.

## API JSON

```bash
curl -X POST http://127.0.0.1:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"job_title":"FullStack Developer","skills":"php, javascript, mysql","location":"jakarta","top_k":10}'
```

## Catatan

- Artifact (`sbert_norm.joblib`, `ranker.joblib`, dst.) di-`load` lewat `joblib`
  (pickle). File ini dibangkitkan secara lokal oleh `prepare_artifacts`, jadi
  tepercaya. Embedding SBERT di-cache; jalankan dengan `--no-cache` untuk hitung ulang.
- Tidak memakai database relasional — seluruh data berasal dari artifact.
