"""
Django settings for the Hybrid Job Recommender web app.

Mengimplementasikan notebook 'Algoritma_FINAL_Hybrid_LearningToRank.ipynb'
(Hybrid TF-IDF-SBERT + LambdaMART) sebagai aplikasi web dengan autentikasi
pengguna (register, login) dan penyimpanan profil pencari kerja.
"""
from pathlib import Path

# webapp/  (berisi manage.py, config/, recommender/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Root proyek (berisi folder Model/ dan Dataset/)
PROJECT_ROOT = BASE_DIR.parent

# Folder artifact model (data.csv, vectorizer_*.joblib, sbert_norm.joblib, ranker.joblib, ...)
MODEL_DIR = PROJECT_ROOT / "Model"
DATASET_DIR = PROJECT_ROOT / "Dataset"

# Keamanan: kunci dev (ganti untuk produksi)
SECRET_KEY = "django-insecure-jobrec-dev-key-change-me"
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "recommender",
]

MIDDLEWARE = [
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
     "OPTIONS": {"min_length": 6}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
]

# Alur autentikasi
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "index"
LOGOUT_REDIRECT_URL = "login"

LANGUAGE_CODE = "id"
TIME_ZONE = "Asia/Jakarta"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
