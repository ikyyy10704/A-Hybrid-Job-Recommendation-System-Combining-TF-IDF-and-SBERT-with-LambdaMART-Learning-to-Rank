"""View web: autentikasi, profil, form rekomendasi, dashboard evaluasi, API JSON."""
import json

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie

from .forms import ProfileForm, RegisterForm
from .models import Profile
from .ml import geo
from .ml.engine import get_engine


def _profile_from_request(request):
    src = request.POST if request.method == "POST" else request.GET
    return {
        "job_title": (src.get("job_title", "") or "").strip(),
        "skills": (src.get("skills", "") or "").strip(),
        "experience": (src.get("experience", "") or "").strip(),
        "education": (src.get("education", "") or "").strip(),
        "location": (src.get("location", "") or "").strip(),
        "top_k": _safe_int(src.get("top_k", 10), 10),
    }


def _safe_int(v, default):
    try:
        return max(1, min(50, int(v)))
    except (TypeError, ValueError):
        return default


def _points_from_results(results):
    """Agregasi lokasi hasil rekomendasi menjadi titik peta {name,lat,lng,count}."""
    from collections import defaultdict
    counts, coords = defaultdict(int), {}
    for r in results:
        hit = geo.resolve(r.get("location", ""))
        if not hit:
            continue
        name, lat, lng = hit
        counts[name] += 1
        coords[name] = (lat, lng)
    return [{"name": n, "lat": coords[n][0], "lng": coords[n][1], "count": c}
            for n, c in counts.items()]


def _map_focus(location):
    """Pusat & zoom peta: ke kota yang dicari bila dikenali, jika tidak ke Indonesia."""
    hit = geo.resolve(location or "")
    if hit:
        name, lat, lng = hit
        return {"center": [lat, lng], "zoom": 10, "label": name}
    return {"center": [-2.5, 118.0], "zoom": 5, "label": "Indonesia"}


def _eval_summary(ev):
    """Ringkasan metrik evaluasi untuk panel pendukung di halaman rekomendasi.

    Menyajikan metrik @10 yang sama dengan dashboard, plus keunggulan metode
    usulan (Hybrid + LambdaMART) atas baseline terbaik pada NDCG@10.
    """
    if not ev:
        return None
    ks = ev["k_list"]
    m10 = ev["metrics"].get(10) or ev["metrics"][ks[min(1, len(ks) - 1)]]
    proposed = next((a for a in ev["ablation"] if a["metode"] == "Hybrid + LambdaMART"), None)
    baselines = [a for a in ev["ablation"] if a["metode"] != "Hybrid + LambdaMART"]
    best_base = max(baselines, key=lambda a: a["ndcg10"]) if baselines else None
    gain = round(proposed["ndcg10"] - best_base["ndcg10"], 2) if (proposed and best_base) else None
    return {
        "ndcg": m10["ndcg"], "precision": m10["precision"], "map": m10["map"],
        "data_size": ev["data_size"], "test_size": ev["test_size"],
        "best_base": best_base["metode"] if best_base else None,
        "gain": gain,
    }


# ============================================================
# Autentikasi
# ============================================================
@ensure_csrf_cookie
def register(request):
    if request.user.is_authenticated:
        return redirect("index")
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.email = form.cleaned_data.get("email", "")
            user.save()
            Profile.objects.create(user=user, full_name=form.cleaned_data.get("full_name", ""))
            login(request, user)
            messages.success(request, "Akun berhasil dibuat. Selamat datang!")
            return redirect("index")
    else:
        form = RegisterForm()
    return render(request, "recommender/register.html", {"form": form})


@login_required
@ensure_csrf_cookie
def profile(request):
    prof, _ = Profile.objects.get_or_create(user=request.user)
    if request.method == "POST":
        form = ProfileForm(request.POST, instance=prof)
        if form.is_valid():
            form.save()
            messages.success(request, "Profil berhasil disimpan.")
            return redirect("profile")
    else:
        form = ProfileForm(instance=prof)
    return render(request, "recommender/profile.html", {"form": form, "prof": prof})


# ============================================================
# Rekomendasi
# ============================================================
# Banyaknya rekomendasi yang diambil saat profil sudah terisi (ditampilkan
# 10 dulu, sisanya lewat tombol "Tampilkan lainnya" — tanpa encode ulang).
AUTO_LIMIT = 100
INITIAL_SHOWN = 10


@login_required
def index(request):
    engine = get_engine()
    prof, _ = Profile.objects.get_or_create(user=request.user)
    has_profile = bool((prof.job_title or "").strip() or (prof.skills or "").strip())
    ctx = {
        "ready": engine.is_ready(),
        "missing": engine.missing_artifacts(),
        "prof": prof,
        "has_profile": has_profile,
        "initial_shown": INITIAL_SHOWN,
    }
    if ctx["ready"] and has_profile:
        results = engine.recommend(
            title=prof.job_title, skills=prof.skills,
            description=prof.experience, experience=prof.experience,
            education=prof.education, location=prof.location,
            top_k=AUTO_LIMIT,
        )
        ctx["results"] = results
        ctx["remaining"] = max(0, len(results) - INITIAL_SHOWN)
        ctx["map_data"] = {**_map_focus(prof.location), "points": _points_from_results(results)}
        ctx["eval"] = _eval_summary(engine.eval_results)
    return render(request, "recommender/index.html", ctx)


@login_required
@ensure_csrf_cookie
def search(request):
    """Pencarian lowongan ad-hoc — murni dari input, TANPA memakai profil."""
    engine = get_engine()
    ctx = {"ready": engine.is_ready(), "missing": engine.missing_artifacts(), "q": {}}
    if request.method == "POST" and ctx["ready"]:
        q = _profile_from_request(request)
        ctx["q"] = q
        if q["job_title"] or q["skills"]:
            results = engine.recommend(
                title=q["job_title"], skills=q["skills"],
                description=q["experience"], experience=q["experience"],
                education=q["education"], location=q["location"], top_k=AUTO_LIMIT)
            ctx["results"] = results
            ctx["remaining"] = max(0, len(results) - INITIAL_SHOWN)
            ctx["initial_shown"] = INITIAL_SHOWN
        else:
            ctx["error"] = "Isi minimal Posisi atau Skills."
    return render(request, "recommender/search.html", ctx)


@login_required
def job_map(request):
    engine = get_engine()
    if not engine.is_ready():
        return render(request, "recommender/map.html",
                      {"ready": False, "missing": engine.missing_artifacts()})
    dist = engine.location_distribution()
    return render(request, "recommender/map.html",
                  {"ready": True, "points": dist["points"], "stats": dist,
                   "top": dist["points"][:12]})


@login_required
def job_detail(request, idx):
    engine = get_engine()
    if not engine.is_ready():
        return render(request, "recommender/job_detail.html",
                      {"ready": False, "missing": engine.missing_artifacts()})
    job = engine.get_job(idx)
    if job is None:
        return render(request, "recommender/job_detail.html", {"ready": True, "job": None})
    return render(request, "recommender/job_detail.html", {"ready": True, "job": job})


@login_required
def dashboard(request):
    engine = get_engine()
    if not engine.is_ready():
        return render(request, "recommender/dashboard.html",
                      {"ready": False, "missing": engine.missing_artifacts()})
    engine.load()
    ev = engine.eval_results
    m10 = ev["metrics"].get(10) or ev["metrics"][ev["k_list"][min(1, len(ev["k_list"]) - 1)]]
    return render(request, "recommender/dashboard.html",
                  {"ready": True, "ev": ev, "m10": m10, "chart": _build_chart(ev)})


def _build_chart(ev):
    """Geometri SVG grafik garis NDCG/Precision/MAP vs K (sel plot notebook)."""
    ks = ev["k_list"]
    series = [
        ("NDCG@K", "#3b82f6", [ev["metrics"][k]["ndcg"] for k in ks]),
        ("Precision@K", "#10b981", [ev["metrics"][k]["precision"] for k in ks]),
        ("MAP@K", "#f59e0b", [ev["metrics"][k]["map"] for k in ks]),
    ]
    W, H = 560, 300
    pad_l, pad_r, pad_t, pad_b = 48, 16, 18, 36
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    all_vals = [v for _, _, vals in series for v in vals]
    ymin = max(0, (min(all_vals) // 5) * 5 - 5)
    ymax = min(100, (max(all_vals) // 5) * 5 + 5)
    yspan = max(ymax - ymin, 1)
    n = len(ks)

    def x(i):
        return pad_l + (plot_w * i / (n - 1) if n > 1 else 0)

    def y(v):
        return pad_t + plot_h * (1 - (v - ymin) / yspan)

    out = []
    for name, color, vals in series:
        pts = [(round(x(i), 1), round(y(v), 1)) for i, v in enumerate(vals)]
        out.append({
            "name": name, "color": color,
            "polyline": " ".join(f"{px},{py}" for px, py in pts),
            "points": [{"cx": px, "cy": py, "v": v} for (px, py), v in zip(pts, vals)],
        })
    yticks = []
    for t in range(5):
        val = ymin + yspan * t / 4
        yticks.append({"y": round(y(val), 1), "label": round(val, 1)})
    xticks = [{"x": round(x(i), 1), "label": k} for i, k in enumerate(ks)]
    return {"w": W, "h": H, "pad_l": pad_l, "pad_b": pad_b, "pad_t": pad_t,
            "plot_w": plot_w, "series": out, "yticks": yticks, "xticks": xticks}


# ============================================================
# API JSON
# ============================================================
@csrf_exempt
def api_recommend(request):
    engine = get_engine()
    if not engine.is_ready():
        return JsonResponse({"error": "Model belum siap. Jalankan: python manage.py prepare_artifacts",
                             "missing": engine.missing_artifacts()}, status=503)

    if request.method == "POST" and request.content_type == "application/json":
        try:
            body = json.loads(request.body or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"error": "Body JSON tidak valid."}, status=400)
        profile_data = {
            "job_title": str(body.get("job_title", "")).strip(),
            "skills": str(body.get("skills", "")).strip(),
            "experience": str(body.get("experience", "")).strip(),
            "education": str(body.get("education", "")).strip(),
            "location": str(body.get("location", "")).strip(),
            "top_k": _safe_int(body.get("top_k", 10), 10),
        }
    else:
        profile_data = _profile_from_request(request)

    if not (profile_data["job_title"] or profile_data["skills"]):
        return JsonResponse({"error": "Isi minimal 'job_title' atau 'skills'."}, status=400)

    results = engine.recommend(
        title=profile_data["job_title"], skills=profile_data["skills"],
        description=profile_data["experience"], experience=profile_data["experience"],
        education=profile_data["education"], location=profile_data["location"],
        top_k=profile_data["top_k"],
    )
    return JsonResponse({"profile": profile_data, "count": len(results), "results": results})
