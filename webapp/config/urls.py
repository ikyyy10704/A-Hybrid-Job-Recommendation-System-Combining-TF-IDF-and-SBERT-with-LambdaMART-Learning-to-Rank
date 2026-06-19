from django.contrib.auth import views as auth_views
from django.urls import path
from django.views.decorators.csrf import ensure_csrf_cookie

from recommender import views
from recommender.forms import LoginForm

urlpatterns = [
    path("", views.index, name="index"),
    path("search/", views.search, name="search"),
    path("map/", views.job_map, name="job_map"),
    path("job/<int:idx>/", views.job_detail, name="job_detail"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("profile/", views.profile, name="profile"),
    path("api/recommend", views.api_recommend, name="api_recommend"),

    # Autentikasi
    path("login/", ensure_csrf_cookie(auth_views.LoginView.as_view(
        template_name="recommender/login.html",
        authentication_form=LoginForm,
        redirect_authenticated_user=True)), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("register/", views.register, name="register"),
]
