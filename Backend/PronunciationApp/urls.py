from django.urls import path, re_path
from . import views

urlpatterns = [
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("signup/", views.signup, name="signup"),
    path("", views.portfolio, name="portfolio"),
    re_path(r"^question/(?P<action>\w+)(?:/(?P<id>\d+))?/$", views.question, name="question"),
]