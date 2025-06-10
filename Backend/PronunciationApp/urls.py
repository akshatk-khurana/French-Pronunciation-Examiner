from django.urls import path, re_path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("signup/", views.signup, name="signup"),
    path("", views.portfolio, name="portfolio"),
    path("choice/", views.choice, name="choice"),
    path("practice/<int:id>/", views.practice, name="practice"),
    re_path(r"^question/(?P<action>\w+)(?:/(?P<id>\d+))?/$", views.question, name="question"),
    path("score/", views.score_pronunciation, name="score"),
    path("pronounce/<str:phrase>/", views.get_pronunciation, name="pronounce"),
] + static(settings.PRONUNCIATION_AUDIO_URL, document_root=settings.PRONUNCIATION_AUDIO_ROOT)