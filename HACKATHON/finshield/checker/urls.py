from django.urls import path
from . import views

urlpatterns = [
    path("", views.check_sms, name="check_sms"),
]
