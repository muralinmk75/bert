from django.urls import path
from . import views
urlpatterns = [
    path('', views.home),
    path('getData', views.home),
]