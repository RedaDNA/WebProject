from django.urls import path
from django.contrib import admin

from .views import home, result

from django.urls import path
urlpatterns = [
    path('', home, name='home'),
    path('result/', result, name='result')
]