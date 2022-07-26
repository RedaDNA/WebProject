from django.urls import path
from django.contrib import admin

from .views import home, result,urlresult

from django.urls import path
urlpatterns = [
    path('', home, name='home'),
    path('result/', result, name='result'),
    path('urlresult/', urlresult, name='urlresult')

]