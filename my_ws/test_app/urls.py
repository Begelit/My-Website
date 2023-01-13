from django.urls import path

from .views import index,home_page

urlpatterns = [
    path('index/', index),
    #path('', index),
    path('', home_page),
]