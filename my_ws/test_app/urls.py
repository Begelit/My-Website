from django.urls import path

from .views import index,home_page,transformer

urlpatterns = [
    #path('', index),
    path('', home_page),
    path('index/', index),
    path('transformer/', transformer),
]