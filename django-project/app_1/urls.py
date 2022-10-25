from django.urls import path
from . import views

urlpatterns = [
	path('',views.home,name = 'home'),
	path('colrow_test/',views.colrow_test,name = 'colrow_test_name'),
	path('login_form/',views.entireTo_login_form,name = 'login_form'),
	path('signin_form/',views.entireTo_signin_form,name = 'signin_form'),
]
