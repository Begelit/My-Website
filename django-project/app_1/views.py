from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def home(request):
	return render(request, 'home.html',{'top_contRowP':'2','center_contRowP':'2',
										'left_contCol':'','center_contCol':'-9',
										'right_contCol':'-2','left_contColP':'',
										'center_contColP':'5','right_contColP':''})

def colrow_test(request):
	return render(request, 'colrow_testing.html',{'name':''})
	
def entireTo_login_form(request):
	return render(request, 'login_form.html',{'top_contRowP':'5','center_contRowP':'5',
												'left_contCol':'','center_contCol':'-6',
												'right_contCol':'','left_contColP':'',
												'center_contColP':'3','right_contColP':''})

def entireTo_signin_form(request):
	return render(request, 'signin_form.html',{'top_contRowP':'5','center_contRowP':'5',
												'left_contCol':'','center_contCol':'-6',
												'right_contCol':'','left_contColP':'',
												'center_contColP':'3','right_contColP':''})