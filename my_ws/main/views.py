from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth import login, authenticate
from .forms import RegisterForm
# Create your views here.
def home(request):
    return render(request, 'main/home.html',{'center_contRowP':'5',
											    'center_contCol':'-6',
												'center_contColP':'3',
                                                'left_contCol':'-3',
                                                'right_contCol':'-3'})

def registrate(response):
    if response.method == "POST":
        form = RegisterForm(response.POST)
        if form.is_valid():
            form.save()
        return redirect("/")
    else:
        form = RegisterForm()
    return render(response, 'main/signUp.html', {'form':form,
                                                'center_contRowP':'5',
											    'center_contCol':'-6',
												'center_contColP':'3',
                                                'left_contCol':'-3',
                                                'right_contCol':'-3'})