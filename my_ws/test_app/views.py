from django.shortcuts import render

def index(request):
    return render(request, 'test_app/index.html', context={'text': 'Hello'})

def home_page(request):
    return render(request, 'test_app/home_page.html', context={'text': 'Hello'})
# Create your views here.
