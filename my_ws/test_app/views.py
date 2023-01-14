from django.shortcuts import render

def index(request):
    return render(request, 'test_app/index.html', context={'text': 'Hello'})

def home_page(request):
    return render(request, 'test_app/home_page.html', context={'text': 'Hello'})

def transformer(request):
    return render(request, 'test_app/base3.html')
# Create your views here.
