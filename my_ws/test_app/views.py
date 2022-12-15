from django.shortcuts import render

def index(request):
    return render(request, 'test_app/index.html', context={'text': 'Hello'})

# Create your views here.
