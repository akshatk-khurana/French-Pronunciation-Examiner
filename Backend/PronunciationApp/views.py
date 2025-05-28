from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from .models import Student
from django.contrib.auth.models import User
from .helpers import *

def login_view(request):
    if request.method == "POST":
        username = sanitise(request.POST["username"])
        password = sanitise(request.POST["password"])

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('portfolio')
        else:
            return render(request, "login.html", {
                'authenticated': False,
                'message': "Incorrect details.",
                'error': True
            })
    elif request.method == "GET":
        return render(request, "login.html", {
            'authenticated': False,
        })
    
def logout_view(request):
    logout(request)
    return redirect('login')

def signup(request):
    if request.method == "POST":
        firstname = sanitise(request.POST["firstname"])
        username = sanitise(request.POST["username"])
        password = sanitise(request.POST["password"])
        course = request.POST["course"]

        if not User.objects.filter(username=username).exists():
            new_user = User.objects.create_user(
                username=username,
                password=password
            )

            new_user.first_name = firstname
            new_user.save()

            new_student = Student(
                fr_course=course,
                user=new_user
            )
            
            new_student.save()
            return redirect('login')
        else:
            return render(request, "signup.html", {
                'authenticated': False,
                'message': "Username exists, choose another one."
            })
    elif request.method == "GET":
        return render(request, "signup.html", {'authenticated': False})

def portfolio(request):
    if not request.user.is_authenticated:
        return redirect('login')