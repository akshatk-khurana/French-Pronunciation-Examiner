from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from .models import Student, Question
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
    else:
        student = request.user.student
        questions = list(student.questions.all())

        return render(request, "portfolio.html", {
            'authenticated': request.user.is_authenticated,
            'name': request.user.first_name,    
            'questions': questions
        })
    
def question(request, action, id=None):
    if request.user.is_authenticated:
        if request.method == "POST":
            if action == "create":
                question = request.POST["question"].strip()
                response = request.POST["response"].strip()
                
                new_question = Question(
                    student=request.user.student,
                    question=question,
                    response=response
                )

                new_question.save()

            elif action == "edit":
                question = request.POST["question"].strip()
                response = request.POST["response"].strip()

                question_to_edit = Question.objects.get(id=id, student=request.user.student)
                question_to_edit.question = question
                question_to_edit.response = response

                question_to_edit.save()

            elif action == "delete":
                question = Question.objects.get(id=id, student=request.user.student)
                question.delete()

            return redirect('portfolio')
        elif request.method == "GET":
            if action == "edit":
                question = Question.objects.get(id=id, student=request.user.student)

                return render(request, "question.html", {
                    'authenticated': True,
                    'name': request.user.first_name,
                    'question': question.question,
                    'response': question.response,
                    'action': "Edit",
                })

            else:
                return render(request, "question.html", {
                    'authenticated': True,
                    'name': request.user.first_name,
                    'question': '',
                    'response': '',
                    'action': "Add",
                })