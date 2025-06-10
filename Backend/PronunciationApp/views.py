from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from .models import Student, Question
from django.contrib.auth.models import User
from gtts import gTTS
from .helpers import *
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings

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
            
def choice(request):
    if request.user.is_authenticated:
        if request.method == "POST":
            chosen_question_id = int(request.POST["chosen_question"])
            return redirect('practice', id=chosen_question_id)

        elif request.method == "GET":
            portfolio_questions = list(Question.objects.filter(student=request.user.student))
            return render(request, "choice.html", {
                'authenticated': True,
                'name': request.user.first_name,
                'questions': portfolio_questions,
            })

def practice(request, id):
    if request.user.is_authenticated:
        if request.method == "GET":
            question = Question.objects.get(id=id, student=request.user.student)
            
            words = question.response.split()

            return render(request, "practice.html", {
                'authenticated': True,
                'question': question,
                'response': words,
            })

@csrf_exempt
def score_pronunciation(request):
    if request.method == "POST":
        audio_file = request.FILES.get('audio')
        word_spoken = request.POST["word"]

        if audio_file:
            print("Audio file received!")

            audio_dir = os.path.join(settings.BASE_DIR, "PronunciationApp", "audio")
            os.makedirs(audio_dir, exist_ok=True)

            file_path = os.path.join(audio_dir, audio_file.name)
            output_path = os.path.join(audio_dir, "recording.mp3")

            with open(file_path, 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)

            if os.path.exists(output_path):
                os.remove(output_path)

            convert_to_mp3(file_path, output_path)

            # print(word_spoken)

            score = "OK" # TO-DO
            # Remove file once processed.

            print("Returning score")
            return JsonResponse(
                {
                    "success": True, 
                    "message": f"Audio received and process as {audio_file.name}.",
                    "score": score,
                }
            )
        else:
            return JsonResponse({"success": False, "message": "No audio file received."})
        
@csrf_exempt
def get_pronunciation(request, phrase):
    audio_dir = settings.PRONUNCIATION_AUDIO_ROOT
    os.makedirs(audio_dir, exist_ok=True)
    filename = f"tts_{phrase.replace(' ', '_').replace('?', '')}.mp3"
    file_path = os.path.join(audio_dir, filename)

    if not os.path.exists(file_path):
        tts = gTTS(text=phrase, lang='fr')
        tts.save(file_path)

    audio_url = f"{settings.PRONUNCIATION_AUDIO_URL}{filename}"
    return JsonResponse({"phrase": phrase, "audio_url": audio_url})