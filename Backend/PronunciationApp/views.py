"""This file contains all the endpoints for the web application.

Each of these view functions is associated with a specific URL in urls.py, 
and are run when the respective URL endpoint is accessed.
"""

from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import os
from gtts import gTTS

from .helpers import *
from .SNN import *
from .models import Student, Question

def login_view(request):
    """Handle all user attempts to login to the application and redirect when necessary.

    Args:
        request: HttpRequest object containing request metadata.
    
    Returns:
        HttpResponse: Renders the login page or redirects to the portfolio if authenticated.
    """
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

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
    """Log out the current user and redirect to login page.

    Args:
        request: HttpRequest object containing request metadata.
    
    Returns:
        HttpResponse: Redirects to the login page.
    """
    logout(request)
    return redirect('login')

def signup(request):
    """Handle user signup. Validate and create new user accounts.

    Args:
        request: HttpRequest object containing request metadata.
    
    Returns:
        HttpResponse: Renders the signup page or redirects to login after successful signup.
    """
    if request.method == "POST":
        firstname = request.POST["firstname"]
        username = request.POST["username"]
        password = request.POST["password"]

        username_is_valid = validate_username(username)
        password_is_valid = validate_password(password)

        if username_is_valid[0] != True:
            return render(request, "signup.html", {
                'authenticated': False,
                'message': username_is_valid[1],
            })
        
        if password_is_valid[0] != True:
            return render(request, "signup.html", {
                'authenticated': False,
                'message': password_is_valid[1],
            })

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
    """Display the user's portfolio of questions if authenticated.

    Args:
        request:  HttpRequest object containing request metadata.
    
    Returns:
        HttpResponse: Renders the portfolio page or redirects to the login page.
    """
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
    """Create, edit, or delete a question for the authenticated user.

    Args:
        request: HttpRequest object containing request metadata.
        action: A string of the action to perform: 'create', 'edit', or 'delete'.
        id: The integer ID of the question if editing or deleting.
    
    Returns:
        HttpResponse: Redirects to portfolio or renders a question creation form.
    """
    if request.user.is_authenticated:
        if request.method == "POST":
            if action == "create":

                # Ensure any unneccesary whitespace is removed.
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
    """Allow user to select a question to practice from their portfolio.

    Args:
        request: HttpRequest object containing request metadata.
    
    Returns:
        HttpResponse: Renders the choice page or redirects to practice.
    """
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
    """Render the practice page for a selected question.

    Args:
        request: HttpRequest object containing request metadata.
        id: The integer ID of the question to practice.
    
    Returns:
        HttpResponse: Renders the practice page.
    """
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
    """Receive browser audio, process it, and score the user's pronunciation using the Siamese neural network.

    Args:
        request: HttpRequest object containing request metadata.
    
    Returns:
        JsonResponse: JSON response with the pronunciation score.
    """
    if request.method == "POST":
        audio_file = request.FILES.get('audio')
        word_spoken = request.POST["word"]
        print(word_spoken)

        if audio_file:
            audio_dir = os.path.join(settings.BASE_DIR, "PronunciationApp", "audio")
            
            os.makedirs(audio_dir, exist_ok=True)

            file_path = os.path.join(audio_dir, audio_file.name)
            output_path = os.path.join(audio_dir, "human.mp3")

            with open(file_path, 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)

            if os.path.exists(output_path):
                os.remove(output_path)

            convert_to_mp3(file_path, output_path)

            get_standard_pronunciation(word_spoken)

            processed_spoken = load_and_process(output_path)

            standard_path = os.path.join(audio_dir, "spoken.mp3")
            processed_standard = load_and_process(standard_path)

            processed_standard = processed_standard.to(dtype=torch.float32)
            processed_spoken = processed_spoken.to(dtype=torch.float32)
            
            models_dir = os.path.join(settings.BASE_DIR, "PronunciationApp", "models")
            saved_model_path = os.path.join(models_dir, "0.1474_Model.pth")

            model = SiameseNetwork()
            model.load_state_dict(torch.load(saved_model_path, map_location='cpu'))
            model.eval()

            with torch.no_grad():
                output = model(processed_standard, processed_spoken)
                print(output)

            score = None

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
    """Generate an audio file with correct pronunciation for the given French phrase.

    Args:
        request: HttpRequest object containing request metadata.
        phrase (str): A string of the phrase to generate audio for.
    
    Returns:
        JsonResponse: JSON response with the URL for correct pronunciation.
    """
    audio_dir = settings.PRONUNCIATION_AUDIO_ROOT
    os.makedirs(audio_dir, exist_ok=True)
    filename = f"{phrase.replace(' ', '_').replace('?', '')}.mp3" # Prevent any issues with filenames
    file_path = os.path.join(audio_dir, filename)

    if not os.path.exists(file_path):
        tts = gTTS(text=phrase, lang='fr')
        tts.save(file_path)

    audio_url = f"{settings.PRONUNCIATION_AUDIO_URL}{filename}"
    return JsonResponse({"phrase": phrase, "audio_url": audio_url})