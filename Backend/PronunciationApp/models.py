from django.db import models
from django.contrib.auth.models import User

class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='student')
    fr_course = models.CharField(max_length=3)

class Question(models.Model):
    question = models.CharField(max_length=1024)
    response = models.CharField(max_length=2048)
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='questions')