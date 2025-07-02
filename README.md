# Écoute-Moi 
A French pronunciation examiner for students studying French Beginners or Continuers to practice their speaking portfolio and receive real-time, word-by-word feedback.

## Features
The core features of this application are:
- An authentication system that allows users to securely access their speaking portfolio and logout once finished.
- A portfolio page which gives users the ability to add, edit and delete existing questions, while viewing their speaking portfolio in its entirety.
- A practice session page that allows users to choose a question they want to practice and directs them to a page with the question and their pre-defined response.
- The ability to click on the question to hear it being spoken (simulating an actual speaking examination) and hover any of the words in a response to hear a correct and accurate pronunciation (from Google Text-To-Speech).
- An incomplete pronunciation scoring functionality, that if working, will change the colour of words to reflect the level of pronunciation of the student.

## File Structure

```plaintext
French-Pronunciation-Examiner/
├── Backend/
│   ├── Backend/          
│   ├── PronunciationApp/ 
│   ├── manage.py
│   ├── db.sqlite3
├── Model Training/
├── README.md
├── requirements.txt
```
The highest level **Backend** folder contains the Django web application aspect of this project, along with the incompletely integrated model. Note that the nested **Backend** folder contains the settings for the entire project, whereas the nested **PronunciationApp** folder contains all the necessary views and URLs for the application. This is in accordance with Django's typical project structure, which is often designed to have multiple applications within one project.

The **Model Training** folder contains all the scripts that were used to preprocess audio files, as well as the files used to actually train the model with the Pytorch framework. Do keep in mind that the model was prepared and trained in Google Collab as it allowed for faster runtimes using cloud GPUs, and that some of this code won't work 'out of the box' in different environments.

## Usage
To run the application itself:

1. Make sure you have Python 3.8+ installed.
2. Install the required libraries and frameworks:

   ```bash
   pip install -r requirements.txt
   ```
3. Navigate to the Backend directory:
   ```bash
   cd Backend
   ```
4. Apply the database migrations:
   ```bash
   python manage.py migrate
   ```
5. Start the development server:
   ```bash
   python manage.py runserver
   ```
Finally, you can open your browser and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to use the application.