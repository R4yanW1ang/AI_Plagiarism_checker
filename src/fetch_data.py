import re
import os
from canvasapi import Canvas
from bs4 import BeautifulSoup

class CanvasSubmissionDownloader:
    def __init__(self, api_url, api_key, course_id, assignment_id):
        """Initialize Canvas API connection and set course and assignment details."""
        self.api_url = api_url
        self.api_key = api_key
        self.course_id = course_id
        self.assignment_id = assignment_id
        self.download_path = f'data/assignment_{assignment_id}/submissions'
        self.canvas = Canvas(self.api_url, self.api_key)
        os.makedirs(self.download_path, exist_ok=True)


    def get_assignment_submissions(self):
        """Retrieve and download submissions for the specified assignment."""
        course = self.canvas.get_course(self.course_id)
        assignment = course.get_assignment(self.assignment_id)

        for submission in assignment.get_submissions():
            user = submission.user_id
            print(f'Processing submission from User ID: {user}')

            if hasattr(submission, 'body') and submission.body:
                cleaned_submission = self.clean_submission_text(submission.body)
                text_file_path = os.path.join(self.download_path, f'submission_{user}.txt')
                with open(text_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(cleaned_submission)
                print(f'Text submission saved for User ID {user}: {text_file_path}')
            
            else:
                print(f'No text or file submission found for User ID {user}.')


    @staticmethod
    def clean_submission_text(raw_text):
        """Cleans HTML tags and unnecessary metadata from submission text."""
        soup = BeautifulSoup(raw_text, "html.parser")
        cleaned_text = soup.get_text()
        cleaned_text = re.sub(r'data-start="\d+"|data-end="\d+"', '', cleaned_text)
        return cleaned_text
