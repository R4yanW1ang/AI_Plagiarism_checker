# Plagiarism Detection Module

This project detects AI-generated content in student submissions and generates interpretability-focused reports.

# set up the environment by running

`pip install -r requirements.txt`

# Configuration

Plug in your own configurations into the config.json in the root directory including:
- canvas token
- canvas url
- COURSE_ID
- ASSIGNMENT_ID

```  
"canvas": {  
    "token": "your own canvas token",  
    "url": "your canvas",  
    "COURSE_ID": 110,  
    "ASSIGNMENT_ID": 229   
}  
```

# Run main.py

`python src/main.py`
