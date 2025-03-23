# Handwriting OCR Readme

## Project Overview
This is a simple Flask application that serves as a starting point for building a web application. It includes basic routes, template rendering, and database integration.

## Features
- User authentication (Login/Signup)
- Database integration (SQLite/PostgreSQL)
- API endpoints
- Template rendering using Jinja2
- Static file handling (CSS, JS, Images)

## Installation
### Prerequisites
Ensure you have the following installed:
- Python (>= 3.x)
- pip (Python package manager)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/gokul-1998/handwriting_recognition/
   cd handwriting_recognition/
2. Create environment file (optional):
   ```sh
   python3 -m venv env
   source env/bin/activate #windows
   ```
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Folder Structure
```
handwriting_recognition/
│── app.py            # Main Flask application file
│── requirements.txt  # Dependencies
│── templates/        # HTML templates (Jinja2)
│── static/           # Static files (CSS)
|── .gitignore        # Ignore file
│── README.md         # Project documentation
```
