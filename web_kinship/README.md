# Web Kinship Verification System

A web application for verifying kinship relationships between people using facial analysis. This is the simple website to check potential family relationships by uploading images of two people or analyzing group photos for family connections.


## Installation

### Prerequisites
- Python 3.10+
- Node.js 16+
- npm

### Backend Setup
1. Create a virtual environment:
```bash
python -m venv kinship_env
source kinship_env/bin/activate 
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run backend server:
```bash
python backend/main.py
```

### Frontend Setup
1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run development server:
```bash
npm run dev
```

## Usage
1. Open your browser and go to `http://localhost:3000`
2. Choose between pair comparison or group analysis
3. Upload images and get kinship predictions

## Note
Make sure both backend (port 8000) and frontend (port 3000) servers are running simultaneously for the application to work properly.
