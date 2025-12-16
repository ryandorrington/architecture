# Architecture Image Analyzer - Implementation Plan
**Project**: AI-Powered Architectural Image Analysis Tool
**Date**: 2025-12-16
**Timeline**: **URGENT** - Interview tomorrow afternoon
**Tech Stack**: React + Flask + Claude Vision API + LangChain

## Overview

A portfolio project demonstrating AI/ML integration skills for Populous AI Team positions. Users upload architectural images and ask questions about architectural styles, elements, and design features. The system uses Claude's vision capabilities with architecture-focused prompt engineering.

## Current State Analysis

**Starting Point**: Empty repository
**Target**: Working MVP that showcases:
- React development (modern hooks, clean UI)
- Flask API development
- Claude Vision API integration
- LangChain usage
- Prompt engineering for domain-specific tasks

**Key Constraint**: Must be buildable and runnable in < 4 hours

## Desired End State

A locally-runnable application where:
1. User uploads an architectural image (validates image types)
2. User asks a question about the architecture
3. System processes image + question through Claude Vision API
4. Response is displayed with architecture-focused insights
5. User can reset and try another image/question

### Success Verification:
- Upload image from the Kaggle dataset
- Ask "What architectural style is this building?"
- Receive detailed architectural analysis
- Reset and try another image

## What We're NOT Doing

- **No authentication/user management**
- **No database/persistence**
- **No conversation history**
- **No multi-image comparison**
- **No deployment** (local only)
- **No Docker** (simple setup via README)
- **No advanced LangChain features** (chains, agents, tools)
- **No testing** (time constraint - focus on working demo)

## Implementation Approach

Build in order of critical path:
1. **Backend first** - Ensures Claude API integration works
2. **Frontend second** - Quick UI once backend proven
3. **Polish last** - Styling and README only if time permits

Use proven patterns from research to minimize debugging time.

---

## Phase 1: Backend Foundation

### Overview
Set up Flask API with Claude Vision integration. This is the critical technical demonstration.

### Changes Required:

#### 1. Project Structure & Dependencies

**Create**: `backend/requirements.txt`
```txt
flask==3.1.0
flask-cors==6.0.2
langchain-anthropic==0.3.8
langchain-core==0.3.29
pydantic==2.10.5
python-dotenv==1.0.1
pillow==11.0.0
```

**Create**: `backend/.env.example`
```env
ANTHROPIC_API_KEY=your_api_key_here
```

**Create**: `backend/.gitignore`
```
.env
__pycache__/
*.pyc
```

#### 2. Main Flask Application

**Create**: `backend/app.py`

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import os
import base64
from PIL import Image
import io
from dotenv import load_dotenv
from typing import List

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configure max upload size (16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Define structured output schema for architectural analysis
class ArchitecturalMetadata(BaseModel):
    """Structured metadata extracted from architectural image"""
    style: str = Field(description="Primary architectural style (e.g., Gothic Revival, Modernist, Brutalist)")
    period: str = Field(description="Estimated time period or era (e.g., 1850-1870, Early 20th Century)")
    materials: List[str] = Field(description="List of visible construction materials (e.g., stone, brick, glass, steel)")
    key_features: List[str] = Field(description="Notable architectural elements (e.g., pointed arches, flying buttresses, curtain walls)")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image_if_needed(image_data, max_pixels=1.15 * 1024 * 1024):
    """
    Resize image to stay under Claude's recommended limit (1.15 megapixels)
    """
    image = Image.open(io.BytesIO(image_data))
    width, height = image.size
    current_pixels = width * height

    if current_pixels > max_pixels:
        # Calculate new dimensions
        ratio = (max_pixels / current_pixels) ** 0.5
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert back to bytes
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or 'JPEG')
        return buffer.getvalue()

    return image_data

def extract_metadata(base64_image: str, media_type: str) -> ArchitecturalMetadata:
    """
    Step 1: Extract structured architectural metadata using LangChain
    """
    # Initialize LangChain ChatAnthropic with structured output
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Use with_structured_output for JSON response
    structured_llm = llm.with_structured_output(ArchitecturalMetadata)

    # Create message with image
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}"
                }
            },
            {
                "type": "text",
                "text": """Analyze this architectural image and extract structured metadata.

Identify:
- The primary architectural style
- The approximate time period/era
- Visible construction materials
- Key architectural features and elements

Be specific and accurate in your analysis."""
            }
        ]
    )

    # Invoke and get structured response
    metadata = structured_llm.invoke([message])
    return metadata

def generate_detailed_analysis(metadata: ArchitecturalMetadata, user_question: str, base64_image: str, media_type: str) -> str:
    """
    Step 2: Generate detailed answer using metadata + user question
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Create context from metadata
    metadata_context = f"""
Based on initial analysis, this building has been identified as:
- Style: {metadata.style}
- Period: {metadata.period}
- Materials: {', '.join(metadata.materials)}
- Key Features: {', '.join(metadata.key_features)}
"""

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}"
                }
            },
            {
                "type": "text",
                "text": f"""You are an expert architectural analyst with deep knowledge of:
- Architectural styles (Modern, Gothic, Brutalist, Art Deco, etc.)
- Structural elements (columns, arches, facades, etc.)
- Design principles and aesthetics
- Historical context and influences
- Materials and construction techniques

{metadata_context}

User's question: {user_question}

Using the metadata above and your visual analysis of the image, provide a thorough architectural analysis addressing their question. Be detailed and insightful."""
            }
        ]
    )

    response = llm.invoke([message])
    return response.content

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Two-step architectural analysis:
    1. Extract structured metadata using LangChain
    2. Generate detailed answer using metadata + question
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        if 'question' not in request.form:
            return jsonify({"error": "No question provided"}), 400

        file = request.files['image']
        question = request.form['question']

        # Validate file
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, webp"}), 400

        # Read and process image
        image_data = file.read()
        image_data = resize_image_if_needed(image_data)

        # Convert to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')

        # Determine media type
        extension = file.filename.rsplit('.', 1)[1].lower()
        media_type_map = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'webp': 'image/webp'
        }
        media_type = media_type_map.get(extension, 'image/jpeg')

        # Step 1: Extract structured metadata using LangChain
        metadata = extract_metadata(base64_image, media_type)

        # Step 2: Generate detailed analysis
        analysis = generate_detailed_analysis(metadata, question, base64_image, media_type)

        return jsonify({
            "success": True,
            "metadata": {
                "style": metadata.style,
                "period": metadata.period,
                "materials": metadata.materials,
                "key_features": metadata.key_features
            },
            "analysis": analysis,
            "model": "claude-sonnet-4-5-20250929"
        }), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found in environment variables")
        print("Please create a .env file with your API key")
        exit(1)

    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)
```

**Note**: Using LangChain's `with_structured_output()` to extract architectural metadata as a Pydantic model, then feeding that context into a second LLM call for detailed analysis. This demonstrates both structured outputs and multi-step prompting.

### Success Criteria:

#### Automated Verification:
- [x] Dependencies install successfully: `cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt` (Note: Updated langchain-core from 0.3.29 to 0.3.39 to resolve dependency conflict)
- [x] Server starts without errors: `python app.py` (Server running on http://localhost:5000)
- [x] Health check returns 200: `curl http://localhost:5000/health` (Returns: {"status": "healthy"})

#### Manual Verification:
- [x] Test with curl and sample image:
```bash
curl -X POST http://localhost:5000/analyze \
  -F "image=@sample_architecture.jpg" \
  -F "question=What architectural style is this?"
```
- [x] Response contains architectural analysis
- [x] No CORS errors in browser console

**Implementation Note**: Test the backend thoroughly before moving to frontend. This is the core technical demonstration.

---

## Phase 2: React Frontend

### Overview
Build a clean, modern React UI with image upload, preview, and results display.

### Changes Required:

#### 1. Create React App

**Run**:
```bash
npx create-react-app frontend
cd frontend
```

**Install additional dependencies** (if needed for styling):
```bash
npm install --save-dev prettier
```

#### 2. Main Application Component

**Edit**: `frontend/src/App.js`

```javascript
import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [question, setQuestion] = useState('');
  const [metadata, setMetadata] = useState(null);
  const [analysis, setAnalysis] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      const previewUrl = URL.createObjectURL(file);
      setImagePreview(previewUrl);
      // Clear previous results
      setAnalysis('');
      setMetadata(null);
      setError('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedImage || !question.trim()) {
      setError('Please select an image and enter a question');
      return;
    }

    setLoading(true);
    setError('');
    setAnalysis('');
    setMetadata(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedImage);
      formData.append('question', question);

      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setMetadata(data.metadata);
        setAnalysis(data.analysis);
      } else {
        setError(data.error || 'Failed to analyze image');
      }
    } catch (err) {
      setError('Failed to connect to server. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setQuestion('');
    setAnalysis('');
    setMetadata(null);
    setError('');
    const fileInput = document.getElementById('image-upload');
    if (fileInput) fileInput.value = '';
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏛️ Architecture Analyzer</h1>
        <p>AI-powered architectural image analysis using Claude Vision + LangChain</p>
      </header>

      <main className="App-main">
        <form onSubmit={handleSubmit} className="upload-form">
          {/* Image Upload */}
          <div className="form-section">
            <label htmlFor="image-upload" className="upload-label">
              Upload Architectural Image
            </label>
            <input
              id="image-upload"
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/webp"
              onChange={handleImageChange}
              className="file-input"
            />
          </div>

          {/* Image Preview */}
          {imagePreview && (
            <div className="preview-section">
              <img
                src={imagePreview}
                alt="Preview"
                className="image-preview"
              />
            </div>
          )}

          {/* Question Input */}
          <div className="form-section">
            <label htmlFor="question-input" className="input-label">
              Ask a question about the architecture
            </label>
            <input
              id="question-input"
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g., What architectural style is this building?"
              className="text-input"
              disabled={loading}
            />
          </div>

          {/* Action Buttons */}
          <div className="button-group">
            <button
              type="submit"
              disabled={!selectedImage || !question.trim() || loading}
              className="btn btn-primary"
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="btn btn-secondary"
              disabled={loading}
            >
              Reset
            </button>
          </div>
        </form>

        {/* Error Display */}
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {/* Structured Metadata Display */}
        {metadata && (
          <div className="metadata-section">
            <h3>Building Information</h3>
            <div className="metadata-grid">
              <div className="metadata-item">
                <span className="metadata-label">Style:</span>
                <span className="metadata-value">{metadata.style}</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-label">Period:</span>
                <span className="metadata-value">{metadata.period}</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-label">Materials:</span>
                <span className="metadata-value">{metadata.materials.join(', ')}</span>
              </div>
              <div className="metadata-item full-width">
                <span className="metadata-label">Key Features:</span>
                <ul className="metadata-list">
                  {metadata.key_features.map((feature, index) => (
                    <li key={index}>{feature}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Detailed Analysis Display */}
        {analysis && (
          <div className="results-section">
            <h2>Detailed Analysis</h2>
            <div className="analysis-text">
              {analysis}
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Built with React + Flask + Claude Vision API + LangChain</p>
      </footer>
    </div>
  );
}

export default App;
```

#### 3. Styling

**Edit**: `frontend/src/App.css`

```css
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

.App {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.App-header {
  padding: 2rem;
  text-align: center;
  color: white;
}

.App-header h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
}

.App-header p {
  font-size: 1.1rem;
  opacity: 0.9;
}

.App-main {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}

.upload-form {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}

.form-section {
  margin-bottom: 1.5rem;
}

.upload-label,
.input-label {
  display: block;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #333;
}

.file-input {
  width: 100%;
  padding: 0.75rem;
  border: 2px dashed #667eea;
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.3s;
}

.file-input:hover {
  border-color: #764ba2;
}

.preview-section {
  margin-bottom: 1.5rem;
  text-align: center;
}

.image-preview {
  max-width: 100%;
  max-height: 400px;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.text-input {
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 1rem;
  transition: border-color 0.3s;
}

.text-input:focus {
  outline: none;
  border-color: #667eea;
}

.button-group {
  display: flex;
  gap: 1rem;
  margin-top: 1.5rem;
}

.btn {
  flex: 1;
  padding: 0.875rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.btn-secondary {
  background: #f5f5f5;
  color: #333;
}

.btn-secondary:hover:not(:disabled) {
  background: #e0e0e0;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error-message {
  margin-top: 1.5rem;
  padding: 1rem;
  background: #fee;
  border-left: 4px solid #f44;
  border-radius: 4px;
  color: #c33;
}

/* Structured Metadata Section */
.metadata-section {
  margin-top: 2rem;
  background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
  border: 2px solid #667eea;
  border-radius: 12px;
  padding: 1.5rem;
}

.metadata-section h3 {
  color: #667eea;
  margin-bottom: 1rem;
  font-size: 1.25rem;
}

.metadata-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.metadata-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.metadata-item.full-width {
  grid-column: 1 / -1;
}

.metadata-label {
  font-weight: 600;
  color: #555;
  font-size: 0.875rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.metadata-value {
  color: #333;
  font-size: 1rem;
  padding: 0.5rem;
  background: white;
  border-radius: 6px;
}

.metadata-list {
  list-style: none;
  padding: 0;
  margin: 0.5rem 0 0 0;
}

.metadata-list li {
  padding: 0.5rem;
  background: white;
  border-radius: 6px;
  margin-bottom: 0.5rem;
  color: #333;
}

.metadata-list li:before {
  content: "▸ ";
  color: #667eea;
  font-weight: bold;
  margin-right: 0.5rem;
}

/* Detailed Analysis Section */
.results-section {
  margin-top: 2rem;
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}

.results-section h2 {
  color: #333;
  margin-bottom: 1rem;
}

.analysis-text {
  line-height: 1.6;
  color: #555;
  white-space: pre-wrap;
}

.App-footer {
  text-align: center;
  padding: 2rem;
  color: white;
  opacity: 0.8;
}

@media (max-width: 768px) {
  .App-header h1 {
    font-size: 2rem;
  }

  .App-main {
    padding: 1rem;
  }

  .upload-form {
    padding: 1.5rem;
  }

  .button-group {
    flex-direction: column;
  }
}
```

#### 4. Update package.json proxy (optional)

**Edit**: `frontend/package.json`

Add this line to enable proxy for development:
```json
{
  "proxy": "http://localhost:5000",
  ...
}
```

### Success Criteria:

#### Automated Verification:
- [x] React app builds without errors: `npm install && npm start`
- [x] No console errors in browser
- [x] App accessible at `http://localhost:3000`

#### Manual Verification:
- [ ] File upload button works and shows preview
- [ ] Question input accepts text
- [ ] Analyze button disabled when image or question missing
- [ ] Reset button clears all fields and preview
- [ ] UI looks clean and professional on desktop and mobile
- [ ] Loading state shows "Analyzing..." during API call

**Implementation Note**: Test the complete flow: upload → question → analyze → result → reset

---

## Phase 3: Documentation & Polish

### Overview
Create comprehensive README and add final touches.

### Changes Required:

#### 1. Root README

**Create**: `README.md`

```markdown
# 🏛️ Architecture Analyzer

AI-powered architectural image analysis using Claude API & LangChain. Upload architectural images and ask questions about styles, elements, and design features.

## Screenshots

![Upload Interface](Screenshot%20from%202025-12-16%2010-10-59.png)
![Building Information](Screenshot%20from%202025-12-16%2010-11-35.png)
![Detailed Analysis](Screenshot%20from%202025-12-16%2010-13-00.png)
![Markdown Rendering](Screenshot%20from%202025-12-16%2010-25-37.png)

## Features

- 📷 **Image Upload** - Upload architectural images (PNG, JPG)
- 🤖 **AI Analysis** - Powered by Claude Sonnet 4.5 with vision capabilities
- 🏗️ **Architecture-Focused** - Custom prompt engineering for architectural insights
- ⚡ **Real-time** - Fast analysis and response
- 🎨 **Clean UI** - Modern, responsive React interface

## Tech Stack

**Frontend:**
- React (Modern hooks-based approach)
- Pure CSS (No dependencies)
- Fetch API for backend communication

**Backend:**
- Flask (Python web framework)
- LangChain + Anthropic SDK (AI integration)
- Claude Sonnet 4.5 (Vision-enabled model)
- Flask-CORS (Cross-origin support)

## Prerequisites

- Python 3.9+
- Node.js 16+
- Anthropic API key ([Get one here](https://console.anthropic.com/))

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd architecture-analyzer
```

### 2. Set up the backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Add your Anthropic API key to .env
# ANTHROPIC_API_KEY=your_api_key_here
```

### 3. Set up the frontend

```bash
cd ../frontend

# Install dependencies
npm install
```

### 4. Run the application

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # Activate venv if not already active
python app.py
```

Backend will run on `http://localhost:5000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

Frontend will run on `http://localhost:3000`

### 5. Use the application

1. Open `http://localhost:3000` in your browser
2. Upload an architectural image (good dataset here: https://www.kaggle.com/datasets/wwymak/architecture-dataset)
3. Ask a question (e.g., "What architectural style is this?")
4. Click "Analyze" to get AI-powered insights
5. Click "Reset" to try another image

## Example Questions

- "What architectural style is this building?"
- "Describe the key structural elements visible in this image"
- "What materials appear to be used in this construction?"
- "What design principles are evident in this architecture?"
- "What historical period does this architecture represent?"

## Project Structure

```
architecture-analyzer/
├── backend/
│   ├── app.py              # Flask application
│   ├── requirements.txt    # Python dependencies
│   ├── .env.example        # Environment variables template
│   └── .gitignore
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   ├── App.css         # Styling
│   │   └── index.js
│   ├── public/
│   └── package.json
└── README.md
```

## API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### `POST /analyze`
Analyze architectural image with question.

**Request:**
- `image` (file): Image file (PNG, JPG, JPEG)
- `question` (string): Question about the architecture

**Response:**
```json
{
  "success": true,
  "analysis": "Detailed architectural analysis...",
  "model": "claude-sonnet-4-5-20250929"
}
```

## Implementation Notes

**Prompt Engineering:**
The system uses a custom architecture-focused system prompt that guides Claude to:
- Focus on architectural styles and periods
- Identify structural elements
- Analyze design principles
- Consider materials and construction
- Provide historical context

**Image Processing:**
- Images are automatically resized to stay under Claude's 1.15 megapixel recommendation
- Supports multiple formats (PNG, JPG, JPEG)
- Maximum upload size: 16 MB

**LangChain Integration:**
The application uses LangChain's `with_structured_output()` feature to extract structured architectural metadata (style, period, materials, key features) as a Pydantic model. This structured data is then used to provide context for the detailed analysis, demonstrating multi-step prompting and data modeling best practices.

## License

MIT

## Author

Built as a portfolio project to demonstrate AI/ML integration skills.
```

#### 2. Add .gitignore entries

**Edit root**: `.gitignore`

Add:
```
# Python
backend/.env
backend/venv/
backend/__pycache__/
backend/*.pyc

# Node
frontend/node_modules/
frontend/build/
frontend/.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### Success Criteria:

#### Automated Verification:
- [x] README renders correctly on GitHub
- [x] All commands in README work when followed

#### Manual Verification:
- [ ] Someone unfamiliar with the project can get it running using README
- [x] Screenshots added (4 screenshots included)
- [x] Repository looks professional and polished

**Implementation Note**: This is the final phase. Ensure everything works end-to-end before considering the project complete.

---

## Testing Strategy

### Manual Testing Checklist:

**Happy Path:**
1. [ ] Upload a valid architectural image
2. [ ] Enter question "What architectural style is this?"
3. [ ] Click Analyze
4. [ ] Verify response discusses architectural features
5. [ ] Click Reset
6. [ ] Verify everything clears

**Error Handling:**
1. [ ] Try to analyze without uploading image - should show error
2. [ ] Try to analyze without entering question - should show error
3. [ ] Upload non-image file - should reject
4. [ ] Test with very large image (>8000px) - should resize automatically
5. [ ] Stop backend and try to analyze - should show connection error

**UI/UX:**
1. [ ] Test on Chrome, Firefox, Safari
2. [ ] Test responsive design (resize browser window)
3. [ ] Verify loading state shows during API call
4. [ ] Verify error messages are clear and helpful

## Performance Considerations

**Image Optimization:**
- Automatic resizing to 1.15 megapixels recommended by Claude
- Maintains aspect ratio during resize
- Uses efficient LANCZOS resampling

**API Efficiency:**
- Single API call per analysis (no unnecessary requests)
- Proper error handling to avoid retries
- CORS configured for local development only

## Time Estimates

Given **interview tomorrow afternoon**, here's a realistic timeline:

| Phase | Time | Priority |
|-------|------|----------|
| Phase 1: Backend | 1.5 hours | **CRITICAL** |
| Phase 2: Frontend | 1.5 hours | **CRITICAL** |
| Phase 3: Documentation | 1 hour | Important |
| **Total** | **~4 hours** | |

**Recommendation**: Start immediately, focus on Phases 1 & 2. Phase 3 can be done in the morning before interview if needed.

## References

### Research Sources:

**Claude Vision API:**
- [Vision - Claude Docs](https://docs.claude.com/en/docs/build-with-claude/vision)
- [Claude API Integration Guide 2025](https://collabnix.com/claude-api-integration-guide-2025-complete-developer-tutorial-with-code-examples/)

**LangChain Integration:**
- [ChatAnthropic - LangChain Docs](https://docs.langchain.com/oss/python/integrations/chat/anthropic)
- [langchain-anthropic PyPI](https://pypi.org/project/langchain-anthropic/)

**React Patterns:**
- [React Image Upload with Preview](https://www.bezkoder.com/react-image-upload-preview-hooks/)
- [How to Upload and Preview Images in ReactJS](https://blog.greenroots.info/how-to-upload-and-preview-images-in-reactjs)

**Flask CORS:**
- [Flask-CORS Documentation](https://flask-cors.readthedocs.io/en/latest/)
- [Creating Image Processing API with Flask](https://transloadit.com/devtips/creating-a-simple-image-processing-api-with-python-and-flask/)

---

## Final Checklist Before Interview

- [ ] Project runs successfully on your machine
- [ ] Can demonstrate full workflow (upload → analyze → reset)
- [ ] README is complete and accurate
- [ ] Code is pushed to GitHub with meaningful commits
- [ ] Prepare to discuss:
  - Why you chose this tech stack
  - How the prompt engineering works
  - Challenges you encountered
  - How you'd extend this for production
  - Integration with LangChain and Claude API

**Good luck with your interview! 🚀**