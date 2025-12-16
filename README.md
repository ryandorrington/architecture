# 🏛️ Architecture Analyzer

AI-powered architectural image analysis using Claude API & LangChain. Upload architectural images and ask questions about styles, elements, and design features.

## Screenshots

![Upload Interface](Screenshot%20from%202025-12-16%2010-10-59.png)
![Building Information](Screenshot%20from%202025-12-16%2010-11-35.png)
![Detailed Analysis](Screenshot%20from%202025-12-16%2010-13-00.png)
![Markdown Rendering](Screenshot%20from%202025-12-16%2010-25-37.png)

## Features

- 📷 **Image Upload** - Upload architectural images (PNG, JPG, WEBP)
- 🤖 **AI Analysis** - Powered by Claude Sonnet 4.5 with vision capabilities
- 🏗️ **Architecture-Focused** - Custom prompt engineering for architectural insights
- 📊 **Structured Metadata** - Extracts style, period, materials, and key features
- ⚡ **Real-time** - Fast analysis and response
- 🎨 **Clean UI** - Modern, responsive React interface
- 📝 **Markdown Support** - Rich formatting for detailed analysis

## Tech Stack

**Frontend:**
- React (Modern hooks-based approach)
- ReactMarkdown (Rich text rendering)
- Pure CSS with gradient design
- Fetch API for backend communication

**Backend:**
- Flask (Python web framework)
- LangChain + Anthropic SDK (AI integration)
- Claude Sonnet 4.5 (Vision-enabled model)
- Flask-CORS (Cross-origin support)
- Pydantic (Structured data validation)

## Prerequisites

- Python 3.9+
- Node.js 16+
- Anthropic API key ([Get one here](https://console.anthropic.com/))

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd architecture
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
2. Upload an architectural image (good dataset: [Kaggle Architecture Dataset](https://www.kaggle.com/datasets/wwymak/architecture-dataset))
3. Ask a question (e.g., "What architectural style is this?")
4. Click "Analyze" to get AI-powered insights
5. Review the structured metadata and detailed analysis
6. Click "Reset" to try another image

## Example Questions

- "What architectural style is this building?"
- "Describe the key structural elements visible in this image"
- "What materials appear to be used in this construction?"
- "What design principles are evident in this architecture?"
- "What historical period does this architecture represent?"
- "How does this building reflect modernist principles?"

## Project Structure

```
architecture/
├── backend/
│   ├── app.py              # Flask application with LangChain integration
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
Analyze architectural image with question using two-step LangChain process.

**Request:**
- `image` (file): Image file (PNG, JPG, JPEG, WEBP)
- `question` (string): Question about the architecture

**Response:**
```json
{
  "success": true,
  "metadata": {
    "style": "Modernist",
    "period": "Late 20th Century",
    "materials": ["concrete", "glass", "steel"],
    "key_features": ["cantilever design", "large glass panels", "geometric forms"]
  },
  "analysis": "Detailed architectural analysis in markdown format...",
  "model": "claude-sonnet-4-5-20250929"
}
```

## Implementation Notes

### Two-Step LangChain Architecture

The application uses an innovative two-step approach:

1. **Metadata Extraction** - Uses LangChain's `with_structured_output()` to extract structured architectural metadata (style, period, materials, key features) as a Pydantic model
2. **Detailed Analysis** - Feeds the structured metadata as context to generate a comprehensive architectural analysis addressing the user's question

This demonstrates:
- Structured output generation with Pydantic
- Multi-step prompting patterns
- Context building from extracted data
- Proper separation of concerns

### Prompt Engineering

The system uses architecture-focused prompts that guide Claude to:
- Identify architectural styles and periods
- Analyze structural elements
- Evaluate design principles
- Assess materials and construction
- Provide historical context

### Image Processing

- Images are automatically resized to stay under Claude's 1.15 megapixel recommendation
- Supports multiple formats (PNG, JPG, JPEG, WEBP)
- Maximum upload size: 16 MB
- Maintains aspect ratio during resize using LANCZOS resampling

## License

MIT

## Author

Built as a portfolio project to demonstrate AI/ML integration skills for AI/ML engineering positions.

**Key Demonstrations:**
- React development with modern hooks
- Flask API development
- Claude Vision API integration
- LangChain structured outputs
- Pydantic data modeling
- Multi-step prompting strategies
- Clean, production-ready code architecture