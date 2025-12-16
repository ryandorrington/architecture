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
