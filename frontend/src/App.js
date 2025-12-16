import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
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
            <div className="analysis-text markdown-content">
              <ReactMarkdown>{analysis}</ReactMarkdown>
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
