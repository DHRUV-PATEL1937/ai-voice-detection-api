// Configuration
const API_KEY = 'sk_test_123456789';
const API_URL = 'http://localhost:8000/api/voice-detection';

// Global variables
let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('audioFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    fileInput.addEventListener('change', handleFileSelect);
    analyzeBtn.addEventListener('click', analyzeAudio);
});

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) return;
    
    // Validate file type
    if (!file.type.includes('audio/mpeg') && !file.name.endsWith('.mp3')) {
        showError('Please select an MP3 file');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    selectedFile = file;
    
    // Display file name
    document.getElementById('fileName').innerHTML = `
        <strong>Selected:</strong> ${file.name} (${formatFileSize(file.size)})
    `;
    
    // Enable analyze button
    document.getElementById('analyzeBtn').disabled = false;
}

// Analyze audio
async function analyzeAudio() {
    if (!selectedFile) return;
    
    // Show loading
    showSection('loadingSection');
    
    try {
        // Convert file to base64
        const base64Audio = await fileToBase64(selectedFile);
        
        // Get selected language
        const language = document.getElementById('language').value;
        
        // Make API request
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': API_KEY
            },
            body: JSON.stringify({
                language: language,
                audioFormat: 'mp3',
                audioBase64: base64Audio
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'API request failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred during analysis');
    }
}

// Display results
function displayResults(result) {
    // Set classification
    const isAI = result.classification === 'AI_GENERATED';
    
    document.getElementById('resultIcon').textContent = isAI ? '🤖' : '👤';
    document.getElementById('resultTitle').textContent = isAI ? 'AI Generated Voice' : 'Human Voice';
    document.getElementById('resultTitle').style.color = isAI ? '#dc3545' : '#28a745';
    
    // Set confidence
    const confidence = Math.round(result.confidenceScore * 100);
    document.getElementById('confidenceScore').textContent = `${confidence}%`;
    document.getElementById('confidenceFill').style.width = `${confidence}%`;
    
    // Set details
    document.getElementById('resultLanguage').textContent = result.language;
    document.getElementById('resultClassification').textContent = result.classification.replace('_', ' ');
    document.getElementById('resultExplanation').textContent = result.explanation;
    
    // Set audio preview
    const audioPlayer = document.getElementById('audioPlayer');
    audioPlayer.src = URL.createObjectURL(selectedFile);
    
    // Show results
    showSection('resultsSection');
}

// Show error
function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    showSection('errorSection');
}

// Reset analysis
function resetAnalysis() {
    selectedFile = null;
    document.getElementById('audioFile').value = '';
    document.getElementById('fileName').innerHTML = '';
    document.getElementById('analyzeBtn').disabled = true;
    showSection('uploadSection');
}

// Show specific section
function showSection(sectionId) {
    const sections = ['uploadSection', 'loadingSection', 'resultsSection', 'errorSection'];
    
    sections.forEach(id => {
        document.getElementById(id).classList.add('hidden');
    });
    
    document.getElementById(sectionId).classList.remove('hidden');
}

// Convert file to base64
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = () => {
            // Remove data URL prefix
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Format file size
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}