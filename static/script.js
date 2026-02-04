// Configuration
// ✅ FIX: Set this to JUST the domain (No /api/voice-detection at the end)
const API_URL = "https://ai-voice-detection-api-8w9j.onrender.com";

// Enhanced logging
function log(emoji, message, data = null) {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
    console.log(`[${timestamp}] ${emoji} ${message}`);
    if (data) console.log(data);
}

log('🔧', 'Script loaded');
log('🌐', 'API URL:', API_URL);
log('📍', 'Current location:', window.location.href);

// Global variables
let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    log('✅', 'DOM Content Loaded');
    
    // Check if all required elements exist
    const elements = {
        audioFile: document.getElementById('audioFile'),
        analyzeBtn: document.getElementById('analyzeBtn'),
        language: document.getElementById('language'),
        uploadSection: document.getElementById('uploadSection'),
        loadingSection: document.getElementById('loadingSection'),
        resultsSection: document.getElementById('resultsSection'),
        errorSection: document.getElementById('errorSection')
    };
    
    log('🔍', 'Checking DOM elements...');
    let allElementsFound = true;
    
    for (const [name, element] of Object.entries(elements)) {
        if (!element) {
            log('❌', `Missing element: ${name}`);
            allElementsFound = false;
        } else {
            log('✅', `Found element: ${name}`);
        }
    }
    
    if (!allElementsFound) {
        log('❌', 'Some elements are missing! Check HTML structure.');
        return;
    }
    
    // Attach event listeners
    if (elements.audioFile) elements.audioFile.addEventListener('change', handleFileSelect);
    if (elements.analyzeBtn) elements.analyzeBtn.addEventListener('click', analyzeAudio);
    
    log('✅', 'Event listeners attached');
    
    // Test API connectivity
    testAPIConnection();
});

// Test API connection
async function testAPIConnection() {
    try {
        log('🔌', 'Testing API connection...');
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        log('✅', 'API is reachable:', data);
    } catch (error) {
        log('❌', 'API connection failed:', error.message);
        log('⚠️', 'This might cause issues with voice detection');
    }
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) {
        log('⚠️', 'No file selected');
        return;
    }
    
    log('📁', 'File selected:', {
        name: file.name,
        type: file.type,
        size: file.size
    });
    
    // Validate file type
    if (!file.type.includes('audio') && !file.name.endsWith('.mp3') && !file.name.endsWith('.wav')) {
        log('❌', 'Invalid file type:', file.type);
        showError('Please select a valid audio file (MP3/WAV)');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        log('❌', 'File too large:', file.size);
        showError('File size must be less than 10MB');
        return;
    }
    
    selectedFile = file;
    
    // Display file name
    const fileNameElement = document.getElementById('fileName');
    if (fileNameElement) {
        fileNameElement.innerHTML = `<strong>Selected:</strong> ${file.name} (${formatFileSize(file.size)})`;
    }
    
    // Enable analyze button
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        log('✅', 'Analyze button enabled');
    }

    // Clear previous results/errors
    hideSection('resultsSection');
    hideSection('errorSection');
}

// Analyze audio
async function analyzeAudio() {
    if (!selectedFile) {
        log('❌', 'No file selected');
        return;
    }
    
    log('🔍', 'Starting analysis...');
    
    // Show loading
    showSection('loadingSection');
    
    try {
        // Convert file to base64
        log('📦', 'Converting to base64...');
        const base64Audio = await fileToBase64(selectedFile);
        log('✅', 'Base64 conversion complete, length:', base64Audio.length);
        
        // Get selected language
        const languageElement = document.getElementById('language');
        const language = languageElement ? languageElement.value : "english";
        log('🌐', 'Language:', language);
        
        const requestData = {
            audioBase64: base64Audio, // Was audioData
            language: language,
            audioFormat: "mp3"        // Added this required field
            // userId removed
        };
        
        // Construct the full API endpoint
        const apiEndpoint = `${API_URL}/api/voice-detection`;
        log('📤', 'Sending request to:', apiEndpoint);
        
        // Make API request
        const response = await fetch(apiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': 'sk_test_123456789' // Ensure this matches your backend settings
            },
            body: JSON.stringify(requestData)
        });
        
        log('📥', 'Response received, status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            log('❌', 'API Error Response:', errorText);
            
            let errorMessage;
            try {
                const error = JSON.parse(errorText);
                errorMessage = error.detail || error.message || 'API request failed';
            } catch (e) {
                errorMessage = errorText || 'API request failed';
            }
            
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        log('✅', 'Result received:', result);
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        log('❌', 'Error during analysis:', error);
        console.error('Full error:', error);
        showError(error.message || 'An error occurred during analysis');
        // If error occurs, show upload section again so user can retry
        // But keep error section visible
        const uploadSection = document.getElementById('uploadSection');
        if(uploadSection) uploadSection.classList.remove('hidden');
        const loadingSection = document.getElementById('loadingSection');
        if(loadingSection) loadingSection.classList.add('hidden');
    }
}

// Display results
function displayResults(result) {
    log('📊', 'Displaying results');
    
    try {
        // Set classification
        const isAI = result.classification === 'AI_GENERATED';
        
        const resultIcon = document.getElementById('resultIcon');
        const resultTitle = document.getElementById('resultTitle');
        
        if (resultIcon) resultIcon.textContent = isAI ? '🤖' : '👤';
        if (resultTitle) {
            resultTitle.textContent = isAI ? 'AI Generated Voice' : 'Human Voice';
            resultTitle.style.color = isAI ? '#dc3545' : '#28a745';
        }
        
        // Set confidence
        const confidence = Math.round(result.confidenceScore * 100);
        log('📈', 'Confidence:', confidence + '%');
        
        const confidenceScore = document.getElementById('confidenceScore');
        const confidenceFill = document.getElementById('confidenceFill');
        
        if (confidenceScore) confidenceScore.textContent = `${confidence}%`;
        if (confidenceFill) {
            confidenceFill.style.width = `${confidence}%`;
            
            // Color coding
            if (confidence >= 80) {
                confidenceFill.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
            } else if (confidence >= 60) {
                confidenceFill.style.background = 'linear-gradient(135deg, #ffc107 0%, #ffb300 100%)';
            } else {
                confidenceFill.style.background = 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)';
            }
        }
        
        // Set details
        const resultLanguage = document.getElementById('resultLanguage');
        const resultClassification = document.getElementById('resultClassification');
        const resultExplanation = document.getElementById('resultExplanation');
        
        if (resultLanguage) resultLanguage.textContent = result.language;
        if (resultClassification) resultClassification.textContent = result.classification.replace('_', ' ');
        if (resultExplanation) resultExplanation.textContent = result.explanation;
        
        // Set audio preview
        const audioPlayer = document.getElementById('audioPlayer');
        if (audioPlayer && selectedFile) {
            audioPlayer.src = URL.createObjectURL(selectedFile);
        }
        
        // Show results
        showSection('resultsSection');
        log('✅', 'Results displayed successfully');
        
    } catch (error) {
        log('❌', 'Error displaying results:', error);
        showError('Error displaying results: ' + error.message);
    }
}

// Show error
function showError(message) {
    log('⚠️', 'Showing error:', message);
    
    const errorMessage = document.getElementById('errorMessage');
    if (errorMessage) {
        errorMessage.textContent = message;
    }
    
    // Directly manipulating class here to ensure error shows on top of upload
    const errorSection = document.getElementById('errorSection');
    if (errorSection) errorSection.classList.remove('hidden');
    
    // Hide loading if it was showing
    const loadingSection = document.getElementById('loadingSection');
    if (loadingSection) loadingSection.classList.add('hidden');
}

// Reset analysis
function resetAnalysis() {
    log('🔄', 'Resetting analysis');
    
    selectedFile = null;
    
    const audioFile = document.getElementById('audioFile');
    const fileName = document.getElementById('fileName');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (audioFile) audioFile.value = '';
    if (fileName) fileName.textContent = ''; // Changed from innerHTML to textContent for safety
    if (analyzeBtn) analyzeBtn.disabled = true;
    
    showSection('uploadSection');
}

// Helper functions to toggle sections
function showSection(sectionId) {
    log('👁️', 'Showing section:', sectionId);
    
    const sections = ['uploadSection', 'loadingSection', 'resultsSection', 'errorSection'];
    
    sections.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.classList.add('hidden');
        }
    });
    
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.remove('hidden');
    } else {
        log('❌', 'Section not found:', sectionId);
    }
}

function hideSection(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
}

// Convert file to base64
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = () => {
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        
        reader.onerror = (error) => {
            log('❌', 'File read error:', error);
            reject(error);
        };
        
        reader.readAsDataURL(file);
    });
}

// Format file size
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Expose reset function to global scope
window.resetAnalysis = resetAnalysis;

// Log when script finishes loading
log('✅', 'Script fully loaded and ready');