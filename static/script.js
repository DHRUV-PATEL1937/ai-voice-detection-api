// Configuration
// Auto-detect environment - works for both localhost and Render
const API_URL = window.location.origin;

console.log('🔧 API URL:', API_URL);

// Global variables
let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ Page loaded');
    
    const fileInput = document.getElementById('audioFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (!fileInput || !analyzeBtn) {
        console.error('❌ Required elements not found');
        return;
    }
    
    fileInput.addEventListener('change', handleFileSelect);
    analyzeBtn.addEventListener('click', analyzeAudio);
    
    console.log('✅ Event listeners attached');
});

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) return;
    
    console.log('📁 File selected:', file.name, file.type, file.size);
    
    // Validate file type
    if (!file.type.includes('audio')) {
        showError('Please select an audio file');
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
    if (!selectedFile) {
        console.error('❌ No file selected');
        return;
    }
    
    console.log('🔍 Starting analysis...');
    
    // Show loading
    showSection('loadingSection');
    
    try {
        // Convert file to base64
        console.log('📦 Converting to base64...');
        const base64Audio = await fileToBase64(selectedFile);
        console.log('✅ Base64 conversion complete');
        
        // Get selected language
        const language = document.getElementById('language').value;
        console.log('🌐 Language:', language);
        
        const requestData = {
            audioData: base64Audio,
            language: language,
            userId: "test_user"
        };
        
        console.log('📤 Sending request to:', `${API_URL}/api/voice-detection`);
        
        // Make API request
        const response = await fetch(`${API_URL}/api/voice-detection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        console.log('📥 Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ API Error:', errorText);
            
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
        console.log('✅ Result:', result);
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('❌ Error:', error);
        showError(error.message || 'An error occurred during analysis');
    }
}

// Display results
function displayResults(result) {
    console.log('📊 Displaying results:', result);
    
    // Set classification
    const isAI = result.classification === 'AI_GENERATED';
    
    document.getElementById('resultIcon').textContent = isAI ? '🤖' : '👤';
    document.getElementById('resultTitle').textContent = isAI ? 'AI Generated Voice' : 'Human Voice';
    document.getElementById('resultTitle').style.color = isAI ? '#dc3545' : '#28a745';
    
    // Set confidence - FIXED: Ensure percentage is displayed
    const confidence = Math.round(result.confidenceScore * 100);
    console.log('📈 Confidence score:', result.confidenceScore, 'Percentage:', confidence);
    
    document.getElementById('confidenceScore').textContent = `${confidence}%`;
    document.getElementById('confidenceFill').style.width = `${confidence}%`;
    
    // Add color to confidence bar based on value
    const confidenceFill = document.getElementById('confidenceFill');
    if (confidence >= 80) {
        confidenceFill.style.backgroundColor = '#28a745'; // Green
    } else if (confidence >= 60) {
        confidenceFill.style.backgroundColor = '#ffc107'; // Yellow
    } else {
        confidenceFill.style.backgroundColor = '#dc3545'; // Red
    }
    
    // Set details
    document.getElementById('resultLanguage').textContent = result.language;
    document.getElementById('resultClassification').textContent = result.classification.replace('_', ' ');
    document.getElementById('resultExplanation').textContent = result.explanation;
    
    // Set audio preview
    const audioPlayer = document.getElementById('audioPlayer');
    audioPlayer.src = URL.createObjectURL(selectedFile);
    
    // Show results
    showSection('resultsSection');
    console.log('✅ Results displayed');
}

// Show error
function showError(message) {
    console.error('⚠️ Showing error:', message);
    document.getElementById('errorMessage').textContent = message;
    showSection('errorSection');
}

// Reset analysis
function resetAnalysis() {
    console.log('🔄 Resetting analysis');
    selectedFile = null;
    document.getElementById('audioFile').value = '';
    document.getElementById('fileName').innerHTML = '';
    document.getElementById('analyzeBtn').disabled = true;
    showSection('uploadSection');
}

// Show specific section
function showSection(sectionId) {
    console.log('👁️ Showing section:', sectionId);
    
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
        console.error('❌ Section not found:', sectionId);
    }
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
        
        reader.onerror = (error) => {
            console.error('❌ File read error:', error);
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