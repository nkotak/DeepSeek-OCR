// Prompt templates
const PROMPTS = {
    free: "<image>\\nFree OCR.",
    markdown: "<image>\\n<|grounding|>Convert the document to markdown.",
    grounding: "<image>\\n<|grounding|>OCR this image.",
    figure: "<image>\\nParse the figure.",
    description: "<image>\\nDescribe this image in detail.",
    custom: ""
};

// State
let uploadedFile = null;
let uploadedImage = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkBackendConnection();
});

function initializeEventListeners() {
    // File upload
    const fileUpload = document.getElementById('file-upload');
    const uploadBox = document.getElementById('upload-box');

    fileUpload.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('drag-over');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('drag-over');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Prompt template selection
    const promptTemplate = document.getElementById('prompt-template');
    promptTemplate.addEventListener('change', handlePromptTemplateChange);

    // Custom prompt
    const customPrompt = document.getElementById('custom-prompt');
    customPrompt.addEventListener('input', updatePromptPreview);

    // Sliders
    const maxTokens = document.getElementById('max-tokens');
    const temperature = document.getElementById('temperature');
    const topP = document.getElementById('top-p');

    maxTokens.addEventListener('input', (e) => {
        document.getElementById('max-tokens-value').textContent = e.target.value;
    });

    temperature.addEventListener('input', (e) => {
        document.getElementById('temperature-value').textContent = parseFloat(e.target.value).toFixed(1);
    });

    topP.addEventListener('input', (e) => {
        document.getElementById('top-p-value').textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Process button
    const processBtn = document.getElementById('process-btn');
    processBtn.addEventListener('click', processImage);

    // Copy and download buttons
    const copyBtn = document.getElementById('copy-btn');
    const downloadBtn = document.getElementById('download-btn');

    copyBtn.addEventListener('click', copyOutput);
    downloadBtn.addEventListener('click', downloadOutput);
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'application/pdf'];
    if (!validTypes.includes(file.type)) {
        showStatus('error', 'Invalid file type. Please upload an image or PDF.');
        return;
    }

    uploadedFile = file;

    // Preview image
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage = e.target.result;
            displayImagePreview(e.target.result, file);
        };
        reader.readAsDataURL(file);
    } else if (file.type === 'application/pdf') {
        showStatus('info', '📄 PDF uploaded. Backend will convert to images.');
        displayPDFInfo(file);
    }

    // Enable process button
    document.getElementById('process-btn').disabled = false;
}

function displayImagePreview(dataUrl, file) {
    const preview = document.getElementById('image-preview');
    const img = document.getElementById('preview-img');
    const info = document.getElementById('image-info');

    img.src = dataUrl;
    preview.style.display = 'block';

    // Get image dimensions
    const tempImg = new Image();
    tempImg.onload = () => {
        info.innerHTML = `
            <strong>Image Info:</strong><br>
            Size: ${tempImg.width} × ${tempImg.height} pixels<br>
            Type: ${file.type}<br>
            File Size: ${formatFileSize(file.size)}
        `;
    };
    tempImg.src = dataUrl;
}

function displayPDFInfo(file) {
    const preview = document.getElementById('image-preview');
    const info = document.getElementById('image-info');

    preview.style.display = 'block';
    document.getElementById('preview-img').style.display = 'none';

    info.innerHTML = `
        <strong>PDF Info:</strong><br>
        Name: ${file.name}<br>
        File Size: ${formatFileSize(file.size)}<br>
        <em>PDF will be converted to images on the backend</em>
    `;
}

function handlePromptTemplateChange(e) {
    const template = e.target.value;
    const customGroup = document.getElementById('custom-prompt-group');

    if (template === 'custom') {
        customGroup.style.display = 'block';
        updatePromptPreview();
    } else {
        customGroup.style.display = 'none';
        updatePromptPreview(PROMPTS[template]);
    }
}

function updatePromptPreview(prompt = null) {
    const preview = document.getElementById('prompt-preview');
    const template = document.getElementById('prompt-template').value;

    let promptText;
    if (prompt) {
        promptText = prompt;
    } else if (template === 'custom') {
        promptText = document.getElementById('custom-prompt').value || '<image>\\nFree OCR.';
    } else {
        promptText = PROMPTS[template];
    }

    preview.innerHTML = `<strong>Prompt:</strong> <code>${escapeHtml(promptText)}</code>`;
}

async function processImage() {
    if (!uploadedFile) {
        showStatus('error', 'No file uploaded');
        return;
    }

    // Get configuration
    const template = document.getElementById('prompt-template').value;
    const prompt = template === 'custom'
        ? document.getElementById('custom-prompt').value
        : PROMPTS[template];
    const maxTokens = parseInt(document.getElementById('max-tokens').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const topP = parseFloat(document.getElementById('top-p').value);
    const cropping = document.getElementById('cropping').checked;

    // Prepare form data
    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('prompt', prompt);
    formData.append('max_tokens', maxTokens);
    formData.append('temperature', temperature);
    formData.append('top_p', topP);
    formData.append('cropping', cropping);

    // Show processing status
    showStatus('processing', '⏳ Processing image... This may take a moment.');
    document.getElementById('process-btn').disabled = true;

    try {
        const response = await fetch('http://localhost:8000/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();

        if (result.error) {
            showStatus('error', `Error: ${result.error}`);
        } else {
            displayOutput(result);
        }
    } catch (error) {
        showStatus('error', `Failed to process image: ${error.message}<br><br>Make sure the backend server is running: <code>python server.py</code>`);
    } finally {
        document.getElementById('process-btn').disabled = false;
    }
}

function displayOutput(result) {
    // Hide examples
    document.getElementById('examples-section').style.display = 'none';
    document.getElementById('connection-status').style.display = 'none';

    // Show output
    document.getElementById('output-section').style.display = 'block';

    // Update content
    document.getElementById('output-text').value = result.text || 'No output generated';
    document.getElementById('token-count').textContent = `${result.num_tokens || 0} tokens generated`;

    // Show success status
    showStatus('success', '✅ Processing complete!');
}

function showStatus(type, message) {
    const status = document.getElementById('status');
    const content = document.getElementById('status-content');

    status.className = `status-box ${type}`;
    content.innerHTML = message;
    status.style.display = 'block';

    // Auto-hide info/success messages after 5 seconds
    if (type === 'info' || type === 'success') {
        setTimeout(() => {
            status.style.display = 'none';
        }, 5000);
    }
}

function copyOutput() {
    const output = document.getElementById('output-text');
    output.select();
    document.execCommand('copy');

    // Visual feedback
    const btn = document.getElementById('copy-btn');
    const originalText = btn.textContent;
    btn.textContent = '✅ Copied!';
    setTimeout(() => {
        btn.textContent = originalText;
    }, 2000);
}

function downloadOutput() {
    const output = document.getElementById('output-text').value;
    const blob = new Blob([output], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ocr_output.txt';
    a.click();
    URL.revokeObjectURL(url);
}

async function checkBackendConnection() {
    try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
            const status = document.getElementById('connection-status');
            status.className = 'info-box success-box';
            status.innerHTML = '<strong>✅ Backend Connected</strong><br>Ready to process images.';
        }
    } catch (error) {
        // Backend not available - show warning (already in HTML)
    }
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
