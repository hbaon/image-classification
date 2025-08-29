/**
 * AI Image Classification System - Frontend JavaScript
 * Handles file upload, drag & drop, and API communication
 */

class ImageClassifier {
    constructor() {
        this.currentFile = null;
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File input change
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Drag and drop events
        const dropZone = document.getElementById('drop-zone');
        
        dropZone.addEventListener('click', () => {
            document.getElementById('file-input').click();
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        // Keyboard navigation
        dropZone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                document.getElementById('file-input').click();
            }
        });

        // Make drop zone focusable
        dropZone.setAttribute('tabindex', '0');
        dropZone.setAttribute('role', 'button');
        dropZone.setAttribute('aria-label', 'Click or drag and drop image to upload');
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        if (!this.isValidImageFile(file)) {
            this.showError('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP, TIFF)');
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('File size must be less than 16MB');
            return;
        }

        this.currentFile = file;
        this.displayFileInfo(file);
        this.showClassifyButton();
        this.hideResults();
    }

    isValidImageFile(file) {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff'];
        return validTypes.includes(file.type);
    }

    displayFileInfo(file) {
        const fileInfo = document.getElementById('file-info');
        const fileName = document.getElementById('file-name');
        const fileSize = document.getElementById('file-size');

        fileName.textContent = file.name;
        fileSize.textContent = this.formatFileSize(file.size);
        
        fileInfo.classList.remove('d-none');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showClassifyButton() {
        document.getElementById('classify-btn').classList.remove('d-none');
    }

    hideResults() {
        document.getElementById('results-section').classList.add('d-none');
        document.getElementById('results-display').classList.add('d-none');
    }

    clearFile() {
        this.currentFile = null;
        document.getElementById('file-info').classList.add('d-none');
        document.getElementById('classify-btn').classList.add('d-none');
        document.getElementById('results-section').classList.add('d-none');
        document.getElementById('file-input').value = '';
    }

    async classifyImage() {
        if (!this.currentFile) return;

        try {
            // Show processing state
            this.showProcessing();
            
            // Create form data
            const formData = new FormData();
            formData.append('image', this.currentFile);

            // Make API request
            const response = await fetch('/classify', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result);
            } else {
                this.showError(result.error || 'Classification failed');
            }

        } catch (error) {
            console.error('Error:', error);
            this.showError('An error occurred during classification. Please try again.');
        } finally {
            this.hideProcessing();
        }
    }

    showProcessing() {
        document.getElementById('processing').classList.remove('d-none');
        document.getElementById('results-section').classList.remove('d-none');
        document.getElementById('results-display').classList.add('d-none');
    }

    hideProcessing() {
        document.getElementById('processing').classList.add('d-none');
    }

    displayResults(result) {
        // Show results section
        document.getElementById('results-section').classList.remove('d-none');
        document.getElementById('results-display').classList.remove('d-none');

        // Display image preview
        this.displayImagePreview();

        // Display predictions
        this.displayPredictions(result.predictions);

        // Display processing time
        document.getElementById('processing-time').textContent = result.processing_time;
    }

    displayImagePreview() {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('preview-image').src = e.target.result;
        };
        reader.readAsDataURL(this.currentFile);
    }

    displayPredictions(predictions) {
        const predictionsList = document.getElementById('predictions-list');
        predictionsList.innerHTML = '';

        predictions.forEach((prediction, index) => {
            const predictionItem = document.createElement('div');
            predictionItem.className = `prediction-item top-${index + 1}`;
            
            const confidencePercentage = prediction.probability;
            const confidenceBar = `
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidencePercentage}%"></div>
                </div>
            `;

            predictionItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h6 class="mb-1">${prediction.class}</h6>
                        <small class="text-muted">Confidence: ${confidencePercentage}%</small>
                    </div>
                    <div class="text-end">
                        <span class="badge bg-primary">${(index + 1)}</span>
                    </div>
                </div>
                ${confidenceBar}
            `;

            predictionsList.appendChild(predictionItem);
        });

        // Animate confidence bars
        setTimeout(() => {
            document.querySelectorAll('.confidence-fill').forEach(bar => {
                bar.style.width = bar.style.width;
            });
        }, 100);
    }

    showError(message) {
        // Create and show error toast
        const toast = document.createElement('div');
        toast.className = 'alert alert-danger alert-dismissible fade show position-fixed';
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        toast.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }

    // Utility method to check if element is in viewport
    isInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }

    // Smooth scroll to element
    scrollToElement(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.imageClassifier = new ImageClassifier();
    
    // Add some nice animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe cards for animation
    document.querySelectorAll('.card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });

    // Add loading animation to features
    document.querySelectorAll('.feature-icon').forEach((icon, index) => {
        icon.style.animationDelay = `${index * 0.2}s`;
        icon.style.animation = 'fadeIn 0.6s ease forwards';
    });
});

// Global functions for HTML onclick handlers
function classifyImage() {
    if (window.imageClassifier) {
        window.imageClassifier.classifyImage();
    }
}

function clearFile() {
    if (window.imageClassifier) {
        window.imageClassifier.clearFile();
    }
}

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to classify
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (window.imageClassifier && window.imageClassifier.currentFile) {
            classifyImage();
        }
    }
    
    // Escape to clear
    if (e.key === 'Escape') {
        if (window.imageClassifier) {
            clearFile();
        }
    }
});

// Service Worker registration for PWA capabilities
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
