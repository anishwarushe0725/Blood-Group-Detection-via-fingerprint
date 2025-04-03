// Common Functions
document.addEventListener('DOMContentLoaded', function() {
    // Initialize any components that need it
    initializeComponents();
    
    // Add event listeners
    setupEventListeners();
});

function initializeComponents() {
    // Initialize date pickers if available
    const datePickers = document.querySelectorAll('.date-picker');
    if (datePickers.length > 0) {
        datePickers.forEach(element => {
            element.valueAsDate = new Date();
        });
    }

    // Initialize tooltips
    const tooltips = document.querySelectorAll('[data-toggle="tooltip"]');
    if (tooltips.length > 0) {
        tooltips.forEach(tooltip => {
            tooltip.addEventListener('mouseenter', showTooltip);
            tooltip.addEventListener('mouseleave', hideTooltip);
        });
    }
}

function setupEventListeners() {
    // Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', validateForm);
    });

    // Fingerprint upload area
    const fingerprintUpload = document.getElementById('fingerprint-upload-area');
    if (fingerprintUpload) {
        setupFingerprintUpload();
    }

    // Print button for reports
    const printButton = document.getElementById('print-report');
    if (printButton) {
        printButton.addEventListener('click', printReport);
    }

    // Password toggle visibility
    const passwordToggles = document.querySelectorAll('.password-toggle');
    passwordToggles.forEach(toggle => {
        toggle.addEventListener('click', togglePasswordVisibility);
    });
}

// Form Validation
function validateForm(event) {
    const form = event.target;
    let isValid = true;
    
    // Get all required inputs
    const requiredInputs = form.querySelectorAll('[required]');
    
    // Check each required input
    requiredInputs.forEach(input => {
        if (!input.value.trim()) {
            isValid = false;
            showError(input, 'This field is required');
        } else {
            clearError(input);
            
            // Validate specific input types
            if (input.type === 'email' && !validateEmail(input.value)) {
                isValid = false;
                showError(input, 'Please enter a valid email address');
            } else if (input.classList.contains('phone-input') && !validatePhone(input.value)) {
                isValid = false;
                showError(input, 'Please enter a valid phone number');
            }
        }
    });
    
    // Check password confirmation if exists
    const password = form.querySelector('#password');
    const confirmPassword = form.querySelector('#confirm-password');
    if (password && confirmPassword && password.value !== confirmPassword.value) {
        isValid = false;
        showError(confirmPassword, 'Passwords do not match');
    }
    
    if (!isValid) {
        event.preventDefault();
    }
    
    return isValid;
}

function showError(input, message) {
    // Get the parent form group
    const formGroup = input.closest('.form-group');
    const errorElement = formGroup.querySelector('.error-message') || createErrorElement(formGroup);
    
    // Add error class to input
    input.classList.add('is-invalid');
    
    // Set error message
    errorElement.textContent = message;
    errorElement.style.display = 'block';
}

function clearError(input) {
    // Get the parent form group
    const formGroup = input.closest('.form-group');
    const errorElement = formGroup.querySelector('.error-message');
    
    // Remove error class from input
    input.classList.remove('is-invalid');
    
    // Hide error message if it exists
    if (errorElement) {
        errorElement.style.display = 'none';
    }
}

function createErrorElement(formGroup) {
    const errorElement = document.createElement('div');
    errorElement.className = 'error-message';
    errorElement.style.color = 'var(--error-color)';
    errorElement.style.fontSize = '0.85rem';
    errorElement.style.marginTop = '0.25rem';
    formGroup.appendChild(errorElement);
    return errorElement;
}

function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(String(email).toLowerCase());
}

function validatePhone(phone) {
    const re = /^\+?[\d\s-]{10,15}$/;
    return re.test(String(phone));
}

// Fingerprint Upload
function setupFingerprintUpload() {
    const uploadArea = document.getElementById('fingerprint-upload-area');
    const fileInput = document.getElementById('fingerprint-input');
    const previewElement = document.getElementById('fingerprint-preview');
    const uploadText = document.getElementById('fingerprint-upload-text');

    if (!uploadArea || !fileInput) return;

    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        // Check if file is an image
        if (!file.type.match('image.*')) {
            showAlert('Please select an image file for the fingerprint', 'error');
            return;
        }

        // Check if file size is less than 5MB
        if (file.size > 5 * 1024 * 1024) {
            showAlert('File size should be less than 5MB', 'error');
            return;
        }

        // Create preview
        const reader = new FileReader();
        reader.onload = function(e) {
            if (previewElement) {
                previewElement.src = e.target.result;
                previewElement.style.display = 'block';
            }
            if (uploadText) {
                uploadText.style.display = 'none';
            }
        };
        reader.readAsDataURL(file);
    }
}

// Utility Functions
function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alert-container') || createAlertContainer();
    
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type}`;
    alertElement.textContent = message;
    
    // Add close button
    const closeButton = document.createElement('button');
    closeButton.className = 'close-alert';
    closeButton.innerHTML = '&times;';
    closeButton.style.float = 'right';
    closeButton.style.cursor = 'pointer';
    closeButton.style.border = 'none';
    closeButton.style.background = 'none';
    closeButton.style.fontSize = '1.5rem';
    closeButton.style.marginLeft = '10px';
    closeButton.addEventListener('click', () => {
        alertContainer.removeChild(alertElement);
    });
    
    alertElement.prepend(closeButton);
    alertContainer.appendChild(alertElement);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alertContainer.contains(alertElement)) {
            alertContainer.removeChild(alertElement);
        }
    }, 5000);
}

function createAlertContainer() {
    const alertContainer = document.createElement('div');
    alertContainer.id = 'alert-container';
    alertContainer.style.position = 'fixed';
    alertContainer.style.top = '20px';
    alertContainer.style.right = '20px';
    alertContainer.style.zIndex = '1000';
    alertContainer.style.width = '300px';
    document.body.appendChild(alertContainer);
    return alertContainer;
}

function togglePasswordVisibility(event) {
    const button = event.target.closest('.password-toggle');
    const passwordInput = button.previousElementSibling;
    
    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        button.innerHTML = '<i class="fas fa-eye-slash"></i>';
    } else {
        passwordInput.type = 'password';
        button.innerHTML = '<i class="fas fa-eye"></i>';
    }
}

function printReport() {
    window.print();
}

// Dashboard specific functions
function initializeDashboard() {
    // Fetch recent patients data if on dashboard
    if (document.querySelector('.dashboard-container')) {
        fetchRecentPatients();
    }
}

function fetchRecentPatients() {
    // This would normally be an AJAX call to the server
    // For now, we'll just update the UI with mock data
    console.log('Fetching recent patients...');
}

// Result Page Animation
function animateResult() {
    const resultElement = document.querySelector('.blood-group');
    if (resultElement) {
        resultElement.classList.add('animate-pulse');
        setTimeout(() => {
            resultElement.classList.remove('animate-pulse');
        }, 2000);
    }
}

// Tooltip Functions
function showTooltip(event) {
    const tooltip = event.target;
    const tooltipText = tooltip.getAttribute('data-tooltip');
    
    const tooltipElement = document.createElement('div');
    tooltipElement.className = 'tooltip-text';
    tooltipElement.textContent = tooltipText;
    tooltipElement.style.position = 'absolute';
    tooltipElement.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
    tooltipElement.style.color = 'white';
    tooltipElement.style.padding = '5px 10px';
    tooltipElement.style.borderRadius = '5px';
    tooltipElement.style.zIndex = '100';
    
    tooltip.appendChild(tooltipElement);
    
    // Position the tooltip
    const rect = tooltip.getBoundingClientRect();
    tooltipElement.style.top = rect.bottom + 5 + 'px';
    tooltipElement.style.left = rect.left + (rect.width / 2) - (tooltipElement.offsetWidth / 2) + 'px';
}

function hideTooltip(event) {
    const tooltip = event.target;
    const tooltipText = tooltip.querySelector('.tooltip-text');
    if (tooltipText) {
        tooltip.removeChild(tooltipText);
    }
}

// Call initializeDashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeDashboard);
