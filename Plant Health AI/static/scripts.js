document.addEventListener('DOMContentLoaded', () => {
    // Initialize AOS animation library
    AOS.init({
        duration: 800,
        easing: 'ease-out',
        once: false,
        mirror: false
    });
    
    // Animated counters
    animateCounters();
    
    // Plant selection functionality
    const plantOptions = document.querySelectorAll('.plant-option');
    const startPrompt = document.querySelector('.start-prompt');
    const resultContainer = document.getElementById('result');
    let selectedPlant = null;

    plantOptions.forEach(option => {
        option.addEventListener('click', function() {
            // Remove selected class from all options
            plantOptions.forEach(opt => opt.classList.remove('selected'));
            // Add selected class to clicked option
            this.classList.add('selected');
            // Store selected plant value
            selectedPlant = this.dataset.value;
            // Check if we can enable the diagnose button
            checkEnableDiagnoseButton();
        });
    });

    // File upload handling
    const fileUpload = document.getElementById('fileUpload');
    const fileUploadContainer = document.querySelector('.file-upload-container');
    const filePreview = document.querySelector('.file-preview');
    const imagePreview = document.getElementById('imagePreview');
    let selectedFile = null;

    fileUpload.addEventListener('change', function(event) {
        if (this.files && this.files[0]) {
            selectedFile = this.files[0];
            
            // Add has-file class to container
            fileUploadContainer.classList.add('has-file');
            
            // Show file preview
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                filePreview.classList.add('visible');
            }
            reader.readAsDataURL(selectedFile);
            
            // Check if we can enable the diagnose button
            checkEnableDiagnoseButton();
        }
    });

    // Diagnose button handling
    const diagnoseButton = document.getElementById('diagnoseButton');
    const predictedClassElement = document.getElementById('predictedClass');
    const resultDetailsElement = document.getElementById('resultDetails');
    const confidenceValueElement = document.getElementById('confidenceValue');
    const confidenceFill = document.querySelector('.confidence-bar .fill');

    function checkEnableDiagnoseButton() {
        diagnoseButton.disabled = !(selectedPlant && selectedFile);
    }

    diagnoseButton.addEventListener('click', function() {
        if (!selectedPlant || !selectedFile) return;

        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('plantType', selectedPlant);

        // Show loading state
        diagnoseButton.classList.add('loading');
        diagnoseButton.disabled = true;

        // Send request to server
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Reset button state
            diagnoseButton.classList.remove('loading');
            diagnoseButton.disabled = false;

            // Hide start prompt and show results
            if (startPrompt) startPrompt.classList.add('hidden');
            resultContainer.classList.add('visible');
            
            // Display results
            predictedClassElement.textContent = formatClassName(data.predicted_class);
            resultDetailsElement.textContent = getDetailedDescription(data.predicted_class);
            
            // Update confidence bar with animation
            const confidence = Math.round(data.confidence);
            
            // First update the value text
            confidenceValueElement.textContent = confidence;
            
            // Then animate the bar width with a slight delay
            setTimeout(() => {
                confidenceFill.style.width = `${confidence}%`;
            
                // Change color of confidence bar based on confidence
                if (confidence >= 90) {
                    confidenceFill.style.background = 'linear-gradient(to right, #10b981, #06b6d4)';
                } else if (confidence >= 70) {
                    confidenceFill.style.background = 'linear-gradient(to right, #10b981, #fbbf24)';
                } else {
                    confidenceFill.style.background = 'linear-gradient(to right, #fbbf24, #ef4444)';
                }
            }, 300);
            
            // Reset feedback state
            resetFeedbackState();
        })
        .catch(error => {
            console.error('Error:', error);
            diagnoseButton.classList.remove('loading');
            diagnoseButton.disabled = false;
            
            // Hide start prompt and show results with error
            if (startPrompt) startPrompt.classList.add('hidden');
            resultContainer.classList.add('visible');
            
            predictedClassElement.textContent = 'Error';
            resultDetailsElement.textContent = 'An error occurred during analysis. Please try again.';
            confidenceValueElement.textContent = '0';
            confidenceFill.style.width = '0%';
        });
    });
    
    // Format class name for better readability
    function formatClassName(className) {
        return className
            .replace(/_/g, ' ')
            .replace('/', '')
            .replace('  ', ' ')
            .replace(/\b\w/g, letter => letter.toUpperCase());
    }

    // Helper function to provide detailed descriptions
    function getDetailedDescription(predictedClass) {
        const descriptions = {
            'Potato___Early_blight': 'Early blight is a common fungal disease affecting potato plants. It causes dark brown spots with concentric rings. Treatment includes fungicides and ensuring proper plant spacing for air circulation.',
            'Potato___Late_blight': 'Late blight is a serious fungal disease that can destroy entire potato crops. It causes dark, water-soaked lesions that turn purplish-black. Immediate fungicide application is recommended.',
            'Potato___healthy': 'Your potato plant appears healthy with no visible signs of disease. Continue with good cultural practices to maintain plant health.',
            
            'Apple___Apple_scab': 'Apple scab is a fungal disease causing dark, scaly lesions on leaves and fruit. Control with fungicides and proper orchard sanitation by removing fallen leaves.',
            'Apple___Black_rot': 'Black rot is a fungal disease causing circular lesions on leaves and fruit rot. Prune infected branches and apply appropriate fungicides.',
            'Apple___Cedar_apple_rust': 'Cedar apple rust causes bright orange-yellow spots on leaves and fruit. Control by removing nearby cedar trees or using fungicides.',
            'Apple___healthy': 'Your apple tree appears healthy with no visible signs of disease. Continue with good cultural practices to maintain tree health.',
            
            'Corn/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Gray leaf spot is a fungal disease causing rectangular gray to tan lesions on corn leaves. Crop rotation and resistant varieties are recommended.',
            'Corn/Corn_(maize)___Common_rust_': 'Common rust appears as small, round to elongated pustules on both leaf surfaces. Apply fungicides early when symptoms first appear.',
            'Corn/Corn_(maize)___Northern_Leaf_Blight': 'Northern leaf blight causes large, cigar-shaped gray-green to tan lesions on corn leaves. Use resistant hybrids and timely fungicide applications.',
            'Corn/Corn_(maize)___healthy': 'Your corn plant appears healthy with no visible signs of disease. Continue with good cultural practices to maintain plant health.',
            
            'Grape/Grape___Black_rot': 'Black rot causes small, dark lesions on leaves and black, shriveled fruit. Remove infected fruit and apply fungicides.',
            'Grape/Grape___Esca_(Black_Measles)': 'Esca (Black Measles) causes tiger-stripe patterns on leaves and black spotting on fruit. Currently no effective control exists; focus on vineyard sanitation.',
            'Grape/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf blight causes reddish-brown spots with yellow halos. Apply fungicides and ensure proper air circulation in the canopy.',
            'Grape/Grape___healthy': 'Your grape vine appears healthy with no visible signs of disease. Continue with good cultural practices to maintain vine health.'
        };

        return descriptions[predictedClass] || 'Analysis complete. This appears to be ' + predictedClass.replace(/_/g, ' ').replace('/', ' ').replace('  ', ' ');
    }
    
    // Feedback system
    const feedbackYesBtn = document.getElementById('feedbackYes');
    const feedbackNoBtn = document.getElementById('feedbackNo');
    const feedbackCommentContainer = document.querySelector('.feedback-comment-container');
    const feedbackCommentTextarea = document.getElementById('feedbackComment');
    const submitFeedbackBtn = document.getElementById('submitFeedback');
    const feedbackThanks = document.getElementById('feedbackThanks');
    
    let feedbackValue = null;
    
    // Reset feedback state
    function resetFeedbackState() {
        feedbackValue = null;
        feedbackYesBtn.classList.remove('active');
        feedbackNoBtn.classList.remove('active');
        feedbackCommentContainer.classList.remove('visible');
        feedbackThanks.classList.remove('visible');
        feedbackCommentTextarea.value = '';
    }
    
    // Handle yes feedback
    feedbackYesBtn.addEventListener('click', function() {
        feedbackValue = 'positive';
        feedbackYesBtn.classList.add('active');
        feedbackNoBtn.classList.remove('active');
        feedbackCommentContainer.classList.add('visible');
    });
    
    // Handle no feedback
    feedbackNoBtn.addEventListener('click', function() {
        feedbackValue = 'negative';
        feedbackNoBtn.classList.add('active');
        feedbackYesBtn.classList.remove('active');
        feedbackCommentContainer.classList.add('visible');
    });
    
    // Handle feedback submission
    submitFeedbackBtn.addEventListener('click', function() {
        if (!feedbackValue) return;
        
        const feedbackData = {
            feedback_type: feedbackValue,
            comments: feedbackCommentTextarea.value,
            predicted_class: predictedClassElement.textContent,
            confidence: parseInt(confidenceValueElement.textContent),
            timestamp: new Date().toISOString()
        };
        
        // Log feedback to console (in a real app, this would be sent to a server)
        console.log('User Feedback:', feedbackData);
        
        // Show thank you message
        feedbackCommentContainer.classList.remove('visible');
        feedbackThanks.classList.add('visible');
        
        // In a real application, you would send this data to your server
        // fetch('/submit-feedback', {
        //     method: 'POST',
        //     headers: {
        //         'Content-Type': 'application/json',
        //     },
        //     body: JSON.stringify(feedbackData)
        // });
    });

    // Animated scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80, // Offset for the fixed header
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Animated counters function
    function animateCounters() {
        const counters = document.querySelectorAll('.number[data-count]');
        
        counters.forEach(counter => {
            const target = parseInt(counter.getAttribute('data-count'));
            const duration = 2000; // 2 seconds
            const startTime = Date.now();
            
            function updateCounter() {
                const currentTime = Date.now();
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                
                // Easing function for smoother animation
                const easeOutQuart = 1 - Math.pow(1 - progress, 4);
                
                // Calculate current count
                const currentCount = Math.round(target * easeOutQuart);
                
                // Update counter text
                counter.textContent = currentCount.toLocaleString();
                
                // Continue animation if not complete
                if (progress < 1) {
                    requestAnimationFrame(updateCounter);
                }
            }
            
            // Start the animation when scrolled into view
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        updateCounter();
                        observer.disconnect();
                    }
                });
            }, { threshold: 0.5 });
            
            observer.observe(counter);
        });
    }
});

