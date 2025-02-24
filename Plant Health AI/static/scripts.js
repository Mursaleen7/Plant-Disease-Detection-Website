document.addEventListener('DOMContentLoaded', () => {
    // Form submission handling
    document.getElementById('uploadForm').addEventListener('submit', function(event) {
        event.preventDefault();
        var formData = new FormData(this);
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerHTML = 
                `<p>Predicted Class: ${data.predicted_class}</p>
                <p>Confidence: ${data.confidence}%</p>`;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').innerHTML = '<p>An error occurred</p>';
        });
    });

    // Accordion functionality for FAQ
    const questionBox = document.getElementsByClassName("question__box");
    for (let i = 0; i < questionBox.length; i++) {
        questionBox[i].addEventListener("click", function () {
            this.classList.toggle("active");
        });
    }
});

