document.getElementById('uploadForm').addEventListener('submit', 
    async function(event) {
    event.preventDefault();

    const imageInput = document.getElementById('imageInput');
    const formData = new FormData();
    formData.append('file', imageInput.files[0]);  // Match the field name to 'file'

    try {
        // Adjust the URL to your Flask app's `/predict` endpoint
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        document.getElementById('response').textContent = 'Prediction: ' + result.prediction;

    } catch (error) {
        document.getElementById('response').textContent = 'Upload failed: ' + error.message;
    }

});
