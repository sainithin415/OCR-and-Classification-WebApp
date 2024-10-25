async function predictCharacter() {
    const fileInput = document.getElementById('fileInput').files[0];
    const formData = new FormData();
    formData.append('file', fileInput);
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    document.getElementById('predictionResult').innerText = `Prediction: ${result.prediction}`;
}

async function extractText() {
    const textInput = document.getElementById('textInput').files[0];
    const formData = new FormData();
    formData.append('file', textInput);
    const response = await fetch('/extract-text', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    document.getElementById('textResult').innerText = `Extracted Text: ${result.text}`;
    document.getElementById('summaryResult').innerText = `Summary: ${result.summary}`;
}
