function uploadImage() {
    const input = document.getElementById('imageInput');
    if (input.files.length === 0) {
      alert('Please select an image first');
      return;
    }
  
    const formData = new FormData();
    formData.append('image', input.files[0]);
  
    fetch('/analyze', {
      method: 'POST',
      body: formData,
    })
      .then(response => response.json())
      .then(data => {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '';
  
        if (data.faces) {
          data.faces.forEach((face, index) => {
            const faceInfo = `
              <div>
                <h3>Face ${index + 1}</h3>
                <p>Symmetry Score: ${face.symmetry_score.toFixed(2)}</p>
                <p>Proportion Score: ${face.proportion_score.toFixed(2)}</p>
                <p>Alignment Score: ${face.alignment_score.toFixed(2)}</p>
                <p>Paralysis Detection: ${face.paralysis_detection}</p>
              </div>
            `;
            resultsDiv.innerHTML += faceInfo;
          });
        } else {
          resultsDiv.innerHTML = '<p>No faces detected.</p>';
        }
      })
      .catch(err => {
        console.error(err);
        alert('An error occurred during analysis.');
      });
  }
  
  function updateResults(data) {
    if (data.error) {
        throw new Error(data.error);
    }

    // Update metrics
    document.getElementById('symmetry-score').textContent = data.symmetry_score?.toFixed(2) || 'N/A';
    document.getElementById('proportion-score').textContent = data.proportion_score?.toFixed(2) || 'N/A';
    
    // Update paralysis detection
    const paralysisElement = document.getElementById('paralysis-detection');
    if (data.paralysis_detection === "Possible Paralysis Detected") {
        paralysisElement.innerHTML = `<span class="paralysis-warning">${data.paralysis_detection}</span>`;
    } else {
        paralysisElement.textContent = data.paralysis_detection || 'N/A';
    }

    // Update text sections
    document.getElementById('overview-text').textContent = data.overview_text;
    document.getElementById('symmetry-text').textContent = data.symmetry_analysis;
    document.getElementById('proportion-text').textContent = data.proportion_analysis;
    document.getElementById('neuro-text').textContent = data.neuro_assessment;

    // Handle recommendations
    const actionSteps = document.getElementById('action-steps');
    actionSteps.innerHTML = data.recommendations.map((rec, index) => `
        <div class="step">
            <div class="step-icon">${index + 1}</div>
            <div>${rec}</div>
        </div>
    `).join('');

    // Show critical alert if needed
    const criticalAlert = document.getElementById('critical-alert');
    if (!data.alignment_score) {
        criticalAlert.style.display = 'flex';
        document.getElementById('alert-message').textContent = 
            'Missing Alignment Score prevents complete assessment';
    }
}