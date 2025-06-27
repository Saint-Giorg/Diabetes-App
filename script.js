document.addEventListener('DOMContentLoaded', function() {
    // Theme Toggle
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            fetch('/toggle-theme')
                .then(() => window.location.reload());
        });
    }

    // Set gauge width on results page
    const gaugeFill = document.querySelector('.gauge-fill');
    if (gaugeFill) {
        const probability = parseFloat(document.querySelector('.probability-badge').textContent);
        gaugeFill.style.width = `${Math.min(100, probability)}%`;
    }

    // Form Validation
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            const glucose = parseFloat(document.getElementById('Glucose').value);
            if (isNaN(glucose)) {
                alert('Please enter a valid glucose level');
                e.preventDefault();
                return;
            }
            
            if (glucose < 50 || glucose > 300) {
                alert('Glucose level must be between 50-300 mg/dL');
                e.preventDefault();
            }
        });
    }
});