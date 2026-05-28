document.addEventListener('DOMContentLoaded', () => {
    // File upload zones setup
    setupDropZone('jdDropZone', 'jdInput', 'jdFileName');
    setupDropZone('cvDropZone', 'cvInput', 'cvFileName');

    const form = document.getElementById('uploadForm');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const resultsGrid = document.getElementById('resultsGrid');
    const runBtn = document.getElementById('runBtn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const jdInput = document.getElementById('jdInput');
        const cvInput = document.getElementById('cvInput');

        if (!jdInput.files.length || !cvInput.files.length) {
            alert('Please select both a Job Description and a Candidate CSV.');
            return;
        }

        // UI State
        runBtn.disabled = true;
        runBtn.innerHTML = '<span>Processing...</span><i class="fa-solid fa-spinner fa-spin"></i>';
        resultsSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        const formData = new FormData(form);

        try {
            const response = await fetch('/api/rank', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Something went wrong');
            }

            renderResults(data.results);
            
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            // Restore UI State
            loadingSection.classList.add('hidden');
            runBtn.disabled = false;
            runBtn.innerHTML = '<span>Rank Candidates</span><i class="fa-solid fa-wand-magic-sparkles"></i>';
        }
    });

    function renderResults(candidates) {
        resultsGrid.innerHTML = '';
        
        candidates.forEach(candidate => {
            const card = document.createElement('div');
            card.className = 'candidate-card';
            
            const llmScore = candidate.llm_score !== null ? candidate.llm_score : 'N/A';
            const retrievalScore = parseFloat(candidate.retrieval_score).toFixed(4);
            
            card.innerHTML = `
                <div class="card-header">
                    <div class="rank-badge">#${candidate.rank}</div>
                    <div class="candidate-info">
                        <h3>${candidate.name}</h3>
                        <p>${candidate.title || 'Candidate'}</p>
                        <p style="font-size: 0.8rem; margin-top:0.2rem; color:var(--text-muted);">ID: ${candidate.candidate_id}</p>
                    </div>
                </div>
                <div class="scores">
                    <div class="score-badge"><span>Final:</span> ${candidate.final_score}</div>
                    <div class="score-badge"><span>LLM:</span> ${llmScore}</div>
                    <div class="score-badge"><span>Retrieval:</span> ${retrievalScore}</div>
                </div>
                <div class="justification">
                    ${candidate.justification}
                </div>
            `;
            resultsGrid.appendChild(card);
        });

        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function setupDropZone(zoneId, inputId, fileNameId) {
        const dropZone = document.getElementById(zoneId);
        const inputElement = document.getElementById(inputId);
        const fileNameElement = document.getElementById(fileNameId);

        dropZone.addEventListener('click', () => inputElement.click());

        inputElement.addEventListener('change', (e) => {
            if (inputElement.files.length) {
                fileNameElement.textContent = inputElement.files[0].name;
                dropZone.style.borderColor = 'var(--accent-1)';
                dropZone.style.background = 'rgba(139, 92, 246, 0.05)';
            }
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        ['dragleave', 'dragend'].forEach(type => {
            dropZone.addEventListener(type, (e) => {
                dropZone.classList.remove('dragover');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');

            if (e.dataTransfer.files.length) {
                inputElement.files = e.dataTransfer.files;
                fileNameElement.textContent = e.dataTransfer.files[0].name;
                dropZone.style.borderColor = 'var(--accent-1)';
                dropZone.style.background = 'rgba(139, 92, 246, 0.05)';
            }
        });
    }
});
