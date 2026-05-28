import os
import tempfile
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template

from config import PipelineConfig
from main import run_pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/rank', methods=['POST'])
def rank_candidates():
    if 'jd' not in request.files or 'candidates' not in request.files:
        return jsonify({'error': 'Missing jd or candidates file'}), 400

    jd_file = request.files['jd']
    candidates_file = request.files['candidates']

    if jd_file.filename == '' or candidates_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save to temp
    jd_path = os.path.join(app.config['UPLOAD_FOLDER'], jd_file.filename)
    candidates_path = os.path.join(app.config['UPLOAD_FOLDER'], candidates_file.filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ranked_results.csv')
    
    jd_file.save(jd_path)
    candidates_file.save(candidates_path)

    # Get optional parameters
    retrieval_top_n = int(request.form.get('retrieval_top_n', 50))
    llm_top_n = int(request.form.get('llm_top_n', 20))
    alpha = float(request.form.get('alpha', 0.70))
    llm_provider = request.form.get('llm_provider', 'auto')
    api_key = request.form.get('api_key', '').strip()

    env_keys = {
        "xai": "XAI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY"
    }

    if api_key and llm_provider in env_keys:
        os.environ[env_keys[llm_provider]] = api_key
    elif api_key and llm_provider == "auto":
        return jsonify({'error': 'Please select a specific LLM Provider if you are providing an API Key.'}), 400

    try:
        config = PipelineConfig(
            retrieval_top_n=retrieval_top_n,
            llm_top_n=llm_top_n,
            hybrid_alpha=alpha,
            llm_provider=llm_provider,
            output_format='csv'
        )

        df = run_pipeline(jd_path, candidates_path, output_path, config)
        
        # Convert NaN to None for JSON serialization
        df = df.where(df.notnull(), None)
        results = df.to_dict(orient='records')

        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup
        for path in [jd_path, candidates_path, output_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)
