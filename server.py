import os
import uuid
import subprocess
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/extract', methods=['POST'])
def extract():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    mode = request.form.get('mode', 'fast') # fast or qwen
    
    # Save file to temp
    ext = os.path.splitext(file.filename)[1]
    if not ext: ext = ".png"
    temp_img_path = f"temp_{uuid.uuid4().hex}{ext}"
    temp_json_path = f"temp_{uuid.uuid4().hex}.json"
    
    file.save(temp_img_path)
    
    try:
        cmd = ["python", "executable.py", temp_img_path, "--output", temp_json_path]
        if mode == 'Fast Mode / No VLM' or mode == 'fast':
            cmd.append("--no_vlm")
        elif mode == 'qwen':
            cmd.extend(["--vlm_provider", "qwen"])
            
        subprocess.run(cmd, check=True)
        
        with open(temp_json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
            
        return jsonify(result)
        
    except subprocess.CalledProcessError as e:
        print("Error running extraction:", e)
        return jsonify({"error": "Extraction failed via executable.py"}), 500
    finally:
        if os.path.exists(temp_img_path):
            try: os.remove(temp_img_path)
            except: pass
        if os.path.exists(temp_json_path):
            try: os.remove(temp_json_path)
            except: pass

if __name__ == '__main__':
    app.run(port=5000, debug=True)
