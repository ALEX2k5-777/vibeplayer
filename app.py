from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import librosa
import numpy as np
import os
import json
from werkzeug.utils import secure_filename
import subprocess
import threading
from urllib.parse import quote
from yt_dlp import YoutubeDL
import hashlib
import time
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
JSON_FILE = os.path.join(BASE_DIR, 'history.json')
CACHE_FILE = os.path.join(BASE_DIR, 'ai_cache.json')

# Configure Ollama for local LLM (no API key needed, completely FREE)
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')  # lightweight & fast

# Check if Ollama is available
def is_ollama_available():
    try:
        response = requests.get(f'{OLLAMA_API_URL}/api/tags', timeout=2)
        return response.status_code == 200
    except:
        return False

# In-memory rate limiter (request_count, last_reset_time)
ai_rate_limit = {'count': 0, 'reset_time': time.time()}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_ai_cache():
    """Load AI response cache"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_ai_cache(cache):
    """Save AI response cache"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass

def get_prompt_hash(prompt):
    """Create hash key for prompt caching"""
    return hashlib.md5(prompt.encode()).hexdigest()

def get_fallback_response(prompt_type):
    """Provide fallback response when quota is exceeded"""
    fallbacks = {
        'lyrics': """[Lyrics not available - API quota exceeded]

This song's lyrics are protected by copyright.
Check your platform's music streaming service or official artist channels for full lyrics.
Consider upgrading your API plan for enhanced features.""",
        'composition': """[Composition details - API quota exceeded]

ESTIMATED TECHNICAL DETAILS:
Key: Based on detected chord
Time Signature: 4/4 (standard)
Structure: Verse - Chorus - Verse - Chorus - Bridge - Chorus

For accurate composition details, consult music theory resources or upgrade your API plan."""
    }
    return fallbacks.get(prompt_type, fallbacks['composition'])

def check_rate_limit():
    """Rate limit: 5 requests per minute"""
    global ai_rate_limit
    current_time = time.time()
    
    # Reset counter if 60 seconds have passed
    if current_time - ai_rate_limit['reset_time'] > 60:
        ai_rate_limit = {'count': 0, 'reset_time': current_time}
    
    # Check if limit exceeded
    if ai_rate_limit['count'] >= 5:
        return False
    
    ai_rate_limit['count'] += 1
    return True

def detect_exact_chord(y, sr):
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mean_chroma = np.mean(chroma, axis=1)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        maj_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        min_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        best_score = -1
        detected_chord = "Unknown"
        for i in range(12):
            score_maj = np.dot(mean_chroma, np.roll(maj_template, i))
            score_min = np.dot(mean_chroma, np.roll(min_template, i))
            if score_maj > best_score:
                best_score, detected_chord = score_maj, f"{notes[i]} Major"
            if score_min > best_score:
                best_score, detected_chord = score_min, f"{notes[i]} Minor"
        return detected_chord
    except: return "Unknown"

@app.route('/')
def serve_index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/static/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/ai', methods=['POST'])
def call_ai():
    """Backend endpoint for AI requests with caching and rate limiting (using Ollama locally - FREE, no quota)"""
    if not check_rate_limit():
        return jsonify({"error": "Rate limit exceeded. Max 5 requests per minute."}), 429
    
    data = request.json
    prompt = data.get('prompt', '').strip()
    prompt_type = data.get('type', 'composition')  # 'lyrics' or 'composition'
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    # Check cache first
    cache = get_ai_cache()
    prompt_hash = get_prompt_hash(prompt)
    
    if prompt_hash in cache:
        return jsonify({"response": cache[prompt_hash], "cached": True})
    
    # Check if Ollama is running
    if not is_ollama_available():
        fallback = get_fallback_response(prompt_type)
        return jsonify({
            "response": fallback,
            "cached": False,
            "fallback": True,
            "message": "Ollama is not running. Start Ollama with: ollama serve"
        }), 200
    
    try:
        # Call Ollama API (local, unlimited, no quota)
        response = requests.post(
            f'{OLLAMA_API_URL}/api/generate',
            json={
                'model': OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                'temperature': 0.7
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result_data = response.json()
            result_text = result_data.get('response', 'No response')
            
            # Cache the response
            cache[prompt_hash] = result_text
            save_ai_cache(cache)
            
            return jsonify({"response": result_text, "cached": False})
        else:
            return jsonify({"error": f"Ollama error: {response.status_code}"}), 500
            
    except requests.Timeout:
        return jsonify({"error": "Request timeout. Try a simpler prompt or shorter response."}), 500
    except Exception as e:
        error_msg = str(e)
        return jsonify({"error": f"AI request failed: {error_msg[:150]}"}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    try:
        # 10s sample for fast processing
        y, sr = librosa.load(path, duration=10)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(tempo)
        chord = detect_exact_chord(y, sr)
        
        # Advanced acoustic features
        try:
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            brightness = int(np.mean(centroid) / 50) if len(centroid) > 0 else 50
            brightness = max(0, min(100, brightness))
        except:
            brightness = 50

        try:
            rms_val = librosa.feature.rms(y=y)
            intensity = int(np.mean(rms_val) * 400) if len(rms_val) > 0 else 50
            intensity = max(0, min(100, intensity))
        except:
            intensity = 50

        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            rhythm = int(np.mean(onset_env) * 40) if len(onset_env) > 0 else 50
            rhythm = max(0, min(100, rhythm))
        except:
            rhythm = 50
        
        result = {
            "song_name": filename,
            "mood": "Energetic" if bpm > 118 else "Chill",
            "bpm": bpm,
            "chord": chord,
            "brightness": brightness,
            "intensity": intensity,
            "rhythm": rhythm,
            "file_url": f"http://127.0.0.1:5000/static/uploads/{filename}"
        }
        
        library = []
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, "r") as f:
                try: library = json.load(f)
                except: library = []
        
        library = [s for s in library if s.get('song_name') != filename]
        library.insert(0, result)
        with open(JSON_FILE, "w") as f:
            json.dump(library, f, indent=4)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/library')
def get_library():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            try: return jsonify(json.load(f))
            except: return jsonify([])
    return jsonify([])

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_song(filename):
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            lib = json.load(f)
        lib = [i for i in lib if i.get('song_name') != filename]
        with open(JSON_FILE, "w") as f:
            json.dump(lib, f, indent=4)
    return jsonify({"message": "OK"})

@app.route('/search', methods=['GET'])
def search_songs():
    """Search for songs on YouTube"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({"error": "No query"}), 400
    
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',
            'default_search': 'ytsearch',
            'socket_timeout': 30,
        }
        
        results = []
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f'ytsearch5:{query}', download=False)
            
            if info and 'entries' in info:
                for entry in info['entries'][:5]:
                    results.append({
                        'id': entry.get('id', ''),
                        'title': entry.get('title', 'Unknown'),
                        'channel': entry.get('channel', 'Unknown'),
                        'duration': entry.get('duration', 0),
                        'url': f"https://www.youtube.com/watch?v={entry['id']}"
                    })
        
        if not results:
            return jsonify({"error": "No videos found for this search"}), 400
            
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Search error: {str(e)[:150]}"}), 500

@app.route('/download', methods=['POST'])
def download_song():
    """Download song from YouTube URL"""
    data = request.json
    url = data.get('url', '').strip()
    title = data.get('title', 'song').strip()
    
    if not url or not title:
        return jsonify({"error": "Missing url or title"}), 400
    
    try:
        # Sanitize filename
        safe_title = secure_filename(title[:50])
        if not safe_title:
            safe_title = "downloaded_song"
        
        # Create filepath with .m4a extension (works without FFmpeg)
        filename = f"{safe_title}.m4a"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Avoid duplicates
        counter = 1
        while os.path.exists(filepath):
            counter += 1
            name = secure_filename(title[:40])
            filename = f"{name}_{counter}.m4a"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Download using yt-dlp Python module
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio',
            'quiet': False,
            'no_warnings': False,
            'outtmpl': filepath[:-4],  # Remove .m4a extension
            'socket_timeout': 30,
            'postprocessors': [],
        }
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_file = ydl.prepare_filename(info)
        except Exception as dl_err:
            return jsonify({"error": f"Download failed: {str(dl_err)[:150]}"}), 500
        
        # Analyze the downloaded file
        if os.path.exists(filepath):
            try:
                y, sr = librosa.load(filepath, duration=10)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm = int(tempo)
                chord = detect_exact_chord(y, sr)
                
                # Advanced acoustic features
                try:
                    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                    brightness = int(np.mean(centroid) / 50) if len(centroid) > 0 else 50
                    brightness = max(0, min(100, brightness))
                except:
                    brightness = 50

                try:
                    rms_val = librosa.feature.rms(y=y)
                    intensity = int(np.mean(rms_val) * 400) if len(rms_val) > 0 else 50
                    intensity = max(0, min(100, intensity))
                except:
                    intensity = 50

                try:
                    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                    rhythm = int(np.mean(onset_env) * 40) if len(onset_env) > 0 else 50
                    rhythm = max(0, min(100, rhythm))
                except:
                    rhythm = 50
                
                result_data = {
                    "song_name": filename,
                    "mood": "Energetic" if bpm > 118 else "Chill",
                    "bpm": bpm,
                    "chord": chord,
                    "brightness": brightness,
                    "intensity": intensity,
                    "rhythm": rhythm,
                    "file_url": f"http://127.0.0.1:5000/static/uploads/{filename}"
                }
                
                # Add to library
                library = []
                if os.path.exists(JSON_FILE):
                    with open(JSON_FILE, "r") as f:
                        try: library = json.load(f)
                        except: library = []
                
                library = [s for s in library if s.get('song_name') != filename]
                library.insert(0, result_data)
                with open(JSON_FILE, "w") as f:
                    json.dump(library, f, indent=4)
                
                return jsonify(result_data)
            except Exception as e:
                return jsonify({"error": f"Analysis failed: {str(e)[:200]}"}), 500
        else:
            return jsonify({"error": "File not found after download"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)