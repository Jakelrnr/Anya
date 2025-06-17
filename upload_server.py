from flask import Flask, send_from_directory, request, jsonify, send_file
import os
import uuid

def create_upload_app():
    app = Flask(__name__)
    AUDIO_FOLDER = "audio_responses"

    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    
    @app.route("/")
    def index():
        print("=== ROOT ROUTE ACCESSED ===")
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")
        
        public_path = os.path.join(current_dir, 'public')
        index_path = os.path.join(public_path, 'index.html')
        
        print(f"Looking for public folder at: {public_path}")
        print(f"Public folder exists: {os.path.exists(public_path)}")
        print(f"Looking for index.html at: {index_path}")
        print(f"index.html exists: {os.path.exists(index_path)}")
        
        if os.path.exists('public'):
            print("Files in public folder:", os.listdir('public'))
        
        if os.path.exists(index_path):
            print("SUCCESS: Serving index.html")
            try:
                # Try direct file reading first
                with open(index_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print("File read successfully, returning content")
                return content, 200, {'Content-Type': 'text/html'}
            except Exception as e:
                print(f"ERROR reading file directly: {e}")
                # Fallback to send_from_directory with absolute path
                try:
                    print("Trying send_from_directory with absolute path...")
                    return send_from_directory(os.path.abspath('public'), "index.html")
                except Exception as e2:
                    print(f"ERROR in send_from_directory: {e2}")
                    return f"Error serving file: {str(e2)}", 500
        else:
            return f"DEBUG: File not found. CWD: {current_dir}, Looking for: {index_path}", 404
    
    @app.route("/<path:filename>")
    def serve_static(filename):
        print(f"=== STATIC FILE REQUESTED: {filename} ===")
        # Check if file exists in public folder
        if os.path.exists(os.path.join('public', filename)):
            print(f"SUCCESS: Serving {filename} from public folder")
            return send_from_directory('public', filename)
        
        print(f"ERROR: File {filename} not found in public folder")
        return "File Not Found", 404

    @app.route("/upload_audio", methods=["POST"])
    def upload_audio():
        print("=== AUDIO UPLOAD REQUESTED ===")
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files["audio"]
        # Use .mp3 extension since that's what's being generated
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(AUDIO_FOLDER, filename)
        file.save(filepath)
        print(f"Audio saved to: {filepath}")
        return jsonify({"filepath": f"{AUDIO_FOLDER}/{filename}"})

    @app.route("/audio_responses/<filename>")
    def serve_audio(filename):
        print(f"=== AUDIO FILE REQUESTED: {filename} ===")
        print("Audio folder absolute path:", os.path.abspath(AUDIO_FOLDER))

        # Use absolute path and send_file to avoid directory issues
        audio_path = os.path.abspath(os.path.join(AUDIO_FOLDER, filename))
        if not os.path.exists(audio_path):
            print("!!! FILE NOT FOUND !!!")
            return f"File not found: {filename}", 404

        try:
            return send_file(audio_path, mimetype="audio/mpeg")
        except Exception as e:
            print(f"Failed to send audio file: {e}")
            return f"Error: {e}", 500

    return app
