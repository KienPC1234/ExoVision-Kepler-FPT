import os
from flask import Flask, send_from_directory, abort
from werkzeug.utils import safe_join

def create_app(base_dir="iframes", libs_dir="libs"):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(module_dir, base_dir)
    LIBS_DIR = os.path.join(module_dir, libs_dir)

    app = Flask(__name__)

    # Iframe routes
    @app.route("/iframe/<path:folder>/")
    def serve_iframe(folder):
        folder_path = os.path.join(BASE_DIR, folder)
        index_path = os.path.join(folder_path, "index.html")
        if os.path.isfile(index_path):
            return send_from_directory(folder_path, "index.html")
        else:
            abort(404)

    @app.route("/iframe/<path:folder>/<path:filename>")
    def serve_iframe_static(folder, filename):
        folder_path = os.path.join(BASE_DIR, folder)
        safe_path = safe_join(folder_path, filename)
        if safe_path and os.path.isfile(safe_path):
            return send_from_directory(folder_path, filename)
        else:
            abort(404)

    # Libs route
    @app.route("/libs/<path:filepath>")
    def serve_libs(filepath):
        safe_path = safe_join(LIBS_DIR, filepath)
        if safe_path and os.path.isfile(safe_path):
            rel_path = os.path.relpath(safe_path, LIBS_DIR)
            return send_from_directory(LIBS_DIR, rel_path)
        else:
            abort(404)

    return app

def run_server(base_dir="iframes", libs_dir="libs", host="0.0.0.0", port=8080):
    app = create_app(base_dir, libs_dir)
    app.run(host=host, port=port)
