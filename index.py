from typing import Literal
from flask import Flask, jsonify
from flask.wrappers import Response
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from src.app.utils.config import Config
from src.app.controllers.saw_controller import saw_bp
from src.app.controllers.wp_controller import wp_bp
from src.app.connection.connection import Connection

from flask_cors import CORS
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)
Config.init_firebase()
CORS(app)

# Register blueprints
app.register_blueprint(saw_bp, url_prefix="/saw")
app.register_blueprint(wp_bp, url_prefix="/wp")

@app.route("/")
def home() -> tuple[Response, Literal[200]] | tuple[Response, Literal[500]]:
    try:
        col_ref = Connection.get_collection("results")
        doc_count = sum(1 for _ in col_ref.stream())
        return (
            jsonify(
                {
                    "message": f"Connected successfully! Collection has {doc_count} documents."
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": f"Connection failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
