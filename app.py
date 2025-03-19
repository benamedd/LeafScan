# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Importer Flask-CORS pour gérer CORS
import leaf_analysis
from PIL import Image
import numpy as np
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Autoriser toutes les origines pour éviter les erreurs CORS

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Handles file upload and image processing."""
    if "file" not in request.files:
        return jsonify({"error": "No file received"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the image temporarily for processing
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    logger.debug(f"Temporary image saved at: {file_path}")

    try:
        # Analyze the image
        logger.debug("Starting image analysis")
        processed_image, leaf_mask, infected_mask = leaf_analysis.load_and_process_image(file_path)
        severity = leaf_analysis.calculate_severity(leaf_mask, infected_mask)
        logger.debug(f"Severity calculated: {severity:.2f}%")

        # Save the infected mask for visualization
        infected_mask_path = os.path.join(app.config["UPLOAD_FOLDER"], "infected_mask.png")
        infected_image = Image.fromarray((infected_mask * 255).astype(np.uint8))
        infected_image.save(infected_mask_path)

        return jsonify({
            "severity": f"{severity:.2f}%",
            "image_url": f"/static/uploads/{file.filename}",
            "infected_mask_url": f"/static/uploads/infected_mask.png"
        })

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Modifier pour accepter toutes les connexions
