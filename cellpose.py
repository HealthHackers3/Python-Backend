from flask import Flask, request, jsonify
from cellpose import models, utils
from PIL import Image
import numpy as np
import os
import sys
import signal

app = Flask(__name__)
model = models.Cellpose(gpu=False, model_type='cyto3')


@app.route('/shutdown', methods=['POST'])
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"message": "Server shutting down..."}), 200


def shutdown_handler(signum, frame):
    """Handle termination signals gracefully."""
    print("Received shutdown signal, terminating Flask server.")
    sys.exit(0)


def resize_image(image):
    """
    Resize the image to ensure no dimension exceeds max_size while maintaining the aspect ratio.
    """
    max_size = 256
    width, height = image.size
    if max(width, height) > max_size:
        scaling_factor = max_size / float(max(width, height))
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(image)


def count_cells(image):
    """
    Count cells in the given image.
    """
    try:
        resized_img = resize_image(image)
        imgs = [resized_img]
        masks, flows, styles, diams = model.eval(imgs, batch_size=16, diameter=None, channels=[0, 0],
                                                 flow_threshold=0.4, do_3D=False)
        outlines = utils.outlines_list(masks[0])
        return len(outlines)
    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")


@app.route('/count_cells', methods=['POST'])
def count_cells_endpoint():
    try:
        # Check if the request contains a file
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        # Get the image file from the request
        image_file = request.files['image']
        image = Image.open(image_file)

        # Count the cells in the image
        cell_count = count_cells(image)

        # Return the cell count as JSON
        return jsonify({"cell_count": cell_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python app.py <port>")
        sys.exit(1)

    try:
        port = int(sys.argv[1])
    except ValueError:
        print("Invalid port number. Please provide a valid integer.")
        sys.exit(1)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(debug=True, port=port)
