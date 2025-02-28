import os
import subprocess
import time
import json
import logging


def shader_to_image(code, image_path, width, height, uniforms={}):
    """
    Calls the Node.js renderer ("render-shaders.js") to render self.code.
    """
    spec = {
        "fragmentShader": code,
        "width": width,
        "height": height,
        "uniforms": uniforms,
    }
    input_json = json.dumps(spec)
    try:
        subprocess.run(
            ["node", "src/render-shaders/render-shaders.js", image_path],
            input=input_json,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error("Error rendering shader: %s", e)
        return None

    timeout = 2
    start_time = time.time()
    while not os.path.exists(image_path) and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    if not os.path.exists(image_path):
        logging.error("Rendered image %s not found", image_path)
        return None
    return image_path
