from aider.coders import Coder
from aider.models import Model, ModelSettings
from aider.io import InputOutput
import shutil

# import logging
import io

shutil.copy("test_shader.glsl", "test_shader_v2.glsl")

fnames = ["test_shader_v2.glsl"]

# settings = ModelSettings(name="openai/o3-mini", extra_params={"reasoning_effort": "low"})

# model = Model("openai/o3-mini")
model = Model("anthropic/claude-3-7-sonnet-20250219")
# model.extra_params = {"reasoning_effort": "low"}
# model = Model("gemini/gemini-1.5-pro-latest")
# model = Model("gemini/gemini-2.0-flash-thinking-exp")

# Use the custom IO class
io = InputOutput(yes=True)

# Create a coder object
coder = Coder.create(main_model=model, fnames=fnames, io=io)

# Run the coder to apply changes
coder.run(
    "Use fractal Brownian motion (fBm) to generate detailed, cloud-like nebula wisps that blend seamlessly with the existing nebula effect. By modulating the fBm parameters with time, you can create evolving, ethereal patterns that add depth and complexity to the overall composition. Ensure the shader is valid glsl es"
)

# Directly read the modified shader content
# new_shader_content = io.read_text("test_shader.glsl")

# # Save the new content to a file
# with open("test_shader_v2.glsl", "w") as f:
#     f.write(new_shader_content)
