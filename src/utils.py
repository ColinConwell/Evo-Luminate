import re
import yaml
from typing import Dict, Optional
import base64, logging
from io import BytesIO
from PIL import Image
import requests
import torch


def load_image_url_base64(url: str, format: str = "JPEG") -> Optional[str]:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Convert image and encode as base64
        img = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        img = img.convert("RGB")  # Convert to RGB (in case of PNG with transparency)
        img.save(buffered, format=format, quality=90)

        # Return base64 encoded string with proper data URL format
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"
    except Exception as e:
        logging.error(f"Failed to load image from URL: {e}")
        return None


def load_image_path_base64(file_path: str, format: str = "JPEG") -> Optional[str]:
    try:
        # Open the local image file
        img = Image.open(file_path)
        buffered = BytesIO()
        img = img.convert("RGB")  # Convert to RGB (in case of PNG with transparency)
        img.save(buffered, format=format, quality=90)

        # Return base64 encoded string with proper data URL format
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"
    except Exception as e:
        logging.error(f"Failed to load image from path: {e}")
        return None


def extractCode(source: str) -> str:
    """
    Extracts code blocks enclosed in triple backticks from a source string.

    Args:
        source (str): The source string containing code blocks

    Returns:
        str: The extracted code without the backticks
    """
    # Regular expression to match code blocks enclosed in triple backticks
    # This pattern matches:
    # - Opening triple backticks, optionally followed by a language identifier
    # - Any content (including newlines) until closing triple backticks
    pattern = r"```(?:\w*\n|\n)(.*?)```"

    # Use re.DOTALL to make '.' match newlines as well
    matches = re.findall(pattern, source, re.DOTALL)

    if not matches:
        return source

    # Return the first match
    # Note: If you want to extract all code blocks, you could return matches instead
    return matches[0]


def extractBlocks(text: str) -> Dict[str, str]:
    """
    Extract content from all tagged blocks in a string and return as a dictionary.

    Args:
        text: The input string containing tagged code blocks like <BLOCK_NAME>content</BLOCK_NAME>

    Returns:
        A dictionary mapping block names to their content
    """
    # Define the pattern to match content between tags
    pattern = r"<(\w+)>(.*?)</\1>"

    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # Create a dictionary of block_name -> content
    blocks = {}
    for tag, content in matches:
        blocks[tag] = content.strip()

    return blocks


def saveCodeBlocks(blocks: Dict[str, str], filepath: str) -> None:
    """
    Save code blocks to a YAML file that preserves formatting and readability.

    Args:
        blocks: Dictionary of block names to code content
        filepath: Path to save the YAML file
    """
    # Configure YAML to use block style for multiline strings
    yaml.add_representer(
        str,
        lambda dumper, data: (
            dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            if "\n" in data
            else dumper.represent_scalar("tag:yaml.org,2002:str", data)
        ),
    )

    with open(filepath, "w") as f:
        yaml.dump(blocks, f, sort_keys=False)


def loadCodeBlocks(filepath: str) -> Dict[str, str]:
    """
    Load code blocks from a YAML file.

    Args:
        filepath: Path to the YAML file

    Returns:
        Dictionary of block names to code content
    """
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def get_device():
    """Get the best available device for PyTorch (CUDA, MPS, or CPU)
    
    Returns:
        torch.device: The best available device for Apple Silicon, NVIDIA GPUs, or CPU fallback
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
