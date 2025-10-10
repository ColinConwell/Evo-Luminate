import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Add the parent directory to the Python path to access eluminate
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eluminate.models import make_image


def test_make_image():
    """Test the make_image function with a simple prompt."""
    prompt = "A beautiful sunset over mountains"
    print(f"Generating image with prompt: '{prompt}'")

    try:
        image = make_image(prompt)
        print(f"Success! Image URL: {image}")
        image.save("test.png")
        return True
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


def test_make_image_in_thread():
    """Test the make_image function when called from a thread."""
    prompt = "A colorful abstract painting"
    print(f"Generating image with prompt: '{prompt}' from a thread")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(make_image, prompt)
        try:
            image = future.result()
            print(f"Success from thread! Image URL: {image}")
            image.save("test2.png")
            return True
        except Exception as e:
            print(f"Error generating image from thread: {e}")
            return False


if __name__ == "__main__":
    print("Testing make_image function...")
    test_make_image()
    print("\nTesting make_image function in a thread...")
    test_make_image_in_thread()
