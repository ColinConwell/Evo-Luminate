import aisuite as ai
from .text_embedding import TextEmbedder
from .image_embedding import ImageEmbedder
import asyncio
from PIL import Image
import requests
from io import BytesIO

# Global LLM client
llm_client = ai.Client()

# Global embedders
text_embedder = TextEmbedder()
image_embedder = ImageEmbedder()

# defaultModel = "openai:o3-mini"
# defaultModel = "openai:gpt-4o"
defaultModel = "openai:o1"

# defaultModel = "anthropic:claude-4.7"


def make_image(prompt, seed=1, aspect_ratio="1:1", num_images=1) -> Image:
    import replicate

    output = replicate.run(
        "black-forest-labs/flux-dev",
        input={
            "prompt": prompt,
            "go_fast": True,
            "megapixels": "1",
            "num_outputs": num_images,
            "aspect_ratio": aspect_ratio,
            "seed": seed,
            "output_format": "webp",
            "output_quality": 90,
            "num_inference_steps": 28,
            "disable_safety_checker": True,
            "guidance": 3.5,
        },
    )

    # Get the image data from the URL
    response = requests.get(output[0])

    # Create and return a PIL Image object
    img = Image.open(BytesIO(response.content))
    return img


# async def _submit_image_request(prompt, seed=1, image_size="square", num_images=1):
#     import fal_client

#     """Internal async function to submit image generation request to fal-ai/flux"""
#     handler = await fal_client.submit_async(
#         "fal-ai/flux/schnell",
#         arguments={
#             "prompt": prompt,
#             "seed": seed,
#             "image_size": image_size,
#             "num_images": num_images,
#             "enable_safety_checker": False,
#         },
#     )
#     return handler


# def make_image(prompt, seed=1, image_size="square") -> str:
#     """
#     Generate images based on a text prompt using fal-ai/flux model.

#     Args:
#         prompt (str): Text description of the image to generate
#         seed (int, optional): Random seed for reproducibility. Defaults to 1.
#         image_size (str, optional): Size/aspect ratio of generated images. Defaults to "square".
#         num_images (int, optional): Number of images to generate. Defaults to 1.

#     Returns:
#         str: URL of the generated image
#     """
#     # Create and run the async task
#     loop = asyncio.new_event_loop()
#     try:
#         # Submit the request and wait for result
#         handler = loop.run_until_complete(
#             _submit_image_request(prompt, seed, image_size, num_images=1)
#         )
#         result = loop.run_until_complete(handler.wait_for_result())
#         return result["images"][0]["url"]
#     finally:
#         # Always close the loop to free resources
#         loop.close()

#     # Extract and return only the image URL
# return result["images"][0]["url"]
