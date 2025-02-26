import aisuite as ai
from text_embedding import TextEmbedder
from image_embedding import ImageEmbedder

# Global LLM client
llm_client = ai.Client()

# Global embedders
text_embedder = TextEmbedder()
image_embedder = ImageEmbedder()
