import aisuite as ai
from .text_embedding import TextEmbedder
from .image_embedding import ImageEmbedder

# Global LLM client
llm_client = ai.Client()

# Global embedders
text_embedder = TextEmbedder()
image_embedder = ImageEmbedder()

# defaultModel = "openai:o3-mini"
# defaultModel = "openai:gpt-4o"
defaultModel = "openai:o1"
