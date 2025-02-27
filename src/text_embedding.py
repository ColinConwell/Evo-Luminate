import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Union


class TextEmbedder:
    def __init__(self, model_name="intfloat/e5-small-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embedText(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text into embeddings using the E5 model.

        Args:
            text: A string or list of strings to encode. For best results,
                  prefix each text with "query: " or "passage: "

        Returns:
            Normalized embeddings as a PyTorch tensor
        """
        # Handle single string input
        if isinstance(text, str):
            text = [text]

        # Add prefix if not already present
        prefixed_texts = []
        for t in text:
            if not (t.startswith("query: ") or t.startswith("passage: ")):
                t = "passage: " + t
            prefixed_texts.append(t)

        # Tokenize the input texts
        batch_dict = self.tokenizer(
            prefixed_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self._average_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


# Example usage
if __name__ == "__main__":
    embedder = TextEmbedder()

    # Single text example
    single_text = "How do I make chocolate chip cookies?"
    embedding = embedder.embedText(single_text)
    print(f"Single embedding shape: {embedding.shape}")

    # Multiple texts example
    texts = [
        "How much protein should a female eat?",
        "What is the definition of summit?",
    ]
    embeddings = embedder.embedText(texts)
    print(f"Multiple embeddings shape: {embeddings.shape}")

    # Compare similarity between two texts
    text1 = "query: how much protein should a female eat"
    text2 = "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day."

    emb1 = embedder.embedText(text1)
    emb2 = embedder.embedText(text2)

    similarity = (emb1 @ emb2.T) * 100
    print(f"Similarity score: {similarity.item():.2f}")
