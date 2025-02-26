import json
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional, Union


from models import llm_client, text_embedder, image_embedder
from artifacts import Artifact


def generate_evolve_ideas(
    parents: List[Artifact], strategy: str = "", count: int = 3
) -> List[str]:

    # Determine if this is variation (1 parent) or combination (2+ parents)
    is_variation = len(parents) == 1
    # Extract parent information
    parent_descriptions = []
    for i, parent in enumerate(parents):
        desc = f"PARENT {i+1}:```{parent.genome}```"
        parent_descriptions.append(desc)

    # Get the prompt from the first parent
    original_prompt = parents[0].prompt

    systemPrompt = f"""You are an expert in creating ideas for evolving artifacts.
    Return a JSON object with an "ideas" field containing an array of {count} strings, 
    where each string is a description of an idea.
    """
    promptPrefix = (
        "Create a variation of the parent"
        if is_variation
        else "Create a combination of the parents"
    )
    prompt = f"""
    {promptPrefix}
    {' '.join(parent_descriptions)}
    PROMPT: {original_prompt}
    """

    # Generate ideas using the model
    response = llm_client.chat.completions.create(
        model="openai:gpt-4o-mini",
        messages=[
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)
    # Parse JSON response
    # try:
    #     response_json = json.loads(response.choices[0].message.content)
    #     ideas = response_json.get("ideas", [])

    #     # Make sure we have some ideas
    #     if not ideas:
    #         logging.warning(f"No ideas found in JSON response for {operation_description}")
    #         return [f"Simple {operation_description} of the original shader"]

    #     return ideas[:count]  # Return at most 'count' ideas
    # except json.JSONDecodeError as e:
    #     logging.error(f"Error parsing JSON response: {e}")
    #     # Fallback to simple string response
    #     text = response.choices[0].message.content.strip()
    #     return [text]
