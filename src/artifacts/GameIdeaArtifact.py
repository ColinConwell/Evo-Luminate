import os
import json
import numpy as np
import torch


from src.models import text_embedder, llm_client, defaultModel
from src.artifacts.Artifact import Artifact


class GameIdeaArtifact(Artifact):
    name = "game idea"
    systemPrompt = """You are an expert in designing p5.js games.
    Rules:
    - The game is single player and will be run in a single html file
    
    Return the following fields:
    - name: The name of the game
    - description: A short description of the game
    - gameplay: How the game is played, including rules and objectives
    - mechanics: The technical systems that make the game work (movement, physics, scoring, etc.)
    - controls: How the player interacts with the game
    - art style: The visual aesthetic of the game
    - game objects: Key entities and their properties
    - win/lose conditions: When and how the game ends
    - any other relevant details
    """

    @classmethod
    def load(cls, id: str, results_dir: str):
        artifact = cls()
        artifact.id = id
        artifact.genome = open(os.path.join(results_dir, f"ideas/{id}.txt")).read()
        artifact.embedding = np.load(os.path.join(results_dir, f"embeddings/{id}.npy"))
        return artifact

    @classmethod
    def create_from_prompt(cls, prompt: str, output_dir: str, **kwargs):
        artifact = cls()
        artifact.prompt = prompt

        response = llm_client.chat.completions.create(
            model=defaultModel,
            max_completion_tokens=20000,
            reasoning_effort=kwargs.get("reasoning_effort", "low"),
            messages=[
                {"role": "system", "content": GameIdeaArtifact.systemPrompt},
                {"role": "user", "content": f"User prompt: {prompt}"},
            ],
        )

        artifact.genome = response.choices[0].message.content.strip()

        os.makedirs(os.path.join(output_dir, "ideas"), exist_ok=True)
        idea_path = os.path.join(output_dir, f"ideas/{artifact.id}.txt")
        with open(idea_path, "w") as f:
            f.write(artifact.genome)

        artifact.compute_embedding()
        os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
        embedding_path = os.path.join(output_dir, f"embeddings/{artifact.id}.npy")
        np.save(embedding_path, artifact.embedding.cpu().numpy())

        return artifact

    def compute_embedding(self) -> torch.Tensor:
        """Compute embedding for this game idea artifact"""
        if self.embedding is not None:
            return self.embedding

        self.embedding = text_embedder.embedText(self.genome)[0]
        return self.embedding

    def post_process(self, output_dir: str, **kwargs):
        rules = """
        - Must be a p5.js game in a single html file
        - Include: <script src="https://cdn.jsdelivr.net/npm/p5@1.4.2/lib/p5.js"></script>
        - Must be playable in a 512x512 iframe
        """

        system_prompt = f"""You are an expert p5.js game developer. 
        Convert the user's game idea into a complete playable and fun game.
        
        Requirements:
        {rules}
        
        Return ONLY the complete HTML file with embedded p5.js library and all game code.
        """

        response = llm_client.chat.completions.create(
            model=defaultModel,
            # model="anthropic:claude-4.7",
            max_completion_tokens=20000,
            reasoning_effort=kwargs.get("reasoning_effort", "low"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Here is the game idea to implement:\n\n{self.genome}",
                },
            ],
        )

        html_game = response.choices[0].message.content.strip()

        response = llm_client.chat.completions.create(
            model="openai:gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize the following game. Return only JSON with the following fields: 'name', 'description', 'rules'. \n {html_game}. ",
                },
            ],
            response_format={"type": "json_object"},
        )
        summary = json.loads(response.choices[0].message.content.strip())

        game_path = os.path.join(output_dir, f"{self.id}.html")
        with open(game_path, "w") as f:
            f.write(html_game)

        summary_path = os.path.join(output_dir, f"{self.id}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f)

        return game_path, summary_path
