import os
import json
import time
import math
import uuid
import random
import argparse
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

# Use non-interactive backend by default; switch if interactive
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eluminate.utils import get_device
from eluminate.image_embedding import ImageEmbedder


# ------------------------- Rendering (p5.js via Node) -------------------------

def _fallback_render_clock(image_path: str, params: Dict[str, Any], width: int, height: int) -> str:
    """Render a simple static clock-like image using matplotlib as a fallback."""
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    bg = np.array(params.get("bg", [240, 240, 245])) / 255.0
    fg = np.array(params.get("fg", [20, 20, 30])) / 255.0
    ring_color = np.array(params.get("ring_color", [120, 180, 255])) / 255.0
    ring_count = int(params.get("ring_count", 12))
    ring_thickness = float(params.get("ring_thickness", 8.0))

    fig, ax = plt.subplots(figsize=(width/96, height/96), dpi=96)
    ax.set_facecolor(bg)
    ax.axis('off')
    ax.set_aspect('equal')

    # Draw outer circle
    R = 0.4
    circle = plt.Circle((0.5, 0.5), R, fill=False, color=fg, linewidth=2)
    ax.add_patch(circle)

    # Rings
    for i in range(ring_count):
        r = R * (i+1)/ring_count
        theta = 2 * math.pi * ((i+1)/ring_count)
        # arc as many small segments
        t = np.linspace(-math.pi/2, -math.pi/2 + theta, 60)
        x = 0.5 + r * np.cos(t)
        y = 0.5 + r * np.sin(t)
        ax.plot(x, y, color=ring_color, linewidth=max(1.0, ring_thickness * (i+1)/ring_count / 6.0))

    # Hands
    ax.plot([0.5, 0.5], [0.5, 0.5 - R*0.7], color=fg, linewidth=2)
    ax.plot([0.5, 0.5 + R*0.5], [0.5, 0.5], color=fg, linewidth=3)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(image_path, dpi=96)
    plt.close(fig)
    return image_path


def render_p5_to_image(sketch_code: str, image_path: str, width: int = 512, height: int = 512, frames: int = 1, fmt: str = "png", params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Render a p5.js sketch to an image using the Node renderer at eluminate/render-p5js/render-p5.js
    """
    spec = {
        "width": width,
        "height": height,
        "frames": frames,
        "format": fmt,
        "sketchCode": sketch_code,
    }
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    try:
        # First try default node
        subprocess.run(
            ["node", "eluminate/render-p5js/render-p5.js", image_path],
            input=json.dumps(spec),
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e1:
        print(f"Error rendering p5 sketch with system node: {e1}")
        # If Volta is available, try forcing Node 18
        try:
            subprocess.run(
                ["volta", "run", "--node", "18", "node", "eluminate/render-p5js/render-p5.js", image_path],
                input=json.dumps(spec),
                text=True,
                check=True,
            )
        except Exception as e2:
            print(f"Volta attempt failed: {e2}")
            # Fallback to matplotlib rendering to keep the walkthrough flowing
            if params is not None:
                try:
                    return _fallback_render_clock(image_path, params, width, height)
                except Exception as fe:
                    print("Fallback rendering failed:", fe)
            return None

    timeout = 5
    start = time.time()
    # The renderer may suffix frame number; find the single PNG in directory
    target_dir = os.path.dirname(image_path)
    target_stem = os.path.splitext(os.path.basename(image_path))[0]
    while time.time() - start < timeout:
        # Find a saved png that begins with target_stem
        for fname in os.listdir(target_dir):
            if fname.startswith(target_stem) and fname.endswith(f".{fmt}"):
                return os.path.join(target_dir, fname)
        time.sleep(0.1)
    print(f"Rendered image not found for {image_path}")
    return None


# ----------------------------- Artifact definition ----------------------------

def default_clock_sketch(params: Dict[str, Any]) -> str:
    """
    Generate a simple p5.js clock sketch using parameters.
    Params keys: bg, fg, ring_color, ring_count, ring_thickness
    """
    bg = params.get("bg", [240, 240, 245])
    fg = params.get("fg", [20, 20, 30])
    ring_color = params.get("ring_color", [120, 180, 255])
    ring_count = int(params.get("ring_count", 12))
    ring_thickness = float(params.get("ring_thickness", 8.0))

    # Simple parameterized radial clock (not time-accurate, illustrative only)
    code = f"""
      p.setup = function() {{
        myCanvas = p.createCanvas(width, height);
        p.angleMode(p.DEGREES);
        p.noLoop();
      }};

      p.draw = function() {{
        p.background({bg[0]}, {bg[1]}, {bg[2]});
        p.translate(width/2, height/2);
        p.noFill();
        p.stroke({fg[0]}, {fg[1]}, {fg[2]});
        p.strokeWeight(2);
        p.circle(0, 0, Math.min(width, height)*0.8);

        let rc = {ring_count};
        let thick = {ring_thickness};
        let baseR = Math.min(width, height) * 0.35;
        for (let i=0; i<rc; i++) {{
          let r = baseR * (i+1)/rc;
          p.stroke({ring_color[0]}, {ring_color[1]}, {ring_color[2]}, 180);
          p.strokeWeight(thick * (i+1)/rc);
          p.arc(0, 0, 2*r, 2*r, -90, -90 + 360 * ((i+1)/rc));
        }}

        // simple hands
        p.stroke({fg[0]}, {fg[1]}, {fg[2]});
        p.strokeWeight(4);
        p.line(0,0, 0, -baseR*0.7);
        p.strokeWeight(6);
        p.line(0,0, baseR*0.5, 0);
      }};
    """
    return code


def mutate_params(params: Dict[str, Any], rate: float = 0.3) -> Dict[str, Any]:
    """Randomly perturb clock parameters."""
    newp = dict(params)
    if random.random() < rate:
        newp["bg"] = [max(0, min(255, c + random.randint(-30, 30))) for c in newp.get("bg", [240, 240, 245])]
    if random.random() < rate:
        newp["fg"] = [max(0, min(255, c + random.randint(-30, 30))) for c in newp.get("fg", [20, 20, 30])]
    if random.random() < rate:
        newp["ring_color"] = [max(0, min(255, c + random.randint(-30, 30))) for c in newp.get("ring_color", [120, 180, 255])]
    if random.random() < rate:
        newp["ring_count"] = int(max(6, min(24, (newp.get("ring_count", 12) + random.choice([-2, -1, 1, 2])))))
    if random.random() < rate:
        newp["ring_thickness"] = float(max(2.0, min(20.0, (newp.get("ring_thickness", 8.0) + random.uniform(-2, 2)))))
    return newp


@dataclass
class ClockArtifact:
    id: str
    params: Dict[str, Any]
    genome: str = field(default_factory=str)
    phenome: Optional[str] = None  # image path
    embedding: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_random(cls) -> "ClockArtifact":
        params = {
            "bg": [random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)],
            "fg": [random.randint(0, 60), random.randint(0, 60), random.randint(0, 60)],
            "ring_color": [random.randint(80, 200), random.randint(80, 200), random.randint(80, 200)],
            "ring_count": random.randint(8, 16),
            "ring_thickness": random.uniform(4, 12),
        }
        gid = str(uuid.uuid4())
        genome = default_clock_sketch(params)
        return cls(id=gid, params=params, genome=genome)

    def mutate(self, mutation_rate: float = 0.3) -> "ClockArtifact":
        new_params = mutate_params(self.params, rate=mutation_rate)
        gid = str(uuid.uuid4())
        return ClockArtifact(id=gid, params=new_params, genome=default_clock_sketch(new_params))


# ----------------------------- Evolutionary steps -----------------------------

def step_initialize_population(pop_size: int) -> List[ClockArtifact]:
    return [ClockArtifact.create_random() for _ in range(pop_size)]


def step_render_population(artifacts: List[ClockArtifact], out_dir: str, width: int, height: int) -> None:
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for a in artifacts:
        img_path = os.path.join(img_dir, f"{a.id}.png")
        rendered = render_p5_to_image(a.genome, img_path, width=width, height=height, frames=1, fmt="png", params=a.params)
        a.phenome = rendered


def step_embed_population(artifacts: List[ClockArtifact], embedder: ImageEmbedder, device: torch.device) -> torch.Tensor:
    embs: List[torch.Tensor] = []
    for a in artifacts:
        if a.phenome and os.path.exists(a.phenome):
            emb = embedder.embedImage(a.phenome)
            a.embedding = emb
            embs.append(emb.to(device=device, dtype=torch.float32))
        else:
            # Fallback: zero vector (avoid crashing)
            embs.append(torch.zeros(512, dtype=torch.float32, device=device))
    return torch.stack(embs)


def step_compute_novelty(embeddings: torch.Tensor, k_neighbors: int = 3) -> Tuple[List[int], torch.Tensor]:
    embeddings = embeddings.to(torch.float32)
    if embeddings.shape[0] <= k_neighbors:
        idx = list(range(embeddings.shape[0]))
        return idx, torch.zeros(embeddings.shape[0], dtype=torch.float32, device=embeddings.device)
    norm = torch.nn.functional.normalize(embeddings, dim=1)
    sim = norm @ norm.T
    dist = 1.0 - sim
    dist.fill_diagonal_(float("inf"))
    k_nearest, _ = torch.topk(dist, k=k_neighbors, dim=1, largest=False)
    novelty = k_nearest.mean(dim=1)
    order = torch.argsort(novelty, descending=True).tolist()
    return order, novelty


def step_select(artifacts: List[ClockArtifact], order: List[int], keep: int) -> List[ClockArtifact]:
    return [artifacts[i] for i in order[:keep]]


def step_evolve(parents: List[ClockArtifact], num_children: int, mutation_rate: float = 0.3) -> List[ClockArtifact]:
    children: List[ClockArtifact] = []
    for _ in range(num_children):
        p = random.choice(parents)
        children.append(p.mutate(mutation_rate))
    return children


def step_dashboard(
    out_dir: str,
    artifacts: List[ClockArtifact],
    novelty: torch.Tensor,
    title: str,
    show: bool = False,
    max_samples: int = 8,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    # Sort by novelty desc for display
    idx = torch.argsort(novelty, descending=True).tolist()
    sel = [artifacts[i] for i in idx[:max_samples] if artifacts[i].phenome and os.path.exists(artifacts[i].phenome)]

    cols = min(4, max(1, len(sel)))
    rows = math.ceil(len(sel) / cols) if sel else 1
    fig = plt.figure(figsize=(4*cols, 3*rows + 3))
    gs = fig.add_gridspec(rows + 1, cols)

    # Image grid
    for i, art in enumerate(sel):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
        img = plt.imread(art.phenome)
        ax.imshow(img)
        nv = novelty[idx[i]].item() if i < len(idx) else float('nan')
        ax.set_title(f"nov={nv:.3f}", fontsize=9)

    # Novelty bar plot
    axn = fig.add_subplot(gs[rows, :])
    axn.plot(novelty.cpu().numpy(), marker='o')
    axn.set_title("Novelty per artifact (avg k-NN distance)")
    axn.set_xlabel("artifact index")
    axn.set_ylabel("novelty")
    fig.suptitle(title)

    out_path = os.path.join(out_dir, f"dashboard_{int(time.time())}.png")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    if show:
        # If interactive environment is desired, use a GUI backend
        pass

    return out_path


# ----------------------------------- main ------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Walkthrough: evolve p5.js clocks step-by-step")
    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "clocks_walkthrough"))
    parser.add_argument("--population_size", type=int, default=8)
    parser.add_argument("--children_per_generation", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--k_neighbors", type=int, default=3)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--interactive", action="store_true", default=False)
    parser.add_argument("--show_dashboard", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_cpu", action="store_true", default=False)
    args = parser.parse_args()

    if not args.interactive:
        matplotlib.use("Agg", force=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device(cpu_only=args.force_cpu)
    print("Using device:", device)
    embedder = ImageEmbedder(device=device)

    def pause(msg: str):
        if args.interactive:
            input(f"\n{msg} Press Enter to continue...")

    # Step 1: Initialize
    population = step_initialize_population(args.population_size)
    pause("Initialized a diverse seed population.")

    # Step 2: Render phenotypes
    step_render_population(population, args.output_dir, args.width, args.height)
    pause("Rendered phenotypes (p5.js → images).")

    # Step 3: Embed
    embeddings = step_embed_population(population, embedder, device)
    order, novelty = step_compute_novelty(embeddings, k_neighbors=args.k_neighbors)
    dash_path = step_dashboard(args.output_dir, population, novelty, title="Generation 0", show=args.show_dashboard)
    print("Saved dashboard:", dash_path)
    pause("Embedded and measured novelty for generation 0.")

    # Generations
    for gen in range(1, args.num_generations + 1):
        # Select parents by novelty order
        parents = step_select(population, order, keep=args.population_size)
        # Evolve new children
        children = step_evolve(parents, num_children=args.children_per_generation, mutation_rate=0.35)
        population.extend(children)
        # Re-render only new children
        step_render_population(children, args.output_dir, args.width, args.height)
        # Re-embed all (small pop sizes so fine)
        embeddings = step_embed_population(population, embedder, device)
        order, novelty = step_compute_novelty(embeddings, k_neighbors=args.k_neighbors)
        # Down-select to target population size
        population = step_select(population, order, keep=args.population_size)
        # Plot dashboard
        dash_path = step_dashboard(args.output_dir, population, novelty[: len(population)], title=f"Generation {gen}", show=args.show_dashboard)
        print(f"Generation {gen} dashboard:", dash_path)
        pause(f"Completed generation {gen}.")

    print("Done. Outputs in:", args.output_dir)


if __name__ == "__main__":
    main()


