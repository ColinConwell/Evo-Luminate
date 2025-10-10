import json
import random
from typing import Dict, List, Any, Optional, Union


class CreativityStrategyManager:
    """Manages loading and converting creativity strategies to prompts."""

    def __init__(self, json_file_path: str):
        """
        Initialize with the path to the JSON file containing creativity strategies.

        Args:
            json_file_path: Path to the JSON file with strategies
        """
        self.strategies = []
        self.load_strategies(json_file_path)

    def load_strategies(self, json_file_path: str) -> None:
        """
        Load creativity strategies from a JSON file.

        Args:
            json_file_path: Path to the JSON file with strategies
        """
        # try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
            self.strategies = data.get("strategies", [])
            print(f"Loaded {len(self.strategies)} creativity strategies")
        # except (json.JSONDecodeError, FileNotFoundError) as e:
        #     print(f"Error loading strategies: {e}")
        #     self.strategies = []

    def get_strategy_by_name(self, name: str) -> Optional[Dict]:
        """
        Retrieve a strategy by its name.

        Args:
            name: Name of the strategy to retrieve

        Returns:
            Strategy dictionary or None if not found
        """
        for strategy in self.strategies:
            if strategy.get("name") == name:
                return strategy
        return None

    def get_random_strategy(self) -> Optional[Dict]:
        """
        Get a random strategy from the loaded set.

        Returns:
            Random strategy dictionary or None if no strategies are loaded
        """
        if not self.strategies:
            return None
        return random.choice(self.strategies)

    def to_prompt(
        self,
        strategy: Union[str, Dict],
        include_theory: bool = False,
        include_example: bool = False,
    ) -> str:
        """
        Convert a strategy to a prompt for a language model.

        Args:
            strategy: Strategy name or dictionary
            include_theory: Whether to include theoretical background
            include_example: Whether to include an example

        Returns:
            Formatted prompt string
        """
        # If strategy is a string (name), get the full strategy dictionary
        if isinstance(strategy, str):
            strategy_dict = self.get_strategy_by_name(strategy)
            if not strategy_dict:
                return f"Error: Strategy '{strategy}' not found"
        else:
            strategy_dict = strategy

        # Build the prompt
        # prompt = f"# Creative Thinking Task: {strategy_dict['name']}\n\n"

        # prompt = f"Creative Strategy: \n{strategy_dict['description']}\n\n"
        prompt = ""
        if include_theory:
            prompt += f"## Theory Base\n{strategy_dict['theory_base']}\n\n"

        prompt += (
            "Follow these steps to think creatively. Do not include in the output:\n\n"
        )

        # Add operations as numbered steps
        for i, operation in enumerate(strategy_dict["operations"]):
            # prompt += f"{i+1}. **{operation['name']}**: {operation['instruction']}\n"
            prompt += f"{i+1}. {operation['instruction']}\n"

        prompt += "\n"

        # Add the example if requested
        if include_example:
            prompt += f"## Example\n{strategy_dict['example']}\n\n"

        return prompt

    def mutate_strategy(self, strategy: Dict, mutation_rate: float = 0.3) -> Dict:
        """
        Create a mutated version of a strategy.

        Args:
            strategy: The strategy to mutate
            mutation_rate: Probability of each component being mutated

        Returns:
            A new strategy with mutations
        """
        # Create a deep copy of the strategy to avoid modifying the original
        import copy

        new_strategy = copy.deepcopy(strategy)

        # Potentially mutate the description
        if random.random() < mutation_rate:
            descriptions = [s["description"] for s in self.strategies]
            if len(descriptions) > 1:
                # Avoid selecting the same description
                other_descriptions = [
                    d for d in descriptions if d != strategy["description"]
                ]
                if other_descriptions:
                    new_strategy["description"] = random.choice(other_descriptions)

        # Potentially mutate operations
        if random.random() < mutation_rate:
            # 50% chance to add a new operation from another strategy
            if random.random() < 0.5 and len(self.strategies) > 1:
                other_strategies = [
                    s for s in self.strategies if s["name"] != strategy["name"]
                ]
                if other_strategies:
                    donor_strategy = random.choice(other_strategies)
                    if donor_strategy["operations"]:
                        new_op = random.choice(donor_strategy["operations"])
                        # Avoid duplicates
                        if new_op not in new_strategy["operations"]:
                            insert_pos = random.randint(
                                0, len(new_strategy["operations"])
                            )
                            new_strategy["operations"].insert(insert_pos, new_op)

            # 50% chance to remove an operation (if there are enough)
            elif len(new_strategy["operations"]) > 2 and random.random() < 0.5:
                remove_idx = random.randint(0, len(new_strategy["operations"]) - 1)
                new_strategy["operations"].pop(remove_idx)

            # Otherwise reorder operations
            else:
                random.shuffle(new_strategy["operations"])

        # Potentially mutate parameters
        for param in new_strategy["parameters"]:
            if random.random() < mutation_rate:
                # For numeric parameters, adjust by up to ±30%
                if isinstance(new_strategy["parameters"][param], (int, float)):
                    adjustment = 1.0 + (random.random() * 0.6 - 0.3)  # -30% to +30%
                    new_value = new_strategy["parameters"][param] * adjustment

                    # Keep values in reasonable bounds
                    if isinstance(new_strategy["parameters"][param], int):
                        new_strategy["parameters"][param] = max(1, int(new_value))
                    else:
                        new_strategy["parameters"][param] = max(
                            0.1, min(1.0, new_value)
                        )

        # Give the mutated strategy a new name
        new_strategy["name"] = f"{strategy['name']} (Mutated)"

        return new_strategy

    def crossover_strategies(self, strategy1: Dict, strategy2: Dict) -> Dict:
        """
        Create a new strategy by crossing over elements of two parent strategies.

        Args:
            strategy1: First parent strategy
            strategy2: Second parent strategy

        Returns:
            A new strategy with elements from both parents
        """
        import copy

        child = copy.deepcopy(strategy1)  # Start with a copy of the first parent

        # Create a hybrid name
        child["name"] = f"Hybrid: {strategy1['name']} × {strategy2['name']}"

        # Combine theory bases
        child["theory_base"] = (
            f"Combination of {strategy1['theory_base']} and {strategy2['theory_base']}"
        )

        # Blend descriptions
        child["description"] = (
            f"This strategy combines elements of both {strategy1['name']} and {strategy2['name']}. "
        )
        child[
            "description"
        ] += f"From {strategy1['name']}: {strategy1['description'][:50]}... "
        child[
            "description"
        ] += f"From {strategy2['name']}: {strategy2['description'][:50]}..."

        # Operations: take some from each parent
        # First half from parent 1, second half from parent 2
        split_point = len(strategy1["operations"]) // 2
        child["operations"] = strategy1["operations"][:split_point]

        # Add non-duplicate operations from parent 2
        for op in strategy2["operations"]:
            if op not in child["operations"]:
                child["operations"].append(op)

        # Combine parameters from both parents
        for param in strategy2["parameters"]:
            if param not in child["parameters"]:
                child["parameters"][param] = strategy2["parameters"][param]
            else:
                # Average numeric parameters
                if isinstance(child["parameters"][param], (int, float)) and isinstance(
                    strategy2["parameters"][param], (int, float)
                ):
                    child["parameters"][param] = (
                        child["parameters"][param] + strategy2["parameters"][param]
                    ) / 2

        # Create a new example that references both parent strategies
        child["example"] = (
            f"This hybrid approach could be applied as follows: {strategy1['example'][:50]}... but with elements from {strategy2['name']}: {strategy2['example'][:50]}..."
        )

        return child


# Example usage
if __name__ == "__main__":
    manager = CreativityStrategyManager("eluminate/creativity_strategies.json")

    # Get a specific strategy
    # replacement = manager.get_strategy_by_name("Replacement Template")
    # if replacement:
    #     prompt = manager.to_prompt(replacement, subject="A new type of educational app")
    #     print(prompt)
    #     print('#'*80)

    # Get a random strategy
    random_strategy = manager.get_random_strategy()
    if random_strategy:
        prompt = manager.to_prompt(random_strategy)
        print("#" * 80)
        print(prompt)
        print("#" * 80)

        # # Create a mutation
        # mutated = manager.mutate_strategy(random_strategy)
        # print(f"Original strategy: {random_strategy['name']}")
        # print(f"Mutated strategy: {mutated['name']}")
        # print(
        #     f"Changed operations count: {len(random_strategy['operations'])} → {len(mutated['operations'])}"
        # )
        # print('#'*80)

        # # Demonstrate crossover if we have at least 2 strategies
        # if len(manager.strategies) >= 2:
        #     strategy1 = manager.strategies[0]
        #     strategy2 = manager.strategies[1]
        #     hybrid = manager.crossover_strategies(strategy1, strategy2)
        #     print(f"Created hybrid strategy: {hybrid['name']}")
        #     print(f"Operations count: {len(hybrid['operations'])}")
        #     prompt = manager.to_prompt(hybrid)
        #     print(prompt)
