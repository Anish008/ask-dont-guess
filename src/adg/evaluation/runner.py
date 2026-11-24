"""Experiment runner.

Per spec §12.2: Grid search over models x methods x datasets x seeds.
"""

from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

from adg.core.logging import get_logger
from adg.data.schema import Example
from adg.evaluation.e2e import E2EResult, run_e2e_evaluation
from adg.methods.base import Method
from adg.simuser.base import SimulatedUser

log = get_logger(__name__)


class ExperimentConfig(BaseModel):
    """Experiment configuration.

    Args:
        name: Experiment name
        methods: Method names to evaluate
        models: Model names to use
        datasets: Dataset names
        seeds: Random seeds
        max_turns: Maximum clarification turns
        output_dir: Output directory for results
    """

    name: str
    methods: list[str]
    models: list[str]
    datasets: list[str]
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    max_turns: int = 3
    output_dir: Path = Field(default_factory=lambda: Path("results/runs"))


class ExperimentResult(BaseModel):
    """Experiment result.

    Args:
        config: Experiment configuration
        results: E2E results per example
        completed: Whether experiment completed
    """

    config: ExperimentConfig
    results: list[E2EResult] = Field(default_factory=list)
    completed: bool = False


class ExperimentRunner:
    """Experiment runner for grid search.

    Per spec §12.2: Runs methods over datasets with seeds,
    caches reference sampling, resumable.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        simulated_user: SimulatedUser,
    ):
        """Initialize experiment runner.

        Args:
            config: Experiment configuration
            simulated_user: Simulated user for E2E evaluation
        """
        self.config = config
        self.simuser = simulated_user
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "experiment_runner_initialized",
            name=config.name,
            methods=config.methods,
            seeds=config.seeds,
        )

    def run(
        self,
        methods: dict[str, Method],
        examples: list[Example],
    ) -> ExperimentResult:
        """Run experiment grid.

        Per spec §12.2: Methods x datasets x seeds.

        Args:
            methods: Method instances by name
            examples: Test examples

        Returns:
            Experiment results
        """
        results: list[E2EResult] = []

        # Grid: methods x examples x interpretations
        total_runs = sum(
            len(ex.interpretations) if ex.interpretations else 1 for ex in examples
        ) * len(methods)

        with tqdm(total=total_runs, desc="Running E2E evaluation") as pbar:
            for method_name in self.config.methods:
                if method_name not in methods:
                    log.warning("method_not_found", method=method_name)
                    continue

                method = methods[method_name]

                for example in examples:
                    # For ambiguous examples, test each interpretation
                    if example.is_ambiguous and example.interpretations:
                        for interpretation in example.interpretations:
                            result = run_e2e_evaluation(
                                method=method,
                                example=example,
                                intended_interpretation=interpretation,
                                simulated_user=self.simuser,
                                max_turns=self.config.max_turns,
                            )
                            results.append(result)
                            pbar.update(1)
                    else:
                        # Clear example: use first interpretation
                        if not example.interpretations:
                            log.warning(
                                "no_interpretations",
                                example_id=example.example_id,
                            )
                            continue

                        result = run_e2e_evaluation(
                            method=method,
                            example=example,
                            intended_interpretation=example.interpretations[0],
                            simulated_user=self.simuser,
                            max_turns=self.config.max_turns,
                        )
                        results.append(result)
                        pbar.update(1)

        return ExperimentResult(
            config=self.config,
            results=results,
            completed=True,
        )

    def save_results(self, result: ExperimentResult) -> Path:
        """Save experiment results.

        Args:
            result: Experiment result

        Returns:
            Path to saved results
        """
        output_file = self.output_dir / f"{self.config.name}_results.json"

        with output_file.open("w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        log.info("results_saved", path=str(output_file))
        return output_file


__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
]
