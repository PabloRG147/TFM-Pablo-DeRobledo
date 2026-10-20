import numpy as np
from qiskit.primitives import BackendEstimatorV2


def seed_for(base_seed, shots, budget, run, stream=0):
    """Genera una semilla reproducible para una configuración."""
    return int(
        np.random.SeedSequence(
            [base_seed, shots, budget, run, stream]
        ).generate_state(1)[0]
    )


class ReseedingEstimator:
    """Utiliza una semilla diferente en cada evaluación."""

    def __init__(self, backend, shots, seed):
        self.rng = np.random.default_rng(seed)

        self.estimator = BackendEstimatorV2(
            backend=backend,
            options={
                "default_precision": 1 / np.sqrt(shots)
            }
        )

    def run(self, pubs, **kwargs):
        evaluation_seed = int(
            self.rng.integers(1, 2**32 - 1)
        )

        self.estimator.options.seed_simulator = evaluation_seed

        return self.estimator.run(
            pubs=pubs,
            **kwargs
        )
