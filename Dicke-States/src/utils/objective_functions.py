# Imports
from __future__ import annotations

import numpy as np

from optuna import Trial
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2
from typing import List


def objective_scipy(
    x: np.ndarray,
    estimator: EstimatorV2,
    qc: QuantumCircuit,
    ising: SparsePauliOp,
    offset: float,
    callback: list = None,
) -> float:
    """Creates an objective function to be
    used with scipy.optimize.minimize.

    Args:
        x (np.ndarray): The parameters given by the optimizer.
        estimator: (EstimatorV2): An object of type EstimatorV2 to
        compute Hamiltonian expectation value.
        qc (QuantumCircuit): A parameterized quantum circuit that
        represents the ansatz.
        ising (SparsePauliOp): The Ising model that encodes the
        optimization problem.
        offset (float): A constant that represents the offset between
        Hamiltonian expectation value and cost function value.
        callback (list, optional): A list to capture the values of
        objective function and parameters for each iteration. Defaults to None.

    Returns: Objective function value.
    """

    ansatz = qc.copy()
    params_mapper = {param: value for param, value in zip(ansatz.parameters, x)}
    ansatz = ansatz.assign_parameters(parameters=params_mapper)
    pubs = [(ansatz, ising)]
    estimator_job = estimator.run(pubs=pubs)
    estimator_result = estimator_job.result()
    obj_val = estimator_result[0].data.evs + offset

    if not callback is None:
        callback.append((obj_val, params_mapper))

    return obj_val


class ObjectiveOptuna:
    """ObjectiveOptuna class"""

    def __init__(
        self,
        estimator: EstimatorV2,
        qc: QuantumCircuit,
        ising: SparsePauliOp,
        offset: float,
        bounds: List[tuple] = None,
        callback: list = None,
    ) -> ObjectiveOptuna:
        """Initializes the class ObjectiveOptuna.

        Args:
            estimator (EstimatorV2): An object of type EstimatorV2 to
            compute Hamiltonian expectation value.
            qc (QuantumCircuit): A parameterized quantum circuit that
            represents the ansatz.
            ising (SparsePauliOp): The Ising model that encodes the
            optimization problem.
            offset (float): A constant that represents the offset between
            Hamiltonian expectation value and cost function value.
            bounds (tuple, optional): A list of tuples with parameters bounds. Defaults to None.
            callback (list, optional): A list to capture the values of
            objective function and parameters for each iteration. Defaults to None.

        Returns: An object of type ObjectiveOptuna.
        """
        self.estimator = estimator
        self.qc = qc
        self.ising = ising
        self.offset = offset
        if bounds is None:
            self.bounds = [(0, 2 * np.pi) for _ in range(qc.num_parameters)]
        else:
            self.bounds = bounds
        self.callback = callback

    def __call__(self, trial: Trial) -> float:
        """Call magic method of Objective Optuna.

        Args:
            trial (Trial): optuna Trial object.

        Returns: Objective function value.
        """

        ansatz = self.qc.copy()
        x = [
            trial.suggest_float(f"x[{idx}]", tp[0], tp[1])
            for idx, tp in enumerate(self.bounds)
        ]
        params_mapper = {param: value for param, value in zip(ansatz.parameters, x)}
        ansatz = ansatz.assign_parameters(parameters=params_mapper)
        pubs = [(ansatz, self.ising)]
        estimator_job = self.estimator.run(pubs=pubs)
        estimator_result = estimator_job.result()
        obj_val = estimator_result[0].data.evs + self.offset

        if not self.callback is None:
            self.callback.append((obj_val, params_mapper))

        return obj_val
