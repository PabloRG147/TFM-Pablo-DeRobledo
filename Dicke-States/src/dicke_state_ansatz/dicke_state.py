from __future__ import annotations


from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit, QuantumRegister


class DickeStateAnsatz:
    """DickeStateAnsatz class"""

    def __init__(self) -> DickeStateAnsatz:
        """
        Initializes DickeStateAnsatz class.
        """
        pass

    @staticmethod
    def generate_quantum_circuit(
        n: int,
        k: int,
        measurement: bool = True,
        add_barriers: bool = False,
        param_name: str = "theta",
    ) -> QuantumCircuit:
        """Generates the quantum circuit that produces
        a parameterized Dicke state with n qubits and
        Hamming weight k.

        Args:
            n (int): The number of qubits.
            k (int): State Hamming weight.
            measurement (bool, optional): A boolean flag to
            enable/disable measurements in the circuit.
            Defaults to True.
            add_barriers (bool, optional): A boolean flag to
            add/remove barriers in the circuit. Defaults to
            False.
            param_name (str, optional): The parameter name.
            Defaults to 'theta'.

        Raises:
            ValueError: if n, k or n and k types are not equal to int.
            ValueError: if n is a negative integer or zero.
            ValueError: if k is a negative integer.
            ValueError: if k is greater than n.

        Returns: An object of QuantumCircuit type which
        implements a parameterized Dicke state with n qubits
        and Hamming weight k.
        """

        # Checks if n and k types are int.
        if not isinstance(n, int) or not isinstance(k, int):
            raise ValueError("n and k must be an integer number!")

        # Checks if n is an integer number equal or greater than 1.
        if not n >= 1:
            raise ValueError("n must be equal or greater than one!")

        # Checks if k is a positive integer number.
        if not k >= 0:
            raise ValueError("k must be equal or greater than zero!")

        # Checks if k is greater than n.
        if k > n:
            raise ValueError("k can't be greater than n!")

        qubits = QuantumRegister(size=n, name="qubits")
        if measurement:
            bits = ClassicalRegister(size=n, name="bits")
            qc = QuantumCircuit(qubits, bits)
        else:
            qc = QuantumCircuit(qubits)

        # k = 0 the quantum state is |0>^{\otimes n}
        if k == 0:
            return qc
        elif n == k:
            # k = n the quantum state is |1>^{\otimes n}
            qc.x(qubit=qubits)
            return qc
        else:
            # Flipping k qubits from the n available in
            # the quantum circuit.
            qc.x(qubit=qubits[[i for i in range(k)]])
            if add_barriers:
                qc.barrier()

            # This counter was introduced to solve the
            # error when parameters have the same name,
            # then we use the counter to generate unique
            # parameters names.
            counter = 0

            # For loops for the creation of parameterized
            # SCS gates.
            for i in range(k)[::-1]:
                for j in range(i, n - 1):
                    param = Parameter(rf"$\{param_name}$[{counter}]")
                    qc.cx(control_qubit=qubits[j], target_qubit=qubits[j + 1])
                    qc.cry(
                        theta=param, control_qubit=qubits[j + 1], target_qubit=qubits[j]
                    )
                    qc.cx(control_qubit=qubits[j], target_qubit=qubits[j + 1])
                    counter += 1
                if add_barriers:
                    qc.barrier()

            if measurement:
                qc.measure(qubit=qubits, cbit=bits)

            return qc
