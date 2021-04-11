import cirq
import numpy as np
from typing import List, Tuple
from matrices import two_level_decomp
from gates import create_gates_from_gray
from bridge import breakdown_multiple_control_op_to_convertible_gate_op
from sycamore import convert_ops_to_sycamore

# ###################################################################################################
def matrix_to_sycamore_operations(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
        .
    """

    num_qubits = len(target_qubits)
    
    # Step 1: Decompose unitary to two-level matrices
    matrices, indices_list = two_level_decomp(matrix)

    # Step 2: Create Fully-controlled gates
    all_gates = []
    for matrix, indices in zip(matrices, indices_list):
        all_gates += create_gates_from_gray(matrix, indices, num_qubits)
        
    # Step 3: Create cirq operations
    ops = [gate.build_op(target_qubits) for gate in all_gates]

    # Step 4: Convert multiple-control gates to convertable gates
    ops_conv = []
    for op in ops:
        ops_simple = breakdown_multiple_control_op_to_convertible_gate_op(op)
        ops_simple.extend()

    # Step 5: Convert "convertible gates" to sycamore-specific gates
    ops_syc = convert_ops_to_sycamore(ops_conv)

    return ops_syc, []
