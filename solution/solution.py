import cirq
import numpy as np
from typing import List, Tuple

# ###################################################################################################
# MAIN
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
        
    # Step 3: Create cirq operations from FC gates
    ops = [gate.build_op(target_qubits) for gate in all_gates]
    
    # Step 4: Simplify FC operations down to 1/2-qubit ops
    simplified_ops = []
    for op in ops:
        simplified_ops += simplify_fc_op(op)

    # Convert "convertible gates" to sycamore-specific gates
    try:
        syc_ops = convert_ops_to_sycamore(simplified_ops, account_for_geometry=True)
    except:
        syc_ops = convert_ops_to_sycamore(simplified_ops, account_for_geometry=False)
    

    return syc_ops, []


# ###################################################################################################
# Operation Methods
# ###################################################################################################
def simplify_fc_op(fc_op):

    op = fc_op
    sub_gate = op.gate.sub_gate
    matrix = cirq.unitary(sub_gate)

    control_values = op.gate.control_values
    ordered_qubits = list(op.qubits)
    control_qubits, target_qubit = ordered_qubits[:-1], ordered_qubits[-1]

    # Check if control is 0 valued. If so, add an X gate before and after
    x_gates = []
    for i, val in enumerate(control_values):
        if val == (0,):
            x = cirq.X(control_qubits[i])
            x_gates.append(x)

    ops_decomp = cirq.decompose_multi_controlled_rotation(matrix=matrix, 
                                                          controls=control_qubits, 
                                                          target=target_qubit, )

    # Reverse x gates
    ops_decomp_xs = []
    ops_decomp_xs.extend(x_gates)
    ops_decomp_xs.extend(ops_decomp)
    ops_decomp_xs.extend(x_gates)

    return ops_decomp_xs

# ###################################################################################################
# Sycamore Operations
# ###################################################################################################
def convert_gatelist_to_cirq_ops(operation_list):

    ops = []
    for op in operation_list:
        gate = op.gate
        target_qubits = op._qubits

        if len(target_qubits) == 1:
            q1 = target_qubits[0]
            ops.append(gate(q1))
        elif len(target_qubits) == 2:
            q1, q2 = target_qubits[0], target_qubits[1]

            # Check if SWAP is necessary
            # Check if q1 and q2 are neighbours
            if not q1.is_adjacent(q2):
                swap_operations, new_q2 = get_swap_qbits_list(q1, q2)
                ops.extend(swap_operations)
                ops.append(gate(q1, new_q2))
                ops.extend(swap_operations[::-1])
            else:
                ops.append(gate(q1, q2))
        else:
            # 3 qbit gate (no time to implement)
            ops.append(op)

    return ops

# ===================================================================================================
def get_swap_qbits_list(q1:cirq.GridQubit, q2:cirq.GridQubit):
    """
    Calculates the swap operations to bring q2 to neighbouring of q1.
    Returns the list of operations required to reach neigbourhood.
    Returns also the new cirq.GridQubit position of the moved q2.
    :param q1: cirq.GridQubit
    :param q2: cirq.GridQubit
    :return:
    """
    q1_coord = [q1._row, q1._col]
    q2_coord = [q2._row, q2._col]

    # Get the indices for each swap.
    # Returns [[[5,0],[5,1]], [[5,1],[5,2]] ... ]
    short_path_steps = get_short_path(q1_coord, q2_coord)

    # Create operations list based on the steps/jumps needed
    operations_list = []

    for step in short_path_steps:
        # step -> [[5,0], [5,1]]
        qi, qj = cirq.GridQubit(*step[0]), cirq.GridQubit(*step[1])
        swap = cirq.SWAP(qi, qj)
        operations_list.append(swap)

    # Get the new q2 position (the last value of the last swap)
    new_q2_coord = short_path_steps[-1][-1]
    new_q2_position = cirq.GridQubit(*new_q2_coord)
    return operations_list, new_q2_position

# ===================================================================================================
def get_short_path(coord2, coord1):
    """
    coords q1 = [4,3],    q2 = [4,1]
    Calculates the shortest path within a device structure.
    Returns an array with the series of swaps required.
    :param coord1:
    :param coord2:
    :return:
    """
    final = []
    path = [coord1]
    diff = np.subtract(coord1, coord2)

    if abs(diff[0]) != 0:
        for i in range(abs(diff[1])):
            if diff[0] < 0:
                step = np.add(path[-1], [1, 0]).tolist()
                path.append(step)

            elif diff[0] > 0:
                step = np.subtract(path[-1], [1, 0]).tolist()
                path.append(step)

            else:
                pass
    else:
        pass

    if abs(diff[1]) != 0:
        for i in range(abs(diff[1])):
            if diff[1] < 0:
                step = np.add(path[-1], [0, 1]).tolist()
                path.append(step)

            elif diff[1] > 0:
                step = np.subtract(path[-1], [0, 1]).tolist()
                path.append(step)

            else:
                pass
    else:
        pass

    for i in range(len(path) - 2):
        final.append([path[i], path[i + 1]])

    return final

# ===================================================================================================
def convert_to_compatible_gates(operations, device):
    """
    Checks if operations are not compatible and converts
    them to compatible gates for a given device.
    :param operations:
    :param device:
    :return:
    """
    if device == cirq.google.Sycamore:
        converter = cirq.google.ConvertToSycamoreGates()
    else:
        NotImplementedError("Code only compatible for Sycamore hardware.")

    ops_compatible = []
    for i, op in enumerate(operations):
        # Check if X is supported.
        sycamore_gates = cirq.google.gate_sets.SYC_GATESET
        if not sycamore_gates.is_supported_operation(op):
            """Convert a gate to sycamore gates."""
            # Do the conversion.
            op_conv = converter.convert(op)
            ops_compatible.extend(op_conv)
        else:
            ops_compatible.append(op)
    return ops_compatible

# ===================================================================================================
def convert_ops_to_sycamore(
        operation_list, account_for_geometry,
        device = cirq.google.Sycamore, create_circuit=False,
):

    if account_for_geometry:
        # Convert operation to gates to cirq operations
        ops_converted = convert_gatelist_to_cirq_ops(operation_list, account_for_geometry)
    else:
        ops_converted = operation_list

    # Convert incompatible gates to compatible
    ops_comp = convert_to_compatible_gates(ops_converted, device)

    if create_circuit:
        # Create a circuit on the device
        circuit = cirq.Circuit(device=device)
        # When we append operations now, they are put into different moments.
        circuit.append(ops_comp)
        # Optimise circuit for Sycamore
        circuit = cirq.google.optimized_for_sycamore(circuit)
        print(circuit)

    return ops_comp


# ###################################################################################################
# Gate Methods
# ###################################################################################################
class FCGate:

    # ----------------------------------------------------------------------------------------------- 
    def __init__(self, control_0, control_1, target, sub_gate):
        
        self.control_0 = control_0
        self.control_1 = control_1
        self.target = target
        self.sub_gate = sub_gate
                
    # ----------------------------------------------------------------------------------------------- 
    def build_op(self, qubits):
        
        # Create Fully Controlled Gate
        num_controls = len(self.control_0) + len(self.control_1)
        control_values = [0] * len(self.control_0) + [1] * len(self.control_1) 
        fc_gate = cirq.ControlledGate(self.sub_gate, num_controls, control_values)

        # Create ordered qubit list for cirq.ControlledGate: [c0,...,c1,...,target]
        control_0_qubits = [qubits[i] for i in self.control_0]
        control_1_qubits = [qubits[i] for i in self.control_1]
        target_qubit = [qubits[self.target]]
        ordered_qubits = control_0_qubits + control_1_qubits + target_qubit

        return fc_gate.on(*ordered_qubits) 

# ===================================================================================================
def dec_to_bin(x, num_qubits):
    """
    x - decimal number to convert to binary
    num_qubits - number of qubits
    """

    dec_string = bin(x).replace("0b","")[::-1]
    x_bin = np.zeros(num_qubits, dtype='int')

    j = num_qubits - 1
    for i in dec_string:
        x_bin[j] = int(i)
        j = j -1

    return x_bin

# ===================================================================================================
def create_gray_arrs(indices, num_qubits):
    
    """
    Return grey code connecting indx1 to indx2 for an n qubit unitary
    indices
    num_qubits - number of qubits
    """
    
    b1 = dec_to_bin(indices[0], num_qubits)
    b2 = dec_to_bin(indices[1], num_qubits)
    s = np.copy(b1)
    
    code = np.array([b1])

    for i in reversed(range(num_qubits)):

        if s[i] != b2[i]:
            
            s[i] =  (s[i] + 1) % 2
            code = np.append(code, [s], axis=0)
        
    return code

# ===================================================================================================
def state1_to_state2_gate(arr1, arr2, sub_gate):
    N = len(arr1)
    
    control_0 = []
    control_1 = []
    target = None
    for i in range(N):
        
        # if ith qubits have same state -> control qubits
        if arr1[i] == arr2[i]:
            if arr1[i] == 0:
                control_0.append(i)
            else:
                control_1.append(i)
        else:
            target = i
        
    return FCGate(control_0, control_1, target, sub_gate)

# ===================================================================================================
def sub_gate_from_two_level_matrix(matrix, atol=0):

    """
    Return a Phase XZ decomposition of a 2x2 unitary matrix
    matrix - 2x2 unitary matrix
    atol - limit on the amount of error introduced by the construction.
    """

    sub_gate = cirq.optimizers.single_qubit_matrix_to_phxz(matrix, atol)

    if sub_gate != None:
        return sub_gate

    else:
        return cirq.I
        #TODO: Decide how to handle identity case
        # print("Identity")

# ===================================================================================================
def create_gates_from_gray(matrix, indices, num_qubits) -> list:
    """
    This is our main method
    """
    
    gray_arrs = create_gray_arrs(indices, num_qubits)
    N = len(gray_arrs)
    
    # Pre-ops: Apply FC_X gates   
    pre_gates = [state1_to_state2_gate(gray_arrs[i], gray_arrs[i+1], cirq.X) 
                 for i in range(N-2)]

    # Main op: Apply FC_U gate
    sub_gate = sub_gate_from_two_level_matrix(matrix)
    main_gate = [state1_to_state2_gate(gray_arrs[N-2], gray_arrs[N-1], sub_gate)]

    # Post-ops: Reverse pre-ops
    post_gates = list(gate for gate in pre_gates[::-1])

    return pre_gates + main_gate + post_gates

import numpy as np

# ###################################################################################################
# Matrix Methods
# ###################################################################################################
def is_unitary(A):
    n = A.shape[0]
    if (A.shape != (n, n)):
        raise ValueError("Matrix is not square.")
    A = np.array(A)
    return np.allclose(np.eye(n), A @ A.conj().T)

# ===================================================================================================
def is_identity(A):
    n = A.shape[0]
    if (A.shape != (n, n)):
        raise ValueError("Matrix is not square.")
    return np.allclose(A, np.eye(n))

# ===================================================================================================
def elimination_matrix(a,b):
    # a, b allowed to be complex
    
    # impose theta real + positive {eq.10}
    theta = np.arctan(abs(b/a))
    
    # lambda is the negative arg() of a
    lamda = - np.angle(a)
    
    # {eq.12}
    mu = np.pi + np.angle(b)
    
    # {eq.7}
    U_special = np.array([ [np.exp(1j*lamda) * np.cos(theta), np.exp(1j*mu) * np.sin(theta)],
                           [-np.exp(-1j*mu) * np.sin(theta), np.exp(-1j*lamda) * np.cos(theta)] ])
    
    return U_special

# ===================================================================================================
def two_level_decomp(A, thr=10**-9):
    
    """
    decomp - list of 2x2 unitaries that decompose A. To reconstruct A, the list of unitaries
             in decomp must be reversed
             
    indices - list of non-trivial rows on which matrices in decomp act. For example, [1,3] indicates
              that row 1 and 3 are non-trivial; all other elements are 1.
    """
    
    n = A.shape[0]
    decomp = []
    indices = []
    A_c = np.copy(A)

    for i in range(n-2):
        for j in range(n-1, i, -1):

            a = A_c[i,j-1]
            b = A_c[i,j]

            # --- need checks --- 
            # if A[i,j] = 0, nothing to do! Except in last row - need to check diagonal element is 1 
            if abs(A_c[i,j]) < thr:
                U_22 = np.eye(2, dtype=np.complex128)  # Identity_22

                if j == i+1:
                    U_22 = np.array([[1 / a, 0], [0, a]])

            # if A[i,j-1] = 0, need to swap columns - again checking last row to ensure diagonal element is 1 
            elif abs(A_c[i,j-1]) < thr:
                U_22 = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # Pauli_X

                if j == i+1:
                    U_22 = np.array([[1 / b, 0], [0, b]])

            # Special unitary matrix
            else: 
                U_22 = elimination_matrix(a,b)

            # ----- U_22 found -----

            # multiply submatrix of A with U_22
            A_c[:,(j-1,j)] = A_c[:,(j-1,j)] @ U_22

            # If not the identity matrix - represents a gate! So should store
            if not is_identity(U_22):
                decomp.append(U_22.conj().T)
                indices.append(np.array([j-1,j]))


        # check for diagonal element equal to 1
        # assert np.allclose(A_c[i,i],1.0)
    
    # lower right hand 2x2 matrix remaining after decomp
    lower_rh_matrix = A_c[n-2:n, n-2:n]
    
    # if not equal to I - is a non trivial gate
    if not is_identity(lower_rh_matrix):
        decomp.append(lower_rh_matrix)
        indices.append(np.array([n-2,n-1]))

    return decomp, indices