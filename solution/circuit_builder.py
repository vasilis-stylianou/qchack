import cirq
import numpy as np

def convert_gatelist_to_cirq_ops(gatelist, target_qubits):
    # Implement once we know the exact input.
    ops = []
    # Index is wrt the factorized matrix
    for gate, indices in gatelist:
        if len(indices) == 1:
            q1 = target_qubits[indices[0]]
            ops.append(gate(q1))
        elif len(indices) == 2:
            q1, q2 = target_qubits[indices[0]], target_qubits[indices[1]]

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
            NotImplementedError("Gates larger than 2 qbits are not implemented.")

    return ops

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


def factorized_gate_list_to_listOps(
        factorized_gate_list,
        # Of the form [[cirq.H, [qx, qy]], [gate_1q, [qx]], ... ]
        target_qubits,
        device = cirq.google.Sycamore,
):
    """
    Converts the factorized list of gates (1q and 2q) into a cirq circuit (non-optimized).
    :param factorized_gate_list:
    :param device:
    :return:
    """
    # Create a circuit on the device
    circuit = cirq.Circuit(device=device)

    # Convert list of gates to cirq operations
    ops = convert_gatelist_to_cirq_ops(factorized_gate_list, target_qubits)

    # Convert incompatible gates to compatible
    ops_comp = convert_to_compatible_gates(ops, device)

    # When we append operations now, they are put into different moments.
    circuit.append(ops_comp)

    # Optimise circuit for Sycamore
    circuit = cirq.google.optimized_for_sycamore(circuit)

    print(circuit)
    return ops_comp



factorized_gate_list = [[cirq.CZ,[0,1]], [cirq.CZ, [2,0]],
                        [cirq.CZ, [3,1]], [cirq.CZ, [2,3]], [cirq.H, [3]]]
target_qubits = [cirq.GridQubit(4,i) for i in range(1,9)]
ops = factorized_gate_list_to_listOps(factorized_gate_list, target_qubits)
print(ops)
print(len(ops))