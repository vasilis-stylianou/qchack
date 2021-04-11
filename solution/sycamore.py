import cirq

import cirq
import numpy as np

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







operation_list = [cirq.PhasedXPowGate(phase_exponent=0.5678998743900456, exponent=0.5863459345743176).on(cirq.GridQubit(4, 1)),
 cirq.PhasedXPowGate(phase_exponent=0.3549946157441739).on(cirq.GridQubit(4, 2)),
 cirq.google.SYC(cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(cirq.GridQubit(4, 1)),
 cirq.google.SYC(cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.5987667922766213, exponent=0.4136540654256824).on(cirq.GridQubit(4, 1)),
 (cirq.Z**-0.9255092746611595).on(cirq.GridQubit(4, 2)),
 (cirq.Z**-1.333333333333333).on(cirq.GridQubit(4, 1)),
 cirq.PhasedXPowGate(phase_exponent=0.44650378384076217, exponent=0.8817921214052824).on(cirq.GridQubit(4, 1)),
 cirq.PhasedXPowGate(phase_exponent=-0.7656774060816165, exponent=0.6628666504604785).on(cirq.GridQubit(4, 2)),
 cirq.google.SYC(cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.6277589946716742, exponent=0.5659160932099687).on(cirq.GridQubit(4, 1)),
 cirq.google.SYC(cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=0.28890767199499257, exponent=0.4340839067900317).on(cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.22592784059288928).on(cirq.GridQubit(4, 1)),
 cirq.google.SYC(cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.4691261557936808, exponent=0.7728525693920243).on(cirq.GridQubit(4, 1)),
 cirq.PhasedXPowGate(phase_exponent=-0.8150261316932077, exponent=0.11820787859471782).on(cirq.GridQubit(4, 2)),
 (cirq.Z**-0.7384700844660306).on(cirq.GridQubit(4, 2)),
 (cirq.Z**-0.7034535141382525).on(cirq.GridQubit(4, 1)),
 cirq.PhasedXPowGate(phase_exponent=0.5678998743900456, exponent=0.5863459345743176).on(cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=0.3549946157441739).on(cirq.GridQubit(4, 2)),
 cirq.google.SYC(cirq.GridQubit(4, 3), cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(cirq.GridQubit(4, 3)),
 cirq.google.SYC(cirq.GridQubit(4, 3), cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.5987667922766213, exponent=0.4136540654256824).on(cirq.GridQubit(4, 3)),
 (cirq.Z**-0.9255092746611595).on(cirq.GridQubit(4, 2)),
 (cirq.Z**-1.333333333333333).on(cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=0.44650378384076217, exponent=0.8817921214052824).on(cirq.GridQubit(4, 1)),
 cirq.PhasedXPowGate(phase_exponent=-0.7656774060816165, exponent=0.6628666504604785).on(cirq.GridQubit(4, 2)),
 cirq.google.SYC(cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.6277589946716742, exponent=0.5659160932099687).on(cirq.GridQubit(4, 1)),
 cirq.google.SYC(cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=0.28890767199499257, exponent=0.4340839067900317).on(cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.22592784059288928).on(cirq.GridQubit(4, 1)),
 cirq.google.SYC(cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.4691261557936808, exponent=0.7728525693920243).on(cirq.GridQubit(4, 1)),
 cirq.PhasedXPowGate(phase_exponent=-0.8150261316932077, exponent=0.11820787859471782).on(cirq.GridQubit(4, 2)),
 (cirq.Z**-0.7384700844660306).on(cirq.GridQubit(4, 2)),
 (cirq.Z**-0.7034535141382525).on(cirq.GridQubit(4, 1)),
 cirq.PhasedXPowGate(phase_exponent=0.44650378384076217, exponent=0.8817921214052824).on(cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.7656774060816165, exponent=0.6628666504604785).on(cirq.GridQubit(4, 3)),
 cirq.google.SYC(cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=-0.6277589946716742, exponent=0.5659160932099687).on(cirq.GridQubit(4, 2)),
 cirq.google.SYC(cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=0.28890767199499257, exponent=0.4340839067900317).on(cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=-0.22592784059288928).on(cirq.GridQubit(4, 2)),
 cirq.google.SYC(cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=-0.4691261557936808, exponent=0.7728525693920243).on(cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.8150261316932077, exponent=0.11820787859471782).on(cirq.GridQubit(4, 3)),
 (cirq.Z**-0.7384700844660306).on(cirq.GridQubit(4, 3)),
 (cirq.Z**-0.7034535141382525).on(cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=0.5678998743900456, exponent=0.5863459345743176).on(cirq.GridQubit(4, 4)),
 cirq.PhasedXPowGate(phase_exponent=0.3549946157441739).on(cirq.GridQubit(4, 3)),
 cirq.google.SYC(cirq.GridQubit(4, 4), cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(cirq.GridQubit(4, 4)),
 cirq.google.SYC(cirq.GridQubit(4, 4), cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=-0.5987667922766213, exponent=0.4136540654256824).on(cirq.GridQubit(4, 4)),
 (cirq.Z**-0.9255092746611595).on(cirq.GridQubit(4, 3)),
 (cirq.Z**-1.333333333333333).on(cirq.GridQubit(4, 4)),
 cirq.PhasedXPowGate(phase_exponent=0.44650378384076217, exponent=0.8817921214052824).on(cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.7656774060816165, exponent=0.6628666504604785).on(cirq.GridQubit(4, 3)),
 cirq.google.SYC(cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=-0.6277589946716742, exponent=0.5659160932099687).on(cirq.GridQubit(4, 2)),
 cirq.google.SYC(cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=0.28890767199499257, exponent=0.4340839067900317).on(cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=-0.22592784059288928).on(cirq.GridQubit(4, 2)),
 cirq.google.SYC(cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=-0.4691261557936808, exponent=0.7728525693920243).on(cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=-0.8150261316932077, exponent=0.11820787859471782).on(cirq.GridQubit(4, 3)),
 (cirq.Z**-0.7384700844660306).on(cirq.GridQubit(4, 3)),
 (cirq.Z**-0.7034535141382525).on(cirq.GridQubit(4, 2)),
 cirq.PhasedXPowGate(phase_exponent=0.5678998743900456, exponent=0.5863459345743176).on(cirq.GridQubit(4, 3)),
 cirq.PhasedXPowGate(phase_exponent=0.3549946157441739).on(cirq.GridQubit(4, 4)),
 cirq.google.SYC(cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)),
 cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(cirq.GridQubit(4, 4)),
 cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(cirq.GridQubit(4, 3)),
 cirq.google.SYC(cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)),
 cirq.PhasedXPowGate(phase_exponent=-0.5987667922766213, exponent=0.4136540654256824).on(cirq.GridQubit(4, 3)),
 (cirq.Z**-0.9255092746611595).on(cirq.GridQubit(4, 4)),
 (cirq.Z**-1.333333333333333).on(cirq.GridQubit(4, 3)),
 cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0).on(cirq.GridQubit(4, 4))]

ops, circuit = convert_ops_to_sycamore(operation_list=operation_list)
print(ops)
print(len(ops))
