import cirq

def breakdown_multiple_control_op_to_convertible_gate_op(multiple_control_operation):

    op = multiple_control_operation
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

    ops_decomp = cirq.decompose_multi_controlled_rotation(matrix=matrix, controls=control_qubits, target=target_qubit, )

    # Reverse x gates
    ops_decomp_xs = []
    ops_decomp_xs.append(x_gates)
    ops_decomp_xs.append(ops_decomp)
    ops_decomp_xs.append(x_gates)

    return ops_decomp_xs
