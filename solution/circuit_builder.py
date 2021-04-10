import cirq

def factorized_gate_list_to_qcircuit(
        factorized_gate_list,
        # Of the form [[gate_2q, [qx, qy]], [gate_1q, [qx]], ... ]
        target_qubits,
        device = cirq.google.Sycamore,
):
    """
    Converts the factorized list of gates (1q and 2q) into a cirq circuit (non-optimized).
    :param factorized_gate_list:
    :param device:
    :return:
    """

    def convert_gatelist_to_cirq_ops(gatelist, target_qubits):
        # Implement once we know the exact input.
        ops = []
        for gate, bits in gatelist:
            if len(bits) == 1:
                q1 = target_qubits[bits[0]]
                ops.append(gate(q1))
            elif len(bits) == 2:
                q1, q2 = target_qubits[bits[0]], target_qubits[bits[1]]
                ops.append(gate(q1, q2))
            else:
                NotImplementedError("Gates larger than 2 qbits are not implemented.")

        return ops

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

        for i, op in enumerate(operations):
            # Check if X is supported.
            if not sycamore_gates.is_supported_operation(op):
                """Convert a gate to xmon gates."""

                # Do the conversion.
                op_conv = converter.convert(op)
                operations[i] = op_conv

        return operations

    # GENERAL PARAMS
    sycamore_gates = cirq.google.gate_sets.SYC_GATESET

    # Create a circuit on the device
    circuit = cirq.Circuit(device=device)

    # Convert list of gates to cirq operations
    ops = convert_gatelist_to_cirq_ops(factorized_gate_list, target_qubits)

    # Convert incompatible gates to compatible
    ops_comp = convert_to_compatible_gates(ops, device)

    # When we append operations now, they are put into different moments.
    circuit.append(ops)

    # Optimise circuit for Sycamore
    circuit = cirq.google.optimized_for_sycamore(circuit)

    print(circuit)
    return circuit


factorized_gate_list = [[cirq.CZ,[0,1]], [cirq.CZ, [2,0]], [cirq.H, [3]]]
target_qubits = [cirq.GridQubit(4,i) for i in range(1,9)]
factorized_gate_list_to_qcircuit(factorized_gate_list, target_qubits)