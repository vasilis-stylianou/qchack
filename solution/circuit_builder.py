import cirq

def factorized_gate_list_to_qcircuit(
        factorized_gate_list,
        # Of the form [[gate_2q, [qx, qy]], [gate_1q, [qx]], ... ]
        device = cirq.google.Sycamore,
):
    """
    Converts the factorized list of gates (1q and 2q) into a cirq circuit (non-optimized).
    :param factorized_gate_list:
    :param device:
    :return:
    """

    def convert_gatelist_to_cirq_ops(gatelist):
        # Implement once we know the exact input.
        return gatelist

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
    # From list to cirq.CNOT.on(cirq.NamedQubit("a"), cirq.NamedQubit("b")))
    ops = convert_gatelist_to_cirq_ops(factorized_gate_list)

    # Convert incompatible gates to compatible
    ops_comp = convert_to_compatible_gates(ops, device)

    # When we append operations now, they are put into different moments.
    circuit.append(ops)

    # Optimise circuit for Sycamore
    circuit = cirq.google.optimized_for_sycamore(circuit)

    print(circuit)
    return circuit

