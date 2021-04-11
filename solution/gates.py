import cirq
import numpy as np

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

# ###################################################################################################
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
        #TODO: Decide how to handle identity case
        print("Identity")

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