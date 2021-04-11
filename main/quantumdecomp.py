

class QuantumDecomp:

    def __init__(self, target_qubits: List[cirq.GridQubit], matrix: np.ndarray):


        self.num_qubits = len(target_qubits)




    def two_level_unitary_to_gates(self):
        """
        input: 2-level matrix, index_1, index_2
        output: sequence of unitary gates acting on qubits

        2-level matrix -> XZPhaseGate  (self._convert_two_matrix_to_XZGATE)

        case sensitive (based on indices):
        1) FC_XZPhaseGate
        2) Single Gate
        """ 




# --------------
# NOW
    def _FC(self):
        """
        FC_U : 
        FC_util :
            - 
            - 
        """


    def _use_grays_method(self):
        """
        input: index 1, index2
        output: arr of arrays
        """




#-----------
# HAve
    def _convert_two_matrix_to_XZGATE(self):
        """
        input: two-level matrix
        output: XZPhaseGate
        """





    def two_level_decompose_gray(self):

        """
        Decomposes unitary to two-level matrices 
        AND permutes indices according to Gray's method

        input: 2^N x 2^N unitary matrix
        output: List[tuple(2-level matrix, index_1, index_2)]
        """


    def _two_level_decompose(self):

        """
        Decomposes unitary to two-level matrices
         
        input: 2^N x 2^N unitary matrix
        output: List[tuple(2-level matrix, index_1, index_2)]
        """


