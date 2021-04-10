import cirq
import numpy as np

def decompose_u(u,tol):

    """
    Return a Phase XZ decomposition of a 2x2 unitary matrix
    u - 2x2 unitary matrix
    tol - limit on the amount of error introduced by the construction.
    """

    decomp = cirq.optimizers.single_qubit_matrix_to_phxz(u,tol)

    if decomp != None:
        return decomp

    else:
        #TODO: Decide how to handle identity case
        print("Identity")
