import numpy as np

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
        assert np.allclose(A_c[i,i],1.0)
    
    # lower right hand 2x2 matrix remaining after decomp
    lower_rh_matrix = A_c[n-2:n, n-2:n]
    
    # if not equal to I - is a non trivial gate
    if not is_identity(lower_rh_matrix):
        decomp.append(lower_rh_matrix)
        indices.append(np.array([n-2,n-1]))

    return decomp, indices

# ===================================================================================================
def gray_method(A):
    
    n = A.shape[0]
    M = np.copy(A)

    # using bitwise_xor find Gray permutations
    permutations = []
    for i in range(n):
        permutations.append(i ^ (i // 2))
        
    # 
    M[:,:] = M[:,permutations]
    M[:,:] = M[permutations,:]
    
    decomp, indices = two_level_decomp(M)
    new_decomp = []
    new_ind = []

    for i in range(len(indices)):
        
        t = np.take(permutations, indices[i])
        if t[0]>t[1]:
            new_decomp.append(decomp[i].T)
            new_ind.append(np.sort(t))

        else:
            new_decomp.append(decomp[i])
            new_ind.append(t)
            
    return new_decomp, new_ind
