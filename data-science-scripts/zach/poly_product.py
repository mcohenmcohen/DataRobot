from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import numpy as np

def poly_product(m1, m2):
    """
    Return data, indices, and indptr required for csr_matrix polynomial product of m1 & m2
    
    New columns correspond to m1 features offset for every feature in m2,
    e.g., m1[1] * m1[2] value is in column = 1 + 2*(num_features_in_m1)
    """
    assert m1.shape[0] == m2.shape[0]  # rows must be equal
    
    # Calculate to create zero filled np arrays
    sizes_1 = m1.indptr[1:] - m1.indptr[:-1]
    sizes_2 = m2.indptr[1:] - m2.indptr[:-1]
    sizes = np.sum(sizes_1 * sizes_2)

    data = np.zeros(sizes, dtype=np.float32)
    indices = np.zeros(sizes, dtype=np.int32)
    indptr = np.zeros(m1.shape[0] + 1, dtype=np.int32)  # indptr starts with zero and is 1 plus longer than others
    for i in range(m1.shape[0]):
        # Cross product data for row
        row_data_1 = m1.data[m1.indptr[i]:m1.indptr[i+1]]
        row_data_2 = m2.data[m2.indptr[i]:m2.indptr[i+1]]
        row_data = (row_data_1 * row_data_2.reshape(-1, 1)).flatten()

        # increment row indptr, needs to happen before data array update
        indptr[i+1] = indptr[i] + len(row_data)
        
        # update data array
        data[indptr[i]:indptr[i+1]] = row_data

        
        # Cross product indices for row
        r1 = m1.indices[m1.indptr[i]:m1.indptr[i+1]]
        r2 = m2.indices[m2.indptr[i]:m2.indptr[i+1]]
        len1 = len(r1)
        len2 = len(r2)
        r1 = r1 * np.ones((len2, len1))
        r2 = r2.reshape(-1, 1) * np.ones((len2, len1)) * m1.shape[1] # r2 is offset by lenth of number of features in m1
        row_ind = (r1 + r2).flatten()
        
        indices[indptr[i]:indptr[i+1]] = row_ind
        
    return csr_matrix((data, indices, indptr)) 


A = csr_matrix(
    np.asarray([[1, 0, 2, 0, 10],
                [0, 3, 0, 0, 5],
                [0, 3, 1, 6, 0]]))

B = csr_matrix(
    np.asarray([[0, 4, 0],
                [1, 0, 6],
                [0, 3, 9]]))


print(poly_product(A, B).todense())


hstack((poly_product(A, B), A.todense(), B.todense()), format='csr')

a0 = np
