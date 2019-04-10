import numpy as np

def transposeMatrix(m):
    return map(list,zip(*m))

def getMatrixMinor(m,i,j):
    rows = len(m) - 1
    ret = [[0 for u in range(rows)] for v in range(rows)]

    for r in range(len(m)):
        for c in range(len(m)):
            if r != i and c != j:
                rr = (r > i) * -1 + r
                cc = (c > j) * -1 + c
                ret[rr][cc] = m[r][c]
    return ret

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1] * 1.0 -m[0][1]*m[1][0] * 1.0

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1] * 1.0 /determinant, -1.0 * m[0][1] / determinant],
                [-1*m[1][0] * 1.0 /determinant, m[0][0] * 1.0 / determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] * 1.0 / determinant
    return cofactors


def gauss_jordan(m, eps = 1.0/(10**10)):
    """Puts given matrix (2D array) into the Reduced Row Echelon Form.
     Returns True if successful, False if 'm' is singular.
    NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
     Written by Jarno Elonen in April 2005, released into Public Domain"""
    
    (h, w) = (len(m), len(m[0]))
    maxrow = [0]
    for y in range(0,h):
        maxrow[0] = y
        for y2 in range(y+1, h):    # Find max pivot
            b = int(abs(m[y2][y]) > abs(m[maxrow[0]][y]))
            maxrow[0] = maxrow[0] * (1 - b) + y2 * b
        (m[y], m[maxrow[0]]) = (m[maxrow[0]], m[y])
        # if abs(m[y][y]) <= eps:     # Singular?
        #     return False
        for y2 in range(y+1, h):    # Eliminate column y
            c = m[y2][y] / m[y][y]
            for x in range(y, w):
                m[y2][x] -= m[y][x] * c
    for y in range(h-1, 0-1, -1): # Backsubstitute
        c  = m[y][y]
        for y2 in range(0,y):
            for x in range(w-1, y-1, -1):
                m[y2][x] -=  m[y][x] * m[y2][y] / c
        m[y][y] /= c
        for x in range(h, w):       # Normalize row y
            m[y][x] /= c
    return True

def gj_inv(M):
    """
    return the inv of the matrix M
    """
    # clone the matrix and append the identity matrix
    # [int(i==j) for j in range_M] is nothing but the i(th row of the identity matrix
  
    m2 = [row[:]+[int(i==j) for j in range(len(M) )] for i,row in enumerate(M) ]
    # extract the appended matrix (kind of m2[m:,...]
    return [row[len(M[0]):] for row in m2] if gauss_jordan(m2) else None


import math
def cholesky(A):
    """Performs a Cholesky decomposition of A, which must 
    be a symmetric and positive definite matrix. The function
    returns the lower variant triangular matrix, L."""
    n = len(A)

    # Create zero matrix for L
    L = [[0.0] * n for i in xrange(n)]

    # Perform the Cholesky decomposition
    for i in xrange(n):
        for k in xrange(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in xrange(k))
            
            if (i == k): # Diagonal elements
                # LaTeX: l_{kk} = \sqrt{ a_{kk} - \sum^{k-1}_{j=1} l^2_{kj}}
                L[i][k] = math.sqrt(abs(A[i][i] - tmp_sum))
            else:
                # LaTeX: l_{ik} = \frac{1}{l_{kk}} \left( a_{ik} - \sum^{k-1}_{j=1} l_{ij} l_{kj} \right)
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L

def cholesky_inv(A):
    L = cholesky(A)

    L_inv = [[0 for r in range(len(L))] for c in range(len(L))]
    n = len(L)
    for k in range(n):
        L_inv[k][k] = 1.0 / L[k][k]
        for i in range(k, n):
            v = 0
            for j in range(k, i-1):
                v += L[i][j] * L_inv[j][k]
            
            L[i][k] = -1 * v / L[i][i]

    L_inv_T = transposeMatrix(L_inv)
    return np.matmul(np.array(L_inv_T), np.array(L_inv))

def cond_assign(b, x, y):
    return b * x + (1 - b) * y

def gj_inv_alternate(A):
    n = len(A)
    X = [[A[j][i] for i in range(n)] for j in range(n)]
    I = [[int(i==j) for i in range(n)] for j in range(n)]

    for j in range(n):
        for i in range(j, n):
            b = int(X[i][j] != 0)
            for k in range(0, n):
                s = X[j][k]
                X[j][k] = cond_assign(b, X[i][k], X[j][k])
                X[i][k] = cond_assign(b, s, X[i][k])

                s = I[j][k]
                I[j][k] = cond_assign(b, I[i][k], I[j][k])
                I[i][k] = cond_assign(b, s, I[i][k])

            t = cond_assign(b, 1.0 / X[j][j], 1.0)
            for k in range(n):
                X[j][k] = t * X[j][k]
                I[j][k] = t * I[j][k]

            for L in range(j):
                t = -1 * X[L][j]
                for k in range(n):
                    X[L][k] = cond_assign(b, X[L][k] + t * X[j][k], X[L][k])
                    I[L][k] = cond_assign(b, I[L][k] + t * I[j][k], I[L][k])

            for L in range(j+1, n):
                t = -1 * X[L][j]
                for k in range(n):
                    X[L][k] = cond_assign(b, X[L][k] + t * X[j][k], X[L][k])
                    I[L][k] = cond_assign(b, I[L][k] + t * I[j][k], I[L][k])
    return I

X = [[1.0, 2.0, 3.0], [2.0, 6.0, 1.0], [4.0, 5.0, 6.0]]
X_inv = gj_inv_alternate(X)
X_inv_np = np.linalg.inv(np.array(X))

print np.array(X)
print np.array(X_inv)
print X_inv_np
print 
print np.matmul(np.array(X), np.array(X_inv))
print np.matmul(np.array(X), X_inv_np)
