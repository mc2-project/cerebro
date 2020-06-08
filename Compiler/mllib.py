from Compiler.types import *
from Compiler.instructions import *
from Compiler.util import tuplify,untuplify
from Compiler import instructions,instructions_base,comparison,program
import inspect,math
import random
import collections
from Compiler.library import *

from Compiler.types_gc import *

from operator import itemgetter

import numpy as np

def get_diff_types(data_list):
    cint_data = [d for d in data_list if type(d) == cint]
    pint_data = [(d, d.pid) for d in data_list if type(d) == pint]
    sint_data = [d for d in data_list if type(d) == sint]

    if len(pint_data) > 1:
        pint_data = sorted(pint_data, key=itemgetter(1))

    return (cint_data, pint_data, sint_data)
    
# This is not parallelized
def int_add(data_list, nparallel=1):
    (cint_data, pint_data, sint_data) = get_diff_types(data_list)

    c_res = cint(0)
    for cd in cint_data:
        c_res += cd

    pd_res = []
    current_pid = None
    for (pd, pid) in pint_data:
        if pid != current_pid:
            current_pid = pid
            pd_res.append(pint(0))
        pd_res[-1] += pd

    res = cint(0)
        
    res += c_res
    for pd in pd_res:
        res += pd
        
    for sd in sint_data:
        res += sd
    
    return res



def sum_lib(lst):
    flattened_lst = []
    for i in range(len(lst)):
        print "TYPE?", type(lst[i])
        if type(lst[i]) in (sfixMatrix, cfixMatrix, sfixMatrixGC, cfixMatrixGC):
            flattened_lst += flatten(lst[i])
            print flattened_lst
        else:
            flattened_lst.append(lst[i])

    
    return sum(flattened_lst)

def max_lib(lst):
    flattened_lst = []
    for i in range(len(lst)):
        print "TYPE?", type(lst[i])
        if type(lst[i]) in (sfixMatrix, cfixMatrix, sfixMatrixGC, cfixMatrixGC):
            flattened_lst += flatten(lst[i])
            print flattened_lst
        else:
            flattened_lst.append(lst[i])

    
    return max(flattened_lst)


def min_lib(lst):
    flattened_lst = []
    for i in range(len(lst)):
        print "TYPE?", type(lst[i])
        if type(lst[i]) in (sfixMatrix, cfixMatrix, sfixMatrixGC, cfixMatrixGC):
            flattened_lst += flatten(lst[i])
            print flattened_lst
        else:
            flattened_lst.append(lst[i])

    
    return min(flattened_lst)



def flatten(A):
    lst = []
    if type(A) in (sfixMatrix, sfixMatrixGC, cfixMatrix, cfixMatrixGC):
        for i in range(A.rows):
            for j in range(A.columns):
                lst.append(A[i][j])

    return lst

import functools
def reduce_lib(lst, reduce_fn):
    flattened_lst = []
    for i in range(len(lst)):
        if type(lst[i]) in(sfixMatrix, cfixMatrix, sfixMatrixGC, cfixMatrixGC):
            flattened_lst += flatten(lst[i])
        else:
            flattened_lst.append(lst[i])

    return reduce(reduce_fn, flattened_lst)

# Copy a portion of the large matrix to the small matrix. 
def copy_matrix(dest, src, rows, cols, index):
    for i in range(rows):
        for j in range(cols):
            dest[i][j] = src[index * rows + j][j]

#  Tree-based multiplication
def int_multiply(data_list, nparallel=2):
    length = len(data_list)

    data = []
    data.append(Array(length, sint))
    for i in range(length):
        data[0][i] = data_list[i]

    while length > 1:
        length = (length / 2) + (length % 2)
        data.append(Array(length, sint))
        @for_range(length)
        def f(i):
            data[-1][i] = sint(0)

    level = 0
    for x in range(len(data) - 1):
        print("level = {}, length = {}".format(level+1, data[level+1].length))
        
        exec_len = data[level].length / 2
        @for_range_multithread(nparallel, exec_len, exec_len)
        def _multiply(i):
            data[level+1][i] = data[level][2 * i] * data[level][2 * i + 1]

        if data[level].length % 2 > 0:
            data[level+1][data[level+1].length - 1] = data[level][data[level].length - 1]

        level += 1

    return data[-1][0]

def _transpose(A, B):
    @for_range(A.rows)
    def f(i):
        @for_range(A.columns)
        def g(j):
            B[j][i] = A[i][j]

def _transpose_gc(A, B):
    for i in range(A.rows):
        for j in range(A.columns):
            B[j][i] = A[i][j]    

def transpose(A):
    if isinstance(A, np.ndarray):
        return A.transpose()

    if not isinstance(A, (Matrix, MatrixGC)):
        raise ValueError("Only matrix can be transposed")

    if isinstance(A, (sintMatrix, sfixMatrix, cintMatrix, cfixMatrix)):
        B = A.__class__(A.columns, A.rows)
        _transpose(A, B)
        return B
    elif isinstance(A, (sintMatrixGC, sfixMatrixGC)):
        B = A.__class__(A.columns, A.rows)
        _transpose_gc(A, B)
        return B
    else:
        raise NotImplementedError

def _matmul(A, B, C, D, int_type, nparallel=1):
    total = A.rows * B.columns * A.columns
    @for_range_multithread(nparallel, total, total)
    def _multiply(i):
        i_index = i / (B.columns * A.columns)
        j_index = i % (B.columns * A.columns) / (A.columns)
        k_index = i % A.columns

        D[i] = A[i_index][k_index] * B[k_index][j_index]

    @for_range_multithread(nparallel, A.rows * B.columns, A.rows * B.columns)
    def _add(i):
        i_index = i / B.columns
        j_index = i % B.columns
        C[i_index][j_index] = int_type(0)
            
        @for_range(A.columns)
        def _add_element(j):
            C[i_index][j_index] += D[i * A.columns + j]

    return C

# Not parallelized
def _matmul_mix(A, B, nparallel=1):
    C = MixMatrix(A.rows, B.columns)
    @for_range(A.rows * B.columns)
    def f(i):
        @for_range(A.columns)
        def g(j):
            v = C.get(i)
            v += A.get(i * A.columns + j) * B.get(j * B.columns + i)
            C.set(i, v)

    return C

def _matmul_gc(A, B, C):
    for i in range(A.rows):
        for j in range(B.columns):
            v = A[i][0] * B[0][j]
            for k in range(1, A.columns):
                v += A[i][k] * B[k][j]
            C[i][j] = v

def matmul(A, B, left_rows, left_cols, right_rows, right_cols, mat_type, nparallel=1):

    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return np.matmul(A, B)


    # Tentative, very janky. Yep, this doesn't work :(. Buyer BEWARE!
    if isinstance(A, sintMatrix) and isinstance(B, sintMatrix):
        C = sintMatrix(A.rows, B.columns)
        D = sintArray(A.rows * B.columns * A.columns)
        return _matmul(A, B, C, D, sint, nparallel)
        #C = sintMatrix(left_rows, right_cols)
        #D = sintArray(left_rows * right_cols * left_cols)
        #return _matmul(A, B, C, D, sint, nparallel)

    elif isinstance(A, cintMatrix) and isinstance(B, cintMatrix):
        C = cintMatrix(A.rows, B.columns)
        D = cintArray(A.rows * B.columns * A.columns)
        return _matmul(A, B, C, D, cint, nparallel)
    elif isinstance(A, sfixMatrix) and isinstance(B, sfixMatrix):
        C = sfixMatrix(A.rows, B.columns)
        D = sfixArray(A.rows * B.columns * A.columns)
        return _matmul(A, B, C, D, sfix, nparallel)
    elif isinstance(A, cfixMatrixGC) or isinstance(B, cfixMatrixGC):
        C = cfixMatrixGC(A.rows, B.columns)
        _matmul_gc(A, B, C)
        return C
    elif isinstance(A, sfixMatrixGC) or isinstance(B, sfixMatrixGC):
        C = sfixMatrixGC(A.rows, B.columns)
        _matmul_gc(A, B, C)
        return C
    elif isinstance(A, MixMatrix) and isinstance(B, MixMatrix):
        return _matmul_mix(A, B, nparallel)

    elif isinstance(A, (sintMatrix, cintMatrix, cfixMatrix, sfixMatrix)) and isinstance(B, (sintMatrix, cintMatrix, cfixMatrix, sfixMatrix)):
        C = sintMatrix(A.rows, B.columns)
        D = sintArray(A.rows * B.columns * A.columns)
        return _matmul(A, B, C, D, sint, nparallel)
    else:
        raise NotImplementedError

def _matadd(A, B, C, int_type, nparallel=1):
    @for_range_multithread(nparallel, A.rows * A.columns, A.rows * A.columns)
    def _add(i):
        i_index = i / A.columns
        j_index = i % A.columns
        
        C[i_index][j_index] = A[i_index][j_index] + B[i_index][j_index]
    
def matadd(A, B, nparallel=1):
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return np.add(A, B)

    if A.rows != B.rows or A.columns != B.columns:
        raise NotImplementedError
    
    if isinstance(A, cintMatrix) and isinstance(B, cintMatrix):
        C = cintMatrix(A.rows, A.columns)
        _matadd(A, B, C, cint, nparallel)
        return C 
    elif isinstance(A, sintMatrix) and isinstance(B, sintMatrix):
        C = sintMatrix(A.rows, A.columns)
        _matadd(A, B, C, sint, nparallel)
        return C 

    elif isinstance(A, sfixMatrix) and isinstance(B, sfixMatrix):
        C = sfixMatrix(A.rows, A.columns)
        _matadd(A, B, C, sfix, nparallel)
        return C

    elif type(A) in (sfixMatrix, cfixMatrix) and type(B) in (sfixMatrix, cfixMatrix):
        C = sfixMatrix(A.rows, A.columns)
        _matadd(A, B, C, sfix, nparallel)
        return C


def _matsub(A, B, C, int_type, nparallel=1):
    @for_range_multithread(nparallel, A.rows * A.columns, A.rows * A.columns)
    def _add(i):
        i_index = i / A.columns
        j_index = i % A.columns
        
        C[i_index][j_index] = A[i_index][j_index] - B[i_index][j_index]

def _matsub_gc(A, B, C):
    for i in range(A.rows):
        for j in range(A.columns):
            C[i][j] = A[i][j] - B[i][j]
        
def matsub(A, B, nparallel=1):
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return np.subtract(A, B)

    if A.rows != B.rows or A.columns != B.columns:
        raise ValueError("[matsub] Matrices must have the same sizes")
    
    if isinstance(A, cintMatrix) and isinstance(B, cintMatrix):
        C = cintMatrix(A.rows, A.columns)
        _matsub(A, B, C, cint, nparallel)
        return C
    elif isinstance(A, sintMatrix) and isinstance(B, sintMatrix):
        C = sintMatrix(A.rows, A.columns)
        _matsub(A, B, C, sint, nparallel)
        return C
    elif isinstance(A, sfixMatrix) and isinstance(B, sfixMatrix):
        C = sfixMatrix(A.rows, A.columns)
        _matsub(A, B, C, sfix, nparallel)
        return C
    elif isinstance(A, sfixMatrixGC) and isinstance(B, sfixMatrixGC):
        C = sfixMatrixGC(A.rows, A.columns)
        _matsub_gc(A, B, C)
        return C
    else:
        raise NotImplementedError


# horizontally stack the input matrices
def matstack_int(matrices):
    pid = None

    s = set([m.columns for m in matrices])
    if s > 1:
        raise ValueError("Can only stack matrices with the same number of columns")
    
    num_rows_list = [m.rows for m in matrices]
    M_rows = sum(num_rows_list)
    M_columns = s.pop()

    M = cintMatrix(M_rows, M_columns)
    int_type = cint
    pid = 0
    s = set(type(m) for m in matrices)
    if len(s) == 1 and cintMatrix in s:
        M = cintMatrix(M_rows, M_columns)
        int_type = cint
    elif len(s) == 1 and pintMatrix in s:
        parties = set([m.pid for m in matrices])
        if len(parties) == 1:
            pid = parties.pop()
            M = pintMatrix(pid, M_rows, M_columns)
            int_type = pint
        else:
            M = sintMatrix(M_rows, M_columns)
            int_type = sint
    else:
        M = sintMatrix(M_rows, M_columns)
        int_type = sint

    row_count = 0
    for m in matrices:
        @for_range(m.rows)
        def f(i):
            @for_range(m.columns)
            def g(j):
                if int_type == pint:
                    M[row_count + i][j] = pint(pid, 0)
                else:
                    M[row_count + i][j] = int_type(0)
                M[row_count + i][j] += m[i][j]

    return M

def matstack(matrices):
    if isinstance(matrices[0], (cintMatrix, pintMatrix, sintMatrix)):
        return matstack_int(matrices)
    else:
        raise NotImplementedError

def _sigmoid_sfix(v):
    sign_v = cfix(1) - cfix(2) * (v < 0)
    denom = (v * sign_v) + sfix(1)
    res = v / denom
    return res

def _sigmoid_sfix_gc(v):
    abs_v = v.absolute()
    denom = abs_v + cfix_gc(1)
    res = v / denom
    return res

def sigmoid(v, nparallel=1):
    if isinstance(v, sfix):
        return _sigmoid_sfix(v)
    elif isinstance(v, (sfixMatrix)):
        res = v.__class__(v.rows, v.columns)
        @for_range_multithread(nparallel, v.rows, v.rows)
        def a(i):
            @for_range_multithread(nparallel, v.columns, v.columns)
            def b(j):
                res[i][j] = _sigmoid_sfix(v[i][j])
        return res
    elif isinstance(v, sfixMatrixGC):
        res = v.__class__(v.rows, v.columns)
        for i in range(v.rows):
            for j in range(v.columns):
                res[i][j] = _sigmoid_sfix_gc(v[i][j])
        return res        
    else:
        raise NotImplementedError

def mat_const_mul(c, m, nparallel=1):

    if isinstance(m, np.ndarray):
        if type(c) in (float, int):
            return c * m 
        else:
            raise ValueError("Type of constant is: {0} when expected float and int.".format(type(c)))


    if isinstance(m, sfixMatrix) or isinstance(m, cfixMatrix):
        if isinstance(m, sfixMatrix):
            res = sfixMatrix(m.rows, m.columns)
        else:
            res = cfixMatrix(m.rows, m.columns)
        """
        @for_range_multithread(nparallel, m.rows * m.columns, m.rows * m.columns)
        def f(i):
            @for_range_multithread(nparallel, m.columns, m.columns)
            def g(j):
                res[i][j] = c * m[i][j]
        """
        @for_range_multithread(nparallel, m.rows * m.columns, m.rows * m.columns)
        def loop(i):
            i_index = i / m.columns
            j_index = i % m.columns
            res[i_index][j_index] = c * m[i_index][j_index]
        
        return res
    elif isinstance(m, sfixMatrixGC) or isinstance(m, cfixMatrixGC):
        if isinstance(m, sfixMatrixGC):
            res = sfixMatrixGC(m.rows, m.columns)
        else:
            res = cfixMatrixGC(m.rows, m.columns)
        for i in range(m.rows):
            for j in range(m.columns):
                res[i][j] = c * m[i][j]
        return res

    else:
        raise NotImplementedError

def mat_assign(o, i, nparallel=1):
    if isinstance(i, (Array, ArrayGC)):
        if o.length != i.length:
            raise ValueError("Arrays must be of the same sizes")
        if isinstance(i, Array):
            @for_range(i.length)
            def f(u):
                o[u] = i[u]
        elif isinstance(i, ArrayGC):
            for u in range(i.length):
                o[u] = i[u]
    elif isinstance(i, (Matrix, MatrixGC)):
        if o.rows != i.rows or o.columns != i.columns:
            raise ValueError("Matrices must be of the same sizes")

        if isinstance(i, Matrix):
            @for_range_multithread(nparallel, i.rows, i.rows)
            def f(u):
                @for_range_multithread(nparallel, i.columns, i.columns)
                def g(v):
                    o[u][v] = i[u][v]
        elif isinstance(i, MatrixGC):
            for u in range(i.rows):
                for v in range(i.columns):
                    o[u][v] = i[u][v]
    elif isinstance(i, list):
        for u in range(len(i)):
            o[u] = i[u]
    else:
        raise NotImplementedError
                    
def array_index_secret_load_if(condition, l, index_1, index_2, nparallel=1):
    supported_types_a = (sint, sfix)
    supported_types_b = (sint_gc, sfix_gc)
    if isinstance(index_1, supported_types_a) and isinstance(index_2, supported_types_a): 
        index = ((1 - condition) * index_1) + (condition * index_2)
        return array_index_secret_load_a(l, index, nparallel=nparallel)
    elif isinstance(index_1, supported_types_b) and isinstance(index_2, supported_types_b):
        index = ((~condition) & index_1).__xor__(condition & index_2)
        return array_index_secret_load_gc(l, index)
    else:
        raise NotImplementedError

def get_identity_matrix(value_type, n):
    if isinstance(value_type, (sfix, sfixMatrix)):
        ret = sfixMatrix(n, n)
        @for_range(n)
        def f(i):
            @for_range(n)
            def g(j):
                v = (i == j)
                v = sint(v)
                vfix = sfix.load_sint(v)
                ret[i][j] = vfix
        return ret
    elif isinstance(value_type, (sfix_gc, sfixMatrixGC)):
        ret = sfixMatrixGC(n, n)
        for i in range(n):
            for j in range(n):
                ret[i][j] = cfix_gc(int(i == j))
        return ret
    else:
        raise NotImplementedError

def matinv(A, nparallel=1):
    if isinstance(A, np.ndarray):
        return np.linalg.inv(A)



    if not isinstance(A, sfixMatrix) and not isinstance(A, cfixMatrix):
        raise NotImplementedError
    
    n = A.rows
    X = A.__class__(A.rows, A.columns)
    mat_assign(X, A)
    
    I = get_identity_matrix(A, A.rows)

    @for_range(n)
    def f0(j):
        #@for_range(j, n)
        @for_range(n)
        def f1(i):
            @if_(i >= j)
            def h():
                b1 = X[i][j].__lt__(sfix(0.00001))
                b2 = X[i][j].__gt__(sfix(-0.00001))
                b = 1 - b1 * b2
                X[i][j] = b * X[i][j]
                @for_range_multithread(nparallel, n, n)
                def f2(k):
                    a1 = X[j][k]
                    a2 = X[i][k]
                    X[j][k] = cond_assign_a(b, a2, a1)
                    X[i][k] = cond_assign_a(b, a1, a2)

                    a1 = I[j][k]
                    a2 = I[i][k]
                    I[j][k] = cond_assign_a(b, a2, a1)
                    I[i][k] = cond_assign_a(b, a1, a2)
                    
                xjj_inv = sfix(1).__div__(X[j][j])
                t = cond_assign_a(b, xjj_inv, sfix(1))
                @for_range_multithread(nparallel, n, n)
                def f3(k):
                    X[j][k] = t * X[j][k]
                    I[j][k] = t * I[j][k]
                
                @for_range(n)
                def f4(L):
                    @if_(L < j)
                    def h():
                        t = sfix(-1) * X[L][j]
                        @for_range_multithread(nparallel, n, n)
                        def g0(k):
                            a1 = X[L][k] + t * X[j][k]
                            a2 = X[L][k]
                            b1 = I[L][k] + t * I[j][k]
                            b2 = I[L][k]
                            X[L][k] = cond_assign_a(b, a1, a2)
                            I[L][k] = cond_assign_a(b, b1, b2)

                # from j+1 to n
                @for_range(n)
                def f5(L):
                    @if_(L > j)
                    def h():
                        t = sfix(-1) * X[L][j]
                        @for_range_multithread(nparallel, n, n)
                        def g0(k):
                            a1 = X[L][k] + t * X[j][k]
                            a2 = X[L][k]
                            b1 = I[L][k] + t * I[j][k]
                            b2 = I[L][k] 
                            X[L][k] = cond_assign_a(b, a1, a2)
                            I[L][k] = cond_assign_a(b, b1, b2)
        
    return I

# Assumes that the piecewise function is public for now
# Format: bounds in the form of [lower, upper]
# Function in the form of a*x + b
class Piecewise(object):
    def __init__(self, num_boundaries):
        self.lower_bound = sfixArray(3)
        self.upper_bound = sfixArray(3)
        self.boundary_points = sfixMatrix(num_boundaries - 2, 4)
        self.counter = regint(0)

    def add_boundary(self, lower, upper, a, b):
        if lower is None:
            self.lower_bound[0] = upper
            self.lower_bound[1] = a
            self.lower_bound[2] = b
        elif upper is None:
            self.upper_bound[0] = lower
            self.upper_bound[1] = a
            self.upper_bound[2] = b
        else:
            self.boundary_points[self.counter][0] = lower
            self.boundary_points[self.counter][1] = upper
            self.boundary_points[self.counter][2] = a
            self.boundary_points[self.counter][3] = b
            self.counter += regint(1)

    # For debugging purposes only
    def debug(self):
        print_ln("[-inf, %s],: %s * x + %s", self.lower_bound[0].reveal(), self.lower_bound[1].reveal(), self.lower_bound[2].reveal())
        @for_range(self.boundary_points.rows)
        def f(i):
            print_ln("[%s, %s]: %s * x + %s", self.boundary_points[i][0].reveal(), self.boundary_points[i][1].reveal(), self.boundary_points[i][2].reveal(), self.boundary_points[i][3].reveal())
        print_ln("[%s, inf],: %s * x + %s", self.upper_bound[0].reveal(), self.upper_bound[1].reveal(), self.upper_bound[2].reveal())

    def evaluate(self, x):
        coefs = sfixArray(2)
        coefs[0] = sfix(0)
        coefs[1] = sfix(0)

        # Check for lower bound
        b = x.__le__(self.lower_bound[0])
        coefs[0] += b * self.lower_bound[1]
        coefs[1] += b * self.lower_bound[2]        
        
        @for_range(self.boundary_points.rows)
        def f(i):
            lower = self.boundary_points[i][0]
            upper = self.boundary_points[i][1]

            b1 = x.__gt__(lower)
            b2 = x.__le__(upper)
            b = b1 * b2
            coefs[0] += b * self.boundary_points[i][2]
            coefs[1] += b * self.boundary_points[i][3]

        # Check for upper bound
        b = x.__gt__(self.upper_bound[0])
        coefs[0] += b * self.upper_bound[1]
        coefs[1] += b * self.upper_bound[2]

        res = coefs[0] * x + coefs[1]
        return res




def LogisticRegression(X, y, batch_size, sgd_iters, dim):
    assert(isinstance(X, Matrix))
    assert(isinstance(y, Matrix))

    if batch_size * sgd_iters >= X.rows:
        raise ValueError("batch_size * sgd_iters = {0} * {1} >= # of rows in X: {2}".format(batch_size, sgd_iters. X.rows))

    if batch_size * sgd_iters >= y.rows:
        raise ValueError("batch_size * sgd_iters = {0} * {1} >= # of rows in X: {2}".format(batch_size, sgd_iters. X.rows))


    
    if isinstance(X, sfixMatrix):
        w = sfixMatrix(dim, 1)
        #alpha_B = cfix(0.01 / batch_size) currently cfix and sfix multiplying doesn't work
        alpha_B = cfix(0.01 / batch_size)
        XB = sfixMatrix(batch_size, dim)
        yB = sfixMatrix(batch_size, 1)
    else:
        w = sfixMatrixGC(dim, 1)
        alpha_B = cfix_gc(0.01 / batch_size)
        XB = sfixMatrixGC(batch_size, dim)
        yB = sfixMatrixGC(batch_size, 1)

    for i in range(sgd_iters):
        batch_low = i * batch_size
        batch_high = (i + 1) * batch_size

        for j in range(batch_size):
            for d in range(dim):
                XB[j][d] = X[batch_low + j][d]
            yB[j][0] = y[batch_low + j][0]

        w_ret = matmul(XB, w, batch_size, dim, dim, 1, sfix)
        #reveal_all(w_ret, "w_ret")
        w_sigmoid = sigmoid(w_ret)
        #reveal_all(w_sigmoid, "w_sigmoid")
        w_sub = matsub(w_sigmoid, yB)
        XB_T = transpose(XB)
        w_1 = matmul(XB_T, w_sub, dim, batch_size, batch_size, 1, sfix)
        #reveal_all(w_1, "w_1")
        w_2 = mat_const_mul(alpha_B, w_1)
        #reveal_all(w_2, "w_2")
        w_res = matsub(w, w_2)
        mat_assign(w, w_res)
        #print_ln("Iter: %s", i)

    return w


def DecisionTree(tree, levels):
    w = tree[0]
    for i in range(levels-1):
        index = w[0]
        split = w[1]
        left_child = w[2]
        right_child = w[3]
        f = x[index]
        cond = (f < split)
        w_res = array_index_secret_load_if(cond, tree, left_child, right_child)
        mat_assign(w, w_res)

    # Return the final prediction class.
    return w[1]


def get_ith_matrix(mat, index, rows, cols, mat_type=sfixMatrix):
    #ret = s_fix_mat(rows, cols)
    #ret = sfixMatrix(rows, cols)
    ret = mat_type(rows, cols)
    for i in range(rows):
        for j in range(cols):
            ret[i][j] = mat[index * rows + i][j]
    return ret

def copy_ith_matrix(dest, src, index, rows, cols):
    for i in range(rows):
        for j in range(cols):
            dest[index * rows + i][j] = src[i][j]

# Local computation of weight vector.
def admm_local(XXinv, Xy, u, z, rho, num_cols):
    temp = matsub(z, u)
    z_u = mat_const_mul(rho, temp)
    #for i in range(z_u.rows):
        #print_ln("Admm local z: %s, temp: %s", z_u[i][0].reveal(), temp[i][0].reveal())
    second_term = matadd(Xy, z_u) #add_matrices(Xy, z_u, NUM_COLS, 1)
    w = matmul(XXinv, second_term, num_cols, num_cols, num_cols, 1, sfix)
    return w


def soft_threshold_vec(threshold, vec, num_cols, mat_type=sfixMatrix):
    #vec_new = s_fix_mat(NUM_COLS, 1)
    #vec_new = sfixMatrix(num_cols, 1)
    vec_new = mat_type(num_cols, 1)
    neg_threshold = sfix(-1) * threshold
    #neg_threshold = threshold.__neg__()
    for i in range(num_cols):
        threshold_fn = Piecewise(3)
        threshold_fn.add_boundary(None, neg_threshold, sfix(0), vec[i][0] + threshold)
        #threshold_fn.add_boundary(None, neg_threshold, c_fix(0), vec[i][0] + threshold)
        threshold_fn.add_boundary(neg_threshold, threshold, sfix(0), sfix(0))
        #threshold_fn.add_boundary(neg_threshold, threshold, c_fix(0), c_fix(0))
        threshold_fn.add_boundary(threshold, None, sfix(0), vec[i][0] - threshold)
        #threshold_fn.add_boundary(threshold, None, c_fix(0), vec[i][0] - threshold)
        val = threshold_fn.evaluate(vec[i][0])
        vec_new[i][0] = val

    return vec_new



def admm_coordinate(w_list, u_list, z, rho, l, num_cols, num_parties, mat_type=sfixMatrix):
    #w_avg = s_fix_mat(num_cols, 1)
    #u_avg = s_fix_mat(num_cols, 1)
    #w_avg = sfixMatrix(num_cols, 1)
    #u_avg = sfixMatrix(num_cols, 1)
    w_avg = mat_type(num_cols, 1)
    u_avg = mat_type(num_cols, 1)
    w_avg = mat_const_mul(cfix(0), w_avg)
    u_avg = mat_const_mul(cfix(0), u_avg)

    for i in range(num_parties):
        w = get_ith_matrix(w_list, i, num_cols, 1, mat_type)
        u = get_ith_matrix(u_list, i, num_cols, 1, mat_type)
        new_w_avg = matadd(w_avg, w) #add_matrices(w_avg, w, NUM_COLS, 1) 
        new_u_avg = matadd(u_avg, u) #add_matrices(u_avg, u, NUM_COLS, 1)
        mat_assign(w_avg, new_w_avg)
        mat_assign(u_avg, new_u_avg)


    #avg = c_fix(1.0 / NUM_PARTIES) cfix multiplication doesn't work

    if mat_type in [sfixMatrix, sintMatrix]:
        avg = sfix(1.0 / num_parties)  # Changing THIS line to cfix completely breaks everything wtf.
        threshold = l / (rho * num_parties)  #sfix(l/(rho * num_parties))
    else:
        avg = sfix_gc(1.0 / num_parties)
        threshold = sfix_gc(l/(rho * num_parties))

    """
    for i in range(w_avg.rows):
        print_ln("w_avg_mul: %s, w_avg: %s", (w_avg[i][0] * cfix(1.0 / num_parties)).reveal(), w_avg[i][0].reveal())
        print_ln("u_avg_mul: %s, u_avg: %s", (u_avg[i][0] * cfix(1.0 / num_parties)).reveal(), u_avg[i][0].reveal())
    """
    new_w_avg = mat_const_mul(avg, w_avg)
    new_u_avg = mat_const_mul(avg, u_avg)

    
    mat_assign(w_avg, new_w_avg)
    mat_assign(u_avg, new_u_avg)

    # Applying thresholding
    u_plus_w = matadd(w_avg, u_avg)
    z_new = soft_threshold_vec(threshold, u_plus_w, num_cols, mat_type)
    #u_list_new = s_fix_mat(num_parties * num_cols, 1)

    #neg_z = s_fix_mat(num_cols, 1)
    #u_list_new = sfixMatrix(num_parties * num_cols, 1)
    #neg_z = sfixMatrix(num_cols, 1)
    u_list_new = mat_type(num_parties * num_cols, 1)
    neg_z = mat_type(num_cols, 1)    
    for i in range(z_new.rows):
        for j in range(z_new.columns):
            neg_z[i][j] = z_new[i][j].__neg__()
            
    for i in range(num_parties):
        u_i = get_ith_matrix(u_list, i, num_cols, 1, mat_type)
        w_i = get_ith_matrix(w_list, i, num_cols, 1, mat_type)
        intermediate_vec = matadd(u_i, w_i) #add_matrices(u_i, w_i, NUM_COLS, 1)
        sum_vec = matadd(intermediate_vec, neg_z) #add_matrices(intermediate_vec, neg_z, NUM_COLS, 1)
        copy_ith_matrix(u_list_new, sum_vec, i, num_cols, 1)

    #reveal_all(z_new, "intermediate_weights")
    return u_list_new, z_new


def ADMM_preprocess(x_data, y_data, rho, num_parties, num_rows, num_cols, mat_type=sfixMatrix):
    #XTX_inv_lst = s_fix_mat(NUM_PARTIES * NUM_COLS, NUM_COLS)
    #XTy_lst = s_fix_mat(NUM_PARTIES * NUM_COLS, 1)
    #XTX_inv_lst = sfixMatrix(num_parties * num_cols, num_cols)
    #XTy_lst = sfixMatrix(num_parties * num_cols, 1)
    XTX_inv_lst = mat_type(num_parties * num_cols, num_cols)
    XTy_lst = mat_type(num_parties * num_cols, 1)

    for i in range(num_parties):
        x_i = get_ith_matrix(x_data, i, num_rows, num_cols, mat_type)
        y_i = get_ith_matrix(y_data, i, num_rows, 1, mat_type)
        X_T = transpose(x_i)
        XTy = matmul(X_T, y_i, num_cols, num_rows, num_rows, 1, sfix)
        XTX = matmul(X_T, x_i, num_cols, num_rows, num_rows, num_cols, sfix)

        #rho_identity = s_fix_mat(NUM_COLS, NUM_COLS)
        #rho_identity = sfixMatrix(num_cols, num_cols)
        rho_identity = mat_type(num_cols, num_cols)
        rho_identity = mat_const_mul(cfix(0), rho_identity)
        for j in range(num_cols):
            rho_identity[j][j] = rho #rho_val #sfix(rho_val)

        XTX_rho_identity = matadd(XTX, rho_identity) #add_matrices(XTX, rho_identity, NUM_COLS, NUM_COLS)
        XTX_inv = matinv(XTX_rho_identity)
        copy_ith_matrix(XTX_inv_lst, XTX_inv, i, num_cols, num_cols)
        copy_ith_matrix(XTy_lst, XTy, i, num_cols, 1)

    return XTX_inv_lst, XTy_lst

def ADMM(XTX_inv_lst, XTy_lst, admm_iter, num_parties, num_cols, rho, l):
    #XTX_inv_lst, XTy_lst = local_compute(x_data, y_data, num_parties. num_rows, num_cols)
    #w_list = s_fix_mat(num_parties * num_cols, 1)
    mat_type = None
    if isinstance(XTX_inv_lst, sfixMatrix):
        mat_type = sfixMatrix 
    elif isinstance(XTX_inv_lst, sfixMatrixGC):
        mat_type = sfixMatrixGC
    elif isinstance(XTX_inv_lst, sintMatrix):
        mat_type = sintMatrix
    else:
        raise ValueError("Type of matrix: {0} does not correspond to anything supported by this library".format(mat_type))

    #w_list = sfixMatrix(num_parties * num_cols, 1)
    #u_list = sfixMatrix(num_parties * num_cols, 1)
    #z = sfixMatrix(num_cols, 1)

    w_list = mat_type(num_parties * num_cols, 1)
    u_list = mat_type(num_parties * num_cols, 1)
    z = mat_type(num_cols, 1)
    w_list = mat_const_mul(cfix(0), w_list)
    u_list = mat_const_mul(cfix(0), u_list)
    z = mat_const_mul(cfix(0), z)
    """
    for i in range(w_list.rows):
        for j in range(w_list.columns):
            print_ln("%s, %s", w_list[i][j].reveal(), u_list[i][j].reveal())
    """
    for i in range(admm_iter):
        for j in range(num_parties):
            XTX_inv = get_ith_matrix(XTX_inv_lst, j, num_cols, num_cols, mat_type)
            XTy = get_ith_matrix(XTy_lst, j, num_cols, 1, mat_type)
            u = get_ith_matrix(u_list, j, num_cols, 1, mat_type)
            w = admm_local(XTX_inv, XTy, u, z, rho, num_cols)
            #reveal_all(w, "local_weight")
            copy_ith_matrix(w_list, w, j, num_cols, 1)

        new_u_lst, new_z = admm_coordinate(w_list, u_list, z, rho, l, num_cols, num_parties, mat_type)
        mat_assign(u_list, new_u_lst)
        mat_assign(z, new_z)

    return z



