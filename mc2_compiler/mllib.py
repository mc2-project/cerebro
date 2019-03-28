from Compiler.types import cint,sint,pint,cfix,sfix,sfloat,cfloat,MPCThread,Array,MemValue,_number,_mem,_register,regint,Matrix,_types
from Compiler.types import sintArray, sintMatrix, cintArray, cintMatrix, MixMatrix
from Compiler.instructions import *
from Compiler.util import tuplify,untuplify
from Compiler import instructions,instructions_base,comparison,program
import inspect,math
import random
import collections
from Compiler.library import *

from operator import itemgetter

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
        @for_range_multithread(nparallel, nparallel, exec_len)
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

def transpose(A):
    if not isinstance(A, Matrix):
        raise ValueError("Matrix can only be ")
    
    B = A.__class__(A.columns, A.rows)
    
    return _transpose(A, B)


def _matmul(A, B, C, D, int_type, nparallel=1):
    @for_range_multithread(nparallel, nparallel, A.rows * B.columns * A.columns)
    def _multiply(i):
        i_index = i / (B.columns * A.columns)
        j_index = i % (B.columns * A.columns) / (A.columns)
        k_index = i % A.columns

        D[i] = A[i_index][k_index] * B[k_index][j_index]

    @for_range_multithread(nparallel, nparallel, A.rows * B.columns)
    def _add(i):
        i_index = i / B.columns
        j_index = i % B.columns
        C[i_index][j_index] = int_type(0)
            
        @for_range(A.columns)
        def _add_element(j):
            C[i_index][j_index] += D[i * A.columns + j]

    return C


def _matmul_mix(A, B, nparallel=1):
    C = MixMatrix(A.rows, B.columns)
    print(A.rows, B.columns)
    @for_range(A.rows * B.columns)
    def f(i):
        @for_range(A.columns)
        def g(j):
            v = C.get(i)
            v += A.get(i * A.columns + j) * B.get(j * B.columns + i)
            C.set(i, v)

    return C

def matmul(A, B, nparallel=1):
    if isinstance(A, sintMatrix) and isinstance(B, sintMatrix):
        C = sintMatrix(A.rows, B.columns)
        D = sintArray(A.rows * B.columns * A.columns)
        return _matmul(A, B, C, D, sint, nparallel)
    elif isinstance(A, cintMatrix) and isinstance(B, cintMatrix):
        C = cintMatrix(A.rows, B.columns)
        D = cintArray(A.rows * B.columns * A.columns)
        return _matmul(A, B, C, D, cint, nparallel)
    elif isinstance(A, sfixMatrix) and isinstance(B, sfixMatrix):
        C = sfixMatrix(A.rows, B.columns)
        D = sfixArray(A.rows * B.columns * A.columns)
        return _matmul(A, B, C, D, sfix, nparallel)
    elif isinstance(A, MixMatrix) and isinstance(B, MixMatrix):
        return _matmul_mix(A, B, nparallel)
    else:
        return NotImplemented

def _matadd(A, B, C, int_type, nparallel=1):
    @for_range_multithread(nparallel, nparallel, A.rows * A.columns)
    def _add(i):
        i_index = i / A.columns
        j_index = i % A.columns
        
        C[i_index][j_index] = A[i_index][j_index] + B[i_index][j_index]
    
def matadd(A, B, nparallel=1):
    if A.rows != B.rows or A.columns != B.columns:
        raise NotImplemented
    
    if isinstance(A, cintMatrix) and isinstance(B, cintMatrix):
        C = cintMatrix(A.rows, A.columns)
        return _matadd(A, B, C, cint, nparallel)
    elif isinstance(A, sintMatrix) and isinstance(B, sintMatrix):
        C = sintMatrix(A.rows, A.columns)
        return _matadd(A, B, C, sint, nparallel)

def _matsub(A, B, C, int_type, nparallel=1):
    @for_range_multithread(nparallel, nparallel, A.rows * A.columns)
    def _add(i):
        i_index = i / A.columns
        j_index = i % A.columns
        
        C[i_index][j_index] = A[i_index][j_index] - B[i_index][j_index]
        
def matsub(A, B, nparallel=1):
    if A.rows != B.rows or A.columns != B.columns:
        raise ValueError("[matsub] Matrices must have the same sizes")
    
    if isinstance(A, cintMatrix) and isinstance(B, cintMatrix):
        C = cintMatrix(A.rows, A.columns)
        return _matsub(A, B, C, cint, nparallel)
    elif isinstance(A, sintMatrix) and isinstance(B, sintMatrix):
        C = sintMatrix(A.rows, A.columns)
        return _matsub(A, B, C, sint, nparallel)
    elif isinstance(A, sfixMatrix) and isinstance(B, sfixMatrix):
        C = sfixMatrix(A.rows, A.columns)
        return _matsub(A, B, C, sfix, nparallel)
    else:
        return NotImplemented


# horizontally stack the input matrices
def matstack(matrices):
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

def _sigmoid_sfix(v):
    res = sfix()
    sign_v = sfix(v < 0)
    denom = 1 + sign_v * v
    res = v / denom
    return res

def sigmoid(v, nparallel=1):
    if isinstance(v, sfix):
        return _sigmoid_sfix(v)
    elif isinstance(v, sfixMatrix):
        res = sfixMatrix(v.rows, v.columns)
        @for_range_multithread(v.rows, nparallel, nparallel)
        def a(i):
            @for_range_multithread(v.columns, nparallel, nparallel)
            def b(j):
                res[i][j] = _sigmoid_sfix(v[i][j])
        return res
    else:
        return NotImplemented

def mat_const_mul(c, m, nparallel=1):
    if isinstance(m, sfixMatrix):
        res = sfixMatrix(m.rows, m.columns)
        @for_range_multithread(m.rows, nparallel, nparallel)
        def f(i):
            @for_range_multithread(m.columns, nparallel, nparalle)
            def g(j):
                res[i][j] = c * m[i][j]
    else:
        return NotImplemented

def mat_assign(o, i, nparallel=1):
    if o.rows != i.rows or o.columns != i.columns:
        raise ValueError("Matrices must be of the same sizes")
    
    @for_range_multithread(i.rows, nparallel, nparallel)
    def f(i):
        @for_range_multithread(i.columns, nparallel, nparalle)
        def g(j):
            o[i][j] = i[i][j]

def array_index_secret(l, index, nparallel=1):
    if instance(l, Array) and isinstance(index, sint):
        res_list = type(l).__init__(l.length)
        @for_range_multithread(l.length, nparallel, nparallel)
        def f(i):
            v = sint(i).__eq__(index)
            res_list[i] = v * l[i]

        res = l.value_type(0)
        @for_range(l.length)
        def f(i):
            res += res_list[i]
        return res
    
    else:
        return NotImplemented

def array_index_secret_if(condition, l, index_1, index_2, nparallel=1):
    if isinstance(index_1, sint) and isinstance(index_2, sint): 
        index = condition * index_1 + (1 - condition) * index_2
        return array_index_secret(l, index, nparallel)
    else:
        return NotImplemented
