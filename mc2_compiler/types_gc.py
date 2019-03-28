# (C) 2017 University of Bristol. See License.txt

from Compiler.types import MemValue, read_mem_value, regint, Array
from Compiler.program import Tape, Program
from Compiler.instructions_gc import *
import operator

class bits(object):
    global_gid = 0
    
    def __init__(self, if_input=False):
        self.gid = None
        if if_input:
            self.set_gid()

    def set_gid(self):
        self.gid = bits.global_gid
        bits.global_gid += 1

    def __str__(self):
        return str(self.gid)

class cbits(bits):
    reg_type = 'cb'
    is_clear = True

    def __init__(self, value):
        super(cbits, self).__init__()
        if (value != 0) and (value != 1):
            raise ValueError("cbits must have a value of 0 or 1")
        self.value = value

    def __invert__(self):
        return cbits((self.value + 1) % 2)

    def __xor__(self, other):
        if isinstance(other, cbits):
            return cbits(self.value ^ other.value)
        else:
            return NotImplemented

    def __and__(self, other):
        if isinstance(other, cbits):
            return cbits(self.value & other.value)
        else:
            return NotImplemented

    __rxor__ = __xor__
    __rand__ = __and__
    
class sbits(bits):

    def __init__(self, if_input=False):
        super(sbits, self).__init__(if_input)
    
    def secret_op(self, other, s_inst):
        return

    def _and(self, other):
        res = sbits()
        and_gc(res, self, other)
        res.set_gid()
        return res

    def _xor(self, other):
        res = sbits()
        xor_gc(res, self, other)
        res.set_gid()
        return res

    def _invert(self):
        res = sbits()
        invert_gc(res, self)
        res.set_gid()
        return res
    
    def __invert__(self):
        return self._invert()

    def __xor__(self, other):
        if isinstance(other, cbits):
            if other.value == 0:
                return self
            else:
                return self.__invert__()
        elif isinstance(other, sbits):
            return self._xor(other)
        else:
            return NotImplemented

    def __and__(self, other):
        if isinstance(other, cbits):
            if other.value == 0:
                return cbits(0)
            else:
                return self
        elif isinstance(other, sbits):
            return self._and(other)
        else:
            return NotImplemented

    def __or__(self, other):
        a = self & other
        b = self ^ other
        c = b ^ a
        return c

    __rxor__ = __xor__
    __rand__ = __and__
    __ror__ = __or__
        

# Logic copied from AG-MPC
# https://github.com/emp-toolkit/emp-tool/blob/stable/emp-tool/circuits/integer.hpp
#
# Note that this is an integer representation, not modulo p

def add_full(dest, op1, op2, size, carry_in=None, carry_out=None):
    if size == 0:
        if carry_in and carry_out:
            return carry_in
        else:
            return None

    carry = carry_in
    if carry is None:
        carry = cbits(0)

    skip_last = int(carry_out == None)

    i = 0
    while (size > skip_last):
        axc = op1[i] ^ carry
        bxc = op2[i] ^ carry
        dest[i] = op1[i] ^ bxc
        t = axc & bxc
        carry = carry ^ t
        i += 1
        size -= 1

    if carry_out is None:
        dest[i] = carry ^ op2[i] ^ op1[i]
        return None
    else:
        # return carry out, since we cannot assign that value in this function
        return carry

def sub_full(dest, op1, op2, size, borrow_in=None, borrow_out=None):
    if size == 0:
        if borrow_in and borrow_out:
            return borrow_in
        else:
            return None

    borrow = borrow_in
    if borrow is None:
        borrow = cbits(0)

    skip_last = int(borrow_out == None)

    i = 0
    while size > skip_last:
        bxa = op1[i] ^ op2[i]
        bxc = borrow ^ op2[i]
        dest[i] = bxa ^ borrow
        t = bxa & bxc
        borrow = borrow ^ t

        i += 1
        size -= 1

    if borrow_out is None:
        dest[i] = op1[i] ^ op2[i] ^ borrow
        return None
    else:
        return borrow

def mul_full(dest, op1, op2, size):
    s = []
    t = []

    for i in range(0, size):
        s.append(cbits(0))
        t.append(None)

    for i in range(0, size):
        for k in range(0, size - i):
            t[k] = op1[k] & op2[i]
        s2 = sint_gc(size - i)
        add_full(s2.bits, s[i:], t, size - i)
        for j in range(0, size - i):
            s[i + j] = s2.bits[j]
        
    for i in range(0, size):
        dest[i] = s[i]

def if_then_else(dest, tsrc, fsrc, size, cond):
    i = 0
    while (size > 0):
        x = tsrc[i] ^ fsrc[i]
        a = cond & x
        dest[i] = a ^ fsrc[i]
        i += 1
        size -= 1

def cond_neg(cond, dest, src, size):
    c = cond

    i = 0
    for j in range(0, size - 1):
        dest[i] = src[i] ^ cond
        t = dest[i] ^ c
        c = c & dest[i]
        dest[i] = t
        i += 1

    dest[i] = cond ^ c ^ src[i]

def div_full(vquot, op1, op2, size, vrem=None):
    overflow = [sbits()] * size
    temp = [sbits()] * size
    rem = [sbits()] * size
    quot = [sbits()] * size
    b = sbits()

    for i in range(0, size):
        rem[i] = op1[i]

    overflow[0] = cbits(0)

    for i in range(1, size):
        overflow[i] = overflow[i - 1] | op2[size - i]
    for i in range(size - 1, -1, -1):
        b = sub_full(temp, rem[i:], op2, size-i, borrow_out=b)
        gc_nop("end sub {}".format(i))
        b = b | overflow[i]
        
        rem_temp = [sbits() for j in range(i, size)]
        if_then_else(rem_temp, rem[i:], temp, size-i, b)
        for j in range(i, size):
            rem[j] = rem_temp[j-i]
        
        quot[i] = ~b

    for i in range(0, size):
        if vrem is not None:
            vrem[i] = rem[i]
        vquot[i] = quot[i]

class sint_gc(object):
    value_type = sbits
    def __init__(self, length, if_input=False):
        self.bits = []
        assert(length > 0)
        self.length = length
        for i in range(self.length):
            self.bits.append(sbits(if_input))

    def test_instance(self, other):
        if not isinstance(other, sint_gc):
            return NotImplemented
        if self.length != other.length:
            return NotImplemented

    def __getitem__(self, index):
        if index >= self.length:
            raise CompilerError("Index exceeds array length")
        return self.bits[index]

    def __and__(self, other):
        self.test_instance(other)

        dest = sint_gc(self.length, 0)
        for i in range(self.length):
            dest.bits[i] = self.bits[i] & other.bits[i]

        return dest

    def __xor__(self, other):
        self.test_instance(other)

        dest = sint_gc(self.length, 0)
        for i in range(self.length):
            dest.bits[i] = self.bits[i] ^ other.bits[i]

        return dest

    __rand__ = __and__
    __rxor__ = __xor__

    # we don't do any carrying here
    def __add__(self, other):
        self.test_instance(other)
        dest = sint_gc(self.length)
        add_full(dest.bits, self.bits, other.bits, self.length)
        return dest

    def __sub__(self, other):
        self.test_instance(other)
        dest = sint_gc(self.length)
        sub_full(dest.bits, self.bits, other.bits, self.length)
        return dest

    def absolute(self):
        dest = sint_gc(self.length)
        for i in range(0, self.length):
            dest.bits[i] = self.bits[self.length-1]
        return (self + dest) ^ dest
    
    def __mul__(self, other):
        self.test_instance(other)
        dest = sint_gc(self.length)
        mul_full(dest.bits, self.bits, other.bits, self.length)
        return dest

    def __div__(self, other):
        self.test_instance(other)
        dest = sint_gc(self.length)
        i1 = self.absolute()
        i2 = other.absolute()
        gc_nop("end absolute value")
        sign = self.bits[self.length - 1] ^ other.bits[other.length - 1]
        gc_nop("end sign")
        div_full(dest.bits, i1.bits, i2.bits, self.length)
        gc_nop("end division")
        dest_temp = [sbits(0) for i in range(self.length)]
	cond_neg(sign, dest_temp, dest.bits, self.length)
        for i in range(len(dest_temp)):
            dest.bits[i] = dest_temp[i]
        return dest

    def __geq__(self, other):
        self.test_instance(other)

        res = sint_gc()
        res = self - other

        return ~res[self.length - 1]

    def __str__(self):
        s = ""
        for i in range(len(self.bits)):
            s += "{}: {}\n".format(i, self.bits[i])

        return s
