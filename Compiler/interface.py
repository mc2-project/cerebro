from types import *
from types_gc import *
import compilerLib

SPDZ = 0
GC = 1

class Params(object):
    intp = 32
    f = 32
    k = 32

    @classmethod
    def set_params(cls, int_precision=32, f=32, k=64):
        cls.intp = int_precision
        cls.f = f
        cls.k = k
        sfix.f = f
        sfix.k = k

class ClearIntegerFactory(object):
    def __call__(self, value):
        if mpc_type == SPDZ:
            return cint(value)
        else:
            return cint_gc(Params.intp, value)
        
class SecretIntegerFactory(object):
    def __call__(self, party):
        if mpc_type == SPDZ:
            return sint.get_private_input_from(party)
        else:
            return sint_gc(Params.intp, party)
    
    def read_input(self, party):
        if mpc_type == SPDZ:
            return sint.get_private_input_from(party)
        else:
            return sint_gc(Params.intp, party)

class SecretFixedPointFactory(object):
    def __call__(self, party):
        if mpc_type == SPDZ:
            return sint.get_private_input_from(party)
        else:
            return sint_gc(Params.intp, party)
    
    def read_input(self, party):
        if mpc_type == SPDZ:
            return sint.get_private_input_from(party)
        else:
            return sint_gc(Params.intp, party)

ClearInteger = ClearIntegerFactory()
SecretInteger = SecretIntegerFactory()
SecretFixedPoint = SecretFixedPointFactory()
compilerLib.VARS["c_int"] = ClearInteger
compilerLib.VARS["s_int"] = SecretInteger
compilerLib.VARS["s_fix"] = SecretFixedPoint
