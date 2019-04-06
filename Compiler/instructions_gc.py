# (C) 2017 University of Bristol. See License.txt

import Compiler.instructions_base as base
import Compiler.instructions as spdz
import Compiler.tools as tools
import collections
import itertools

class SecretBitsAF(base.RegisterArgFormat):
    reg_type = 'sb'
class ClearBitsAF(base.RegisterArgFormat):
    reg_type = 'cb'

base.ArgFormats['sb'] = SecretBitsAF
base.ArgFormats['sbw'] = SecretBitsAF
base.ArgFormats['cb'] = ClearBitsAF
base.ArgFormats['cbw'] = ClearBitsAF

opcodes_gc = dict(
    INV = 0x1,
    XOR = 0x2,
    AND = 0x3,
)

class InstructionGC(object):
    def __init__(self, *args):
        self.args = args
        program_gc.curr_block.instructions.append(self)
        
    def check_args(self):
        for arg in self.args:
            #assert(arg.gid is not None)
            pass

class invert_gc(InstructionGC):
    __slots__ = []
    code = opcodes_gc["INV"]
    arg_format = ['sb', 'sb']

    def __str__(self):
        self.check_args()
        return "1 1 {} {} INV".format(self.args[1], self.args[0])

class xor_gc(InstructionGC):
    __slots__ = []
    code = opcodes_gc["XOR"]
    arg_format = ['sb', 'sb', 'sb']

    def __str__(self):
        self.check_args()
        return "2 1 {} {} {} XOR".format(self.args[1], self.args[2], self.args[0])

class and_gc(InstructionGC):
    __slots__ = []
    code = opcodes_gc["AND"]
    arg_format = ['sb', 'sb', 'sb']

    def __str__(self):
        self.check_args()
        return "2 1 {} {} {} AND".format(self.args[1], self.args[2], self.args[0])

# Used for debugging circuit files
class gc_nop(InstructionGC):
    def __str__(self):
        return "{}".format(self.args[0])
