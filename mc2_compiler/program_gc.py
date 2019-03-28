# (C) 2017 University of Bristol. See License.txt

from Compiler.config import *
from Compiler.exceptions import *
import compilerLib
import allocator as al
import random
import time
import sys, os, errno
import inspect
from collections import defaultdict
import itertools
import math
import Compiler.instructions_gc
        
class ProgramGC(object):
    
    class BasicBlock(object):
        def __init__(self):
            self.instructions = []
        
    def __init__(self, args, options, param=-1):
        self.activeblock = ProgramGC.BasicBlock()
        self.init_names(args)

    def init_names(self, args):
        self.programs_dir=args[0];
        print 'Compiling program in', self.programs_dir
        if  self.programs_dir.endswith('/'):
           self.programs_dir = self.programs_dir[:-1]
        progname = self.programs_dir.split('/')[-1]
        
        if progname.endswith('.mpc'):
            progname = progname[:-4]
        
        self.infile = self.programs_dir + '/' + progname + '.mpc'
        print progname
        """
        self.name is input file name (minus extension) + any optional arguments.
        Used to generate output filenames
        """
        self.name = progname
        if len(args) > 1:
            self.name += '-' + '-'.join(args[1:])

    @property
    def curr_block(self):
        return self.activeblock

    def write_bytes(self, outfile=None):
        for i in self.activeblock.instructions:
            print i
