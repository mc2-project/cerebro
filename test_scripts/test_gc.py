import unittest
import struct
import inspect
import gmpy2
import sys
import operator
import math
import os
import subprocess
from pathlib import Path
from optparse import OptionParser


class Preprocessing:
	cwd = os.getcwd()
	root = cwd
	def compile(self, program_name, compile_options):
		#os.system("python compile.py a " + program_name)
		status_code = subprocess.call("python compile.py b {0} {1}".format(compile_options, program_name), shell=True)
		if status_code != 0:
			assert(False)

	def run_online(self, program_name):
		p1 = subprocess.Popen(["./bin/run_circuit.x 1 5000 {0} {1} {2}".format(program_name)], shell=True)
		p2 = subprocess.Popen(["./bin/run_circuit.x 2 5000 {0} {1} {2}".format(program_name)], shell=True)
		exit_codes = [p.wait() for p in p1, p2]
		# Exit code of non-zero means failure.
		if exit_codes[0] != 0 or exit_codes[1] != 0:
			assert(False)

	# Need for compiler to have function write in line.
	def run_test(self, program_name):
		#status_code = subprocess.call(["python test_scripts/test-result.py {}".format(program_name)], shell=True)
		# Read output file 
		output_file = os.path.join(program_name, "agmpc.output")
		outputs = []
		with open(output_file, "rb") as f:
			# Length is 4 bytes 
			len_file = f.read(4)
			print "LEN OUTPUT FILE: ", len_file
			# Assume length is multiple of 8.
			for i in range(len_file // 8):
				byte = struct.unpack('B',f.read(1))[0]
				outputs.append(byte)


		print outputs


class TestGC(unittest.TestCase):
	preprocessing = Preprocessing()
	
	def test_cond(self):
		test_name = 'test_cond'
		program_name = 'Programs/%s' % (test_name)
		self.preprocessing.compile(program_name, '')
		self.preprocessing.run_online(program_name)
		self.preprocessing.run_test(test_name)
	

	

	def test_unroll(self):
		test_name = 'test_unroll'
		program_name = 'Programs/%s' % (test_name)
		self.preprocessing.compile(program_name, '-ur')
		self.preprocessing.run_online(program_name)
		self.preprocessing.run_test(test_name)



	def test_inline(self):
		test_name = 'test_inline'
		program_name = 'Programs/%s' % (test_name)
		self.preprocessing.compile(program_name, "-in")
		self.preprocessing.run_online(program_name)
		self.preprocessing.run_test(test_name)

	def test_fused(self):
		test_name = 'test_fused'
		program_name = 'Programs/%s' % (test_name)
		self.preprocessing.compile(program_name, "")
		self.preprocessing.run_online(program_name)
		self.preprocessing.run_test(test_name)

	



if __name__=="__main__":
	unittest.main()



