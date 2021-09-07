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
		status_code = subprocess.call("python compile.py a {0} {1}".format(compile_options, program_name), shell=True)
		if status_code != 0:
			assert(False)

	def gen_data(self, program_name):
		program_name = program_name.rsplit('/', 1)[-1]
		status_code = subprocess.call("cd Input_Data && python gen_data.py ./ {0}".format(program_name), shell=True)
		if status_code != 0:
			assert(False)

	def run_online(self, program_name):
		p0 = subprocess.Popen(["./Player.x 0 {}".format(program_name)], shell=True)
		p1 = subprocess.Popen(["./Player.x 1 {}".format(program_name)], shell=True)

		exit_codes = [p.wait() for p in p0, p1]

		# Exit code of non-zero means failure.
		if exit_codes[0] != 0 or exit_codes[1] != 0:
			assert(False)


	def run_test(self, program_name):
		status_code = subprocess.call(["python test_scripts/test-result.py {}".format(program_name)], shell=True)
		if status_code != 0:
			assert(False)


class TestScaleMamba(unittest.TestCase):
	preprocessing = Preprocessing()

	def test_cond(self):
		test_name = 'test_multi_cond'
		program_name = 'Programs/%s' % (test_name)
		self.preprocessing.compile(program_name, '')
		self.preprocessing.gen_data(program_name)
		self.preprocessing.run_online(program_name)
		self.preprocessing.run_test(test_name)
	
	def test_cond(self):
		test_name = 'test_cond'
		program_name = 'Programs/%s' % (test_name)
		self.preprocessing.compile(program_name, '')
		self.preprocessing.gen_data(program_name)
		self.preprocessing.run_online(program_name)
		self.preprocessing.run_test(test_name)
	
	def test_unroll(self):
		test_name = 'test_unroll'
		program_name = 'Programs/%s' % (test_name)
		self.preprocessing.compile(program_name, '-ur')
		self.preprocessing.gen_data(program_name)
		self.preprocessing.run_online(program_name)
		self.preprocessing.run_test(test_name)


	
	def test_inline(self):
		test_name = 'test_inline'
		program_name = 'Programs/%s' % (test_name)
		self.preprocessing.compile(program_name, "-in")
		self.preprocessing.gen_data(program_name)
		self.preprocessing.run_online(program_name)
		self.preprocessing.run_test(test_name)


	def test_fused(self):
		test_name = 'test_fused'
		program_name = 'Programs/%s' % (test_name)
		self.preprocessing.compile(program_name, "")
		self.preprocessing.gen_data(program_name)
		self.preprocessing.run_online(program_name)
		self.preprocessing.run_test(test_name)

	



if __name__=="__main__":
	unittest.main()



