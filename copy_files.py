import argparse
import subprocess
import shlex


def main():
	parser = argparse.ArgumentParser(description='Copy some files to ssh machines')
	
	parser.add_argument('--key', '-k', type=str, help="Key file")


	#parser.add_argument('key', type=str, help="Specify key file used to ssh into machine")
	parser.add_argument('--file', '-f', type=str, help="Specify file that want to be copied onto ssh machines")
	parser.add_argument('--ssh_path', '-s', type=str, help="Destination file path in remote ssh server")
	parser.add_argument('--ip', '-i', nargs='+', help="List of IP Addresses in which you want to copy the file")



	args = parser.parse_args()
	#print(args.key)
	#print(args.file)
	#print(args.ssh_path)
	#print(args.ip)


	lst_commands = construct_command(args.key, args.file, args.ssh_path, args.ip)

	exec_command(lst_commands)

def construct_command(key, file, ssh_path, lst_ips):
	assert(file != None)
	cmd = "scp "
	
	if (key):
		cmd += " -i " + key + " "

	cmd += file + " "

	lst_commands = [cmd + lst_ips[i] + ":" + ssh_path for i in range(len(lst_ips))]


	print(lst_commands)
	return lst_commands


def exec_command(lst_commands):
	for cmd in lst_commands:
		print(shlex.split(cmd))
		#subprocess.run(cmd)
		subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)


if __name__ == "__main__":
	main()


