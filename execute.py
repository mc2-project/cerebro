import argparse
import json
import subprocess, shlex, os

def process_input(input_filename):
    def f(s, x, start=0):
        pos = s.find(x, start)
        if pos == -1:
            return start-1
        return f(s, x, pos+1)

    split = f(input_filename, "/")
    directory = input_filename[:split]
    filename = input_filename[split+1:-3]
    cmd = "python Input_Data/gen_data.py {} {}".format(directory, filename)
    proc = subprocess.Popen(shlex.split(cmd))
    print "Finished processing input"
    return 


def execute_framework(decision, party_id, program_name):
    root_dir = os.getcwd()
    print "Executing {} framework for program {}".format(decision, program_name)
    proc = None
    if decision == "arithmetic":
        os.chdir("./crypto_backend/SCALE-MAMBA/")
        proc = subprocess.Popen(shlex.split("./Player {} {}".format(party_id, program_name)))
    elif decision == "boolean":
        os.chdir("./crypto_backend/emp-toolkit/emp-agmpc/")
        proc = subprocess.Popen(shlex.split("./bin/run_circuit {} 2000 {} {} {} ".format(party_id, program_name, program_name, program_name)))
    else:
        raise ValueError("Framework {} is not supported as a valid execution backend".format(decision))

    print proc.communicate()[0]
    print "Finished executing {} framework for program {}".format(decision, program_name)
    os.chdir(root_dir)

def process_output(decision, program_name):
    if decision == "boolean":
        proc = subprocess.Popen(shlex.split("python Output_Data/agmpc_output_parser.py {}".format(program_name + "/agmpc.txt.output")))
        print proc.communicate()[0]

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_filename", type=str, help="Path to the inputs")
    parser.add_argument("--plan", type=str, default="", dest="plan_file", help="""Path to the file with the plan; default file is "./mpc_exec_plan" """)

    args = parser.parse_args()
    plan_file = args.plan_file
    if plan_file == "":
        plan_file = "./mpc_exec_plan"

    f = open(plan_file, 'r')
    plan_json = f.read()
    f.close()
    plan = json.loads(plan_json)

    decision = plan["decision"]
    program = plan["program"]
    party_id = plan["party_id"]
    
    process_input(args.input_filename)
    execute_framework(decision, party_id, program)
    process_output()

if __name__ == "__main__":
    main()
