#!/usr/bin/env python


#     ===== Compiler usage instructions =====
# 
# See documentation for details on the Compiler package


import argparse
import json, subprocess, shlex, os

import Compiler
import Compiler.planning as planning


def main():
    parser = argparse.ArgumentParser(description="A compiler/planner for secure multi-party computation")
    parser.add_argument("constants_file", type=str, help="Path to the file for the constants in the cost model")
    parser.add_argument("party", type=str, help="Party number (e.g., 0, 1, etc.)")
    parser.add_argument('program', type=str, help="Name of the .mpc program; file should be placed in Programs")
    parser.add_argument("--plan", dest="plan_file", default="", help="Path to the output plan")

    # Temporarily disable splitting since it is a work-in-progress
    parser.add_argument("-sp", "--split", action="store_false", default=False, dest="split", help="Whether or not to split the program")
    parser.add_argument("-ur", "--unroll", action="store_true", default=True, dest="unroll", help="Whether or not to unroll a loop")
    parser.add_argument("-in", "--inline", action="store_true", default=False, dest="inline", help="Whether or not to inline functions")

    args = parser.parse_args()
    options = args
    party_id = options.party
    constants_file = options.constants_file
    program_name = options.program
    decision = Compiler.plan(program_name, constants_file, options)

    # write out the final plan to a file
    plan = {}
    plan["decision"] = decision
    plan["program"] = program_name
    plan["party_id"] = party_id
    plan_json = json.dumps(plan)

    plan_file = args.plan_file
    if plan_file == "":
        plan_file = "./mpc_exec_plan"

    f = open(plan_file, 'w')
    f.write(plan_json)
    f.close()
    
if __name__ == '__main__':
    main()
