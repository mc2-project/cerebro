#!/usr/bin/env python


#     ===== Compiler usage instructions =====
# 
# See documentation for details on the Compiler package


import argparse
import json, subprocess, shlex, os, yaml

import Compiler
import Compiler.planning as planning


def main():
    parser = argparse.ArgumentParser(description="A compiler/planner for secure multi-party computation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")

    # Temporarily disable splitting since it is a work-in-progress
    parser.add_argument("-sp", "--split", action="store_false", default=False, dest="split", help="Whether or not to split the program")
    parser.add_argument("-ur", "--unroll", action="store_true", default=True, dest="unroll", help="Whether or not to unroll a loop")
    parser.add_argument("-in", "--inline", action="store_true", default=False, dest="inline", help="Whether or not to inline functions")

    args = parser.parse_args()
    config_file = open(args.config, 'r')
    config = yaml.load(config_file.read(), Loader=yaml.SafeLoader)
    config_file.close()

    party_id = config["party_id"]
    constants_file = config["constants_file"]
    program_dir = config["program_dir"]
    program_name = config["program_name"]

    options = args
    options.party = party_id
    decision = Compiler.plan(program_dir + "/" + program_name, constants_file, options)

    # write out the final plan to a file
    plan = {}
    plan["decision"] = decision
    plan_json = json.dumps(plan)

    plan_file = config["plan_file"]

    f = open(plan_file, 'w')
    f.write(plan_json)
    f.close()
    
if __name__ == '__main__':
    main()
