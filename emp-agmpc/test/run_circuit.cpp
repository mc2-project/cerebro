#include <fstream>
#include <string>
#include <vector>

#include <emp-tool/emp-tool.h>
#include "emp-agmpc/emp-agmpc.h"

// This function parses the circuit input file to determine
// which input wires belong to which party
std::pair<uint64_t, uint64_t> parse_circuit_input(uint64_t party_id,
                                                  std::string filename,
                                                  std::vector<std::pair<uint64_t, uint64_t> > &input_wires) {
  
  std::ifstream infile(filename.c_str());
  uint64_t num_inputs = 0, num_outputs = 0;
  infile >> num_inputs >> num_outputs;
  uint64_t party = 0, begin = 0, end = 0;
  while (infile >> party >> begin >> end) {
    if (party != party_id) {
      continue;
    }

    input_wires.push_back(std::make_pair<uint64_t, uin64_t>(begin, end));
  }

  return std::make_pair<uint64_t, uint64_t>(num_inputs, num_outputs);
}

// This function reads the input from "filename" and sets the data in "input"
// Each party sets a value for every input wire
// If a party does not provide real input for a wire, the value is set to be 0
void create_circuit_input(bool *input, std::vector<std::pair<uint64_t, uint64_t> > &input_wires, std::string filename) {
  std::ifstream infile(filename.c_str());
  for (size_t i = 0; i < input_wires.size(); i++) {
    uint64_t offset = input_wires.at(i).first;
    uint64_t length = input_wires.at(i).second - offset;
    infile.read(input + offset, length);
  }
}

int main(int argc, char **argv) {
  if (argc != 7) {
    printf("USAGE: %s <party_id> <num_parties> <port_base> <circuit_file> <input_format_file> <input_file>\n", argv[0]);
    return 1;
  }

  // Define some constants for the circuit setup.
  const static int party_id = atoi(argv[1]);
  const static int num_parties = atoi(argv[2]);
  const static int port_base  = atoi(argv[3]);
  std::string circuit_file(argv[4]);
  std::string input_format_file(argv[5]);
  std::string input_file(argv[6]);
  const static int bit_length = 32;

  // Input circuit file
  CircuitFile cf(circuit_file.c_str());

  // Parse input
  std::vector<std::pair<uint64_t, uint64_t> > input_wires;
  std::pair<uint64_t, uint64_t> num_wires = parse_circuit_input(party_id, input_format_file, inputs);
  bool *input = new bool[num_wires[0]];
  memset(input, false, input_wires);
  create_circuit_input(input, input_wires, input_file);
  bool *output = new bool[num_wires[1]];

  NetIOMP<nP> *io_list[num_parties];
  for (uint64_t i = 0; i < num_parties; i++) {
    io_list[i] = new NetIOMP<num_parties>(party_id, port_base + i);
  }

  ThreadPool pool(2 * num_parties);
  CMPC<num_parties>* mpc = new CMPC<num_parties>(io_list, &pool, party_id, &cf);

  // Run the secure computation
  mpc->function_independent();
  mpc->function_dependent();
  mpc->online(input, output, {0}, {num_wires[0]});

  // Clean up memory
  for (uint64_t i = 0; i < num_parties; i++) {
    delete io_list[i];
  }

  delete input;
  delete output;
  delete mpc;

  return 0;
}
