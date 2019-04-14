#include <stdint.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <emp-tool/emp-tool.h>
#include "emp-agmpc/emp-agmpc.h"


const static int num_parties = NUM_PARTY_FOR_RUNNING;
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
    if (party + 1 != party_id) {
      continue;
    }

    input_wires.push_back(std::make_pair(begin, end));
  }

  return std::make_pair(num_inputs, num_outputs);
}

// This function reads the input from "filename" and sets the data in "input"
// Each party sets a value for every input wire
// If a party does not provide real input for a wire, the value is set to be 0
void create_circuit_input(bool *input, std::vector<std::pair<uint64_t, uint64_t> > &input_wires, std::string filename) {
  uint8_t buffer[1];
  std::ifstream infile;
  infile.open(filename.c_str());
  for (size_t i = 0; i < input_wires.size(); i++) {
    uint64_t offset = input_wires.at(i).first;
    uint64_t length = input_wires.at(i).second + 1 - offset;

    uint64_t counter = length - 1;
    for (size_t j = 0; j < length / 8; j++) {
      infile.read((char *) buffer, 1);
      for (size_t k = 0; k < 8; k++) {
        size_t shift = 7 - k;
        input[counter] = ((buffer[0] & (1 << shift)) > 0);
        counter -= 1;
      }
    }

    if (length % 8 > 0) {
      infile.read((char *) buffer, 1);
      for (size_t k = 0; k < length % 8; k++) {
        size_t shift = 7 - k;
        input[counter] = ((buffer[0] & (1 << shift)) > 0);
        counter -= 1;
      }
    }
  }
  infile.close();
}

void bench_once(NetIOMP<num_parties> * ios[2], ThreadPool * pool,
                string circuit_file,
                string output_file,
                int party, bool *in, bool *out) {
  
  if(party == 1)cout <<"CIRCUIT:\t"<<circuit_file<<endl;
  CircuitFile cf(circuit_file.c_str());

  std::cout << cf.n1 << cf.n2 << cf.n3 << std::endl;

  auto start = clock_start();
  CMPC<num_parties>* mpc = new CMPC<num_parties>(ios, pool, party, &cf);
  ios[0]->flush();
  ios[1]->flush();
  double t2 = time_from(start);
  cout <<"Setup:\t"<<party<<"\t"<< t2 <<"\n"<<flush;

  start = clock_start();
  mpc->function_independent();
  ios[0]->flush();
  ios[1]->flush();
  t2 = time_from(start);
  cout <<"FUNC_IND:\t"<<party<<"\t"<<t2<<" \n"<<flush;

  start = clock_start();
  mpc->function_dependent();
  ios[0]->flush();
  ios[1]->flush();
  t2 = time_from(start);
  if(party == 1)cout <<"FUNC_DEP:\t"<<party<<"\t"<<t2<<" \n"<<flush;

  start = clock_start();
  mpc->online(in, out);
  ios[0]->flush();
  ios[1]->flush();
  t2 = time_from(start);

  if (party == 1) {
    for (int i = 0; i < cf.n3; i++) {
      cout << "output[" << i << "] = " << out[i] << std::endl;
    }
  }

  if (party == 1) {
    // Write the out bits to a file
    // Note that we are writing bytes at a time
    std::cout << "Writing to output file" << output_file << std::endl;
    fstream outfile;
    outfile.open(output_file.c_str(), std::ios::out);
    uint8_t v;
    for (int i = 0; i < cf.n3; i += 8) {
      for (int j = 0; j < 8; j++) {
        if (out[i + j]) {
          v |= (1 << j);
        }
      }
      std::cout << "v = " << unsigned(v) << std::endl;
      outfile.write((char *) &v, 1);
      v = 0;
    }
    outfile.close();
  }
  
  delete mpc;
}


int main(int argc, char **argv) {
  if (argc != 6) {
    printf("USAGE: %s <party_id> <port_base> <circuit_file_folder> <input_folder> <output_folder>\n", argv[0]);
    return 1;
  }

  // Define some constants for the circuit setup.
  const static int party = atoi(argv[1]);
  
  // Fail early if the party ID exceeds number of parties
  if (party > num_parties) {
    return 0;
  }

  const static int port  = atoi(argv[2]);
  std::string circuit_file_folder(argv[3]);

  std::string circuit_file(circuit_file_folder);
  circuit_file.append("/agmpc.txt");
  std::string input_format_file(circuit_file_folder);
  input_format_file.append("/agmpc.txt.input");
  std::string output_format_file(circuit_file_folder);
  output_format_file.append("/agmpc.txt.output");

  std::string input_folder(argv[4]);
  std::string input_file(input_folder);
  input_file.append("/agmpc.input");

  std::string output_folder(argv[5]);
  std::string output_file(output_folder);
  output_file.append("/agmpc.output");

  NetIOMP<num_parties> io(party, port);
#ifdef LOCALHOST
  NetIOMP<num_parties> io2(party, port+2*(num_parties+1)*(num_parties+1)+1);
#else
  NetIOMP<num_parties> io2(party, port+2*(num_parties+1));
#endif
  NetIOMP<num_parties> *ios[2] = {&io, &io2};
  ThreadPool pool(2*(num_parties-1)+2);  

  // Parse input format
  std::vector<std::pair<uint64_t, uint64_t> > input_wires;
  std::pair<uint64_t, uint64_t> num_wires = parse_circuit_input(party, input_format_file, input_wires);
  std::cout << "Num inputs: " << num_wires.first << std::endl;
  std::cout << "Num output: " << num_wires.second << std::endl;

  for (size_t i = 0; i < input_wires.size(); i++) {
    std::cout << "Party "<< party << " has inputs " << input_wires[i].first << " - " << input_wires[i].second << std::endl;
  }
  
  bool *input = new bool[num_wires.first];
  bool *output = new bool[num_wires.second];
  memset(input, false, num_wires.first);
  memset(output, false, num_wires.second);
  
  // Parse input data
  create_circuit_input(input, input_wires, input_file);

  for (size_t i = 0; i < num_wires.first; i++) {
    cout << "input[" << i << "] = " << input[i] << std::endl;
  }

  // Benchmark and write out the result to output_file
  bench_once(ios, &pool, circuit_file, output_file, party, input, output);

  delete input;
  delete output;

  return 0;
}
