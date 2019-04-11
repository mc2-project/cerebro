#include <stdint.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <emp-tool/emp-tool.h>
#include "emp-agmpc/emp-agmpc.h"


const static int num_parties = 3;
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

    input_wires.push_back(std::make_pair(begin, end));
  }

  return std::make_pair(num_inputs, num_outputs);
}


void bench_once(NetIOMP<num_parties> * ios[2], ThreadPool * pool, string filename, int party, bool *in, bool *out) {
  if(party == 1)cout <<"CIRCUIT:\t"<<filename<<endl;
  CircuitFile cf(filename.c_str());

  auto start = clock_start();
  CMPC<num_parties>* mpc = new CMPC<num_parties>(ios, pool, party, &cf);
  ios[0]->flush();
  ios[1]->flush();
  double t2 = time_from(start);
//  ios[0]->sync();
//  ios[1]->sync();
  if(party == 1)cout <<"Setup:\t"<<party<<"\t"<< t2 <<"\n"<<flush;

  start = clock_start();
  mpc->function_independent();
  ios[0]->flush();
  ios[1]->flush();
  t2 = time_from(start);
  if(party == 1)cout <<"FUNC_IND:\t"<<party<<"\t"<<t2<<" \n"<<flush;

  start = clock_start();
  mpc->function_dependent();
  ios[0]->flush();
  ios[1]->flush();
  t2 = time_from(start);
  if(party == 1)cout <<"FUNC_DEP:\t"<<party<<"\t"<<t2<<" \n"<<flush;

  //bool *in = new bool[cf.n1+cf.n2]; bool *out = new bool[cf.n3];
  memset(in, false, cf.n1+cf.n2);
  start = clock_start();
  mpc->online(in, out);
  ios[0]->flush();
  ios[1]->flush();
  t2 = time_from(start);
  //uint64_t band2 = ios->count();
  //if(party == 1)cout <<"bandwidth\t"<<party<<"\t"<<band2<<endl;
  if(party == 1)cout <<"ONLINE:\t"<<party<<"\t"<<t2<<" \n"<<flush;
  if(party == 1) {
    string res = "";
    for(int i = 0; i < cf.n3; ++i)
      res += (out[i]?"1":"0");
    cout << "Result: " << res << endl;
    for (int i = 0; i < cf.n3; i += 64) {
      cout << "Result individual: " << bool_to64(out + i * 64) << endl;
    }
  }
  delete mpc;
}

// This function reads the input from "filename" and sets the data in "input"
// Each party sets a value for every input wire
// If a party does not provide real input for a wire, the value is set to be 0
void create_circuit_input(bool *input, std::vector<std::pair<uint64_t, uint64_t> > &input_wires, std::string filename) {
  std::ifstream infile(filename.c_str());
  for (size_t i = 0; i < input_wires.size(); i++) {
    uint64_t offset = input_wires.at(i).first;
    uint64_t length = input_wires.at(i).second - offset;
    infile.read(reinterpret_cast<char *>(input + offset), length / 8);
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    printf("USAGE: %s <party_id> <port_base> <circuit_file> <input_format_file> <input_file>\n", argv[0]);
    return 1;
  }

  // Define some constants for the circuit setup.
  const static int party = atoi(argv[1]);
  const static int port  = atoi(argv[2]);
  std::string circuit_file(argv[3]);
  std::string input_format_file(argv[4]);
  std::string input_file(argv[5]);


  if (party > num_parties) {
    return 0;
  }

  NetIOMP<num_parties> io(party, port);
#ifdef LOCALHOST
  NetIOMP<num_parties> io2(party, port+2*(num_parties+1)*(num_parties+1)+1);
#else
  NetIOMP<num_parties> io2(party, port+2*(num_parties+1));
#endif
  NetIOMP<num_parties> *ios[2] = {&io, &io2};
  ThreadPool pool(2*(num_parties-1)+2);  

  // Input circuit file
  CircuitFile cf(circuit_file.c_str());

  // Parse input
  std::vector<std::pair<uint64_t, uint64_t> > input_wires;
  std::pair<uint64_t, uint64_t> num_wires = parse_circuit_input(party, input_format_file, input_wires);
  bool *input = new bool[num_wires.first];
  memset(input, false, num_wires.first);
  create_circuit_input(input, input_wires, input_file);
  bool *output = new bool[num_wires.second];

  bench_once(ios, &pool, circuit_file, party, input, output);


  /*
  NetIOMP<num_parties> *io_list[num_parties];
  for (uint64_t i = 0; i < num_parties; i++) {
    io_list[i] = new NetIOMP<num_parties>(party_id, port_base + i);
  }

  ThreadPool pool(2 * num_parties);
  CMPC<num_parties>* mpc = new CMPC<num_parties>(io_list, &pool, party_id, &cf);
  */


  /*
  // Run the secure computation
  mpc->function_independent();
  mpc->function_dependent();

  int start[] = {0};
  int end[] = {(int) num_wires.first};
  mpc->online(input, output, start, end);
  */

  /*
  // Clean up memory
  for (uint64_t i = 0; i < num_parties; i++) {
    delete io_list[i];
  }


  delete input;
  delete output;
  delete mpc;
  */

  return 0;
}
