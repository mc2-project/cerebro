/*
Copyright (c) 2017, The University of Bristol, Senate House, Tyndall Avenue, Bristol, BS8 1TH, United Kingdom.
Copyright (c) 2018, COSIC-KU Leuven, Kasteelpark Arenberg 10, bus 2452, B-3001 Leuven-Heverlee, Belgium.

All rights reserved
*/

#include "File_Input_Output.h"
#include "Exceptions/Exceptions.h"

long File_Input_Output::open_channel(unsigned int channel)
{
  cout << "Opening file " << channel << endl;
  fstream *myfile = new fstream();
  std::string file_name(file_dir);
  file_name.append("/f");
  file_name.append(std::to_string(channel));
  myfile->open(file_name, ios::binary);
  open_channels.insert(std::pair<unsigned int, fstream*>(channel, myfile));
  return 0;
}

void File_Input_Output::close_channel(unsigned int channel)
{
  cout << "Closing channel " << channel << endl;
  fstream *myfile = open_channels[channel];
  delete myfile;
  auto it = open_channels.find(channel);
  open_channels.erase(it);
}

gfp File_Input_Output::private_input_gfp(unsigned int channel)
{
  fstream *myfile = open_channels[channel];
  word x;
  myfile->read((char *) &x, sizeof(x));
  gfp y;
  y.assign(x);
  return y;
}

void File_Input_Output::private_output_gfp(const gfp &output, unsigned int channel)
{
  cout << "Output channel " << channel << " : ";
  output.output(cout, true);
  cout << endl;
}

gfp File_Input_Output::public_input_gfp(unsigned int channel)
{
  cout << "Enter value on channel " << channel << " : ";
  word x;
  cin >> x;
  gfp y;
  y.assign(x);

  // Important to have this call in each version of public_input_gfp
  Update_Checker(y, channel);

  return y;
}

void File_Input_Output::public_output_gfp(const gfp &output, unsigned int channel)
{
  cout << "Output channel " << channel << " : ";
  output.output(cout, true);
  cout << endl;
}

long File_Input_Output::public_input_int(unsigned int channel)
{
  cout << "Enter value on channel " << channel << " : ";
  long x;
  cin >> x;

  // Important to have this call in each version of public_input_gfp
  Update_Checker(x, channel);

  return x;
}

void File_Input_Output::public_output_int(const long output, unsigned int channel)
{
  cout << "Output channel " << channel << " : " << output << endl;
}

void File_Input_Output::output_share(const Share &S, unsigned int channel)
{
  // (*outf) << "Output channel " << channel << " : ";
  // S.output(*outf, true);
}

Share File_Input_Output::input_share(unsigned int channel)
{
  // cout << "Enter value on channel " << channel << " : ";
  // Share S;
  // S.input(*inpf, true);
  // return S;
}

void File_Input_Output::trigger(Schedule &schedule)
{
  printf("Restart requested: Enter a number to proceed\n");
  int i;
  cin >> i;

  // Load new schedule file program streams, using the original
  // program name
  //
  // Here you could define programatically what the new
  // programs you want to run are, by directly editing the
  // public variables in the schedule object.
  unsigned int nthreads= schedule.Load_Programs();
  if (schedule.max_n_threads() < nthreads)
    {
      throw Processor_Error("Restart requires more threads, cannot do this");
    }
}

void File_Input_Output::debug_output(const stringstream &ss)
{
  printf("%s", ss.str().c_str());
  fflush(stdout);
}

void File_Input_Output::crash(unsigned int PC, unsigned int thread_num)
{
  printf("Crashing in thread %d at PC value %d\n", thread_num, PC);
  throw crash_requested();
}
