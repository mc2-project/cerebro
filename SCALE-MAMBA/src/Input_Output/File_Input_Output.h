/*
Copyright (c) 2017, The University of Bristol, Senate House, Tyndall Avenue, Bristol, BS8 1TH, United Kingdom.
Copyright (c) 2018, COSIC-KU Leuven, Kasteelpark Arenberg 10, bus 2452, B-3001 Leuven-Heverlee, Belgium.

All rights reserved
*/
#ifndef _FileInputOutput
#define _FileInputOutput

/* A simple IO class which just uses standard
 * input/output to communicate values
 *
 * Whereas share values are input/output using
 * a steam, with either human or non-human form
 */

#include "Input_Output_Base.h"

#include <fstream>
#include <iostream>
#include <string>
#include <map>
using namespace std;

class File_Input_Output : public Input_Output_Base
{
  std::map<unsigned int, fstream *> open_channels;
  std::string file_dir;

public:
  File_Input_Output()
      : Input_Output_Base()
  {
    ;
  }

  void init(std::string _file_dir)
  {
    this->file_dir = _file_dir ;
  }

  virtual long open_channel(unsigned int channel);
  virtual void close_channel(unsigned int channel);

  virtual gfp private_input_gfp(unsigned int channel);
  virtual void private_output_gfp(const gfp &output, unsigned int channel);

  virtual void public_output_gfp(const gfp &output, unsigned int channel);
  virtual gfp public_input_gfp(unsigned int channel);

  virtual void public_output_int(const long output, unsigned int channel);
  virtual long public_input_int(unsigned int channel);

  virtual void output_share(const Share &S, unsigned int channel);
  virtual Share input_share(unsigned int channel);

  virtual void trigger(Schedule &schedule);

  virtual void debug_output(const stringstream &ss);

  virtual void crash(unsigned int PC, unsigned int thread_num);
};

#endif
