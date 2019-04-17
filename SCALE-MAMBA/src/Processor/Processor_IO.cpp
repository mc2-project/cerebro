/*
Copyright (c) 2017, The University of Bristol, Senate House, Tyndall Avenue, Bristol, BS8 1TH, United Kingdom.
Copyright (c) 2018, COSIC-KU Leuven, Kasteelpark Arenberg 10, bus 2452, B-3001 Leuven-Heverlee, Belgium.

All rights reserved
*/

#include "Processor_IO.h"
#include "Processor.h"

#include <unistd.h>

extern vector<sacrificed_data> SacrificeD;

void Processor_IO::private_input(unsigned int player, int target, unsigned int channel,
                                 Processor &Proc, Player &P, Machine &machine,
                                 offline_control_data &OCD)
{
  (void)(OCD);
  Proc.increment_counters(Share::SD.M.shares_per_player(P.whoami()));

  if(player == P.whoami()){
	  Proc.get_Sp_ref(target).assign(machine.get_IO().private_input_gfp(channel), P.get_mac_keys());
  }else{
	  Proc.get_Sp_ref(target).assign(0, P.get_mac_keys());
  }
}

void Processor_IO::private_output(unsigned int player, int source, unsigned int channel,
                                  Processor &Proc, Player &P,
                                  Machine &machine,
                                  offline_control_data &OCD)
{
  (void)(OCD);
  vector<Share> shares(1);
  vector<gfp> values(1);

  shares[0].assign(Proc.get_Sp_ref(source));

  // should use the open_to_one_begin/open_to_one_end

  // Via Channel one to avoid conflicts with START/STOP Opens
  Proc.OP.Open_To_One_Begin(player, shares, P);
  
  if(player == P.whoami()){
     Proc.OP.Open_To_One_End(values, shares, P); 
     machine.get_IO().private_output_gfp(values[0], channel);
  }
}
