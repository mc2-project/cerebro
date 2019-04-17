/*
Copyright (c) 2017, The University of Bristol, Senate House, Tyndall Avenue, Bristol, BS8 1TH, United Kingdom.
Copyright (c) 2018, COSIC-KU Leuven, Kasteelpark Arenberg 10, bus 2452, B-3001 Leuven-Heverlee, Belgium.

All rights reserved
*/

#include <unistd.h>
#include <chrono>

#include "Online.h"
#include "Processor/Processor.h"


void online_phase(int online_num, Player &P, offline_control_data &OCD,
                  Machine &machine)
{
  printf("Doing online for player %d in online thread %d\n", P.whoami(),
         online_num);
  fflush(stdout);



  auto online_start_time = std::chrono::high_resolution_clock::now();

  printf("Starting online phase\n");

  // Initialise the program
  Processor Proc(online_num, P.nplayers());

  bool flag= true;

  // synchronize
  fprintf(stderr, "Signal online thread ready %d\n", online_num);
  machine.Signal_Ready(online_num, true);

  while (flag)
    { // Pause online thread until it has a program to run
      int program= machine.Pause_While_Nothing_To_Do(online_num);

      if (program == -1)
        {
          flag= false;
          fprintf(stderr, "\tThread %d terminating\n", online_num);
        }
      else
        { // Execute the program
          Proc.execute(machine.progs[program], machine.get_OTI_arg(online_num), P,
                       machine, OCD);

          machine.Signal_Finished_Tape(online_num);
        }
    }

  machine.Lock_Until_Ready(online_num);

  printf("Exiting online phase : %d\n", online_num);

  auto online_end_time = std::chrono::high_resolution_clock::now();
  double online_total_time = std::chrono::duration_cast<std::chrono::microseconds>(online_end_time - online_start_time).count();
  printf("Total online time = %lf microseconds => %lf ms\n", online_total_time, online_total_time / 1000.00);
}

