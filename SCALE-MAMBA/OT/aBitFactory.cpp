/*
Copyright (c) 2017, The University of Bristol, Senate House, Tyndall Avenue, Bristol, BS8 1TH, United Kingdom.
Copyright (c) 2018, COSIC-KU Leuven, Kasteelpark Arenberg 10, bus 2452, B-3001 Leuven-Heverlee, Belgium.

All rights reserved
*/

#include "aBitFactory.h"

#include "COT.h"
#include "LSSS/PRSS.h"
#include "Tools/Crypto.h"
#include "Tools/Timer.h"
#include "Tools/parse.h"
#include "config.h"

#define max_l 131072

void aBitFactory::Initialize(Player &P)
{
  gf2n::init_field(OT_comp_sec);

  unsigned int n= P.nplayers(), whoami= P.whoami();

  G.ReSeed(20000);
  /* Create my base choice bits */
  vector<int> choicebits(OT_comp_sec);
  for (unsigned int i= 0; i < OT_comp_sec; i++)
    {
      choicebits[i]= G.get_uchar() & 1;
    }

  CryptoPP::RandomPool RNG;
  COTS.resize(n);
  COTR.resize(n);

  /* We do first the Random OTs between i and whoami where
   * the sender is the value which is smallest
   */
  string input, output;
  for (unsigned int i= 0; i < n; i++)
    {
      if (i < whoami)
        { // In this case i is the sender
          COTR[i].init(P, i, RNG, choicebits);
        }
      else if (i > whoami)
        { // In this case whoami is the sender
          COTS[i].init(P, i, RNG);
        }
    }

  /* Now do it the other way around */
  for (unsigned int i= 0; i < n; i++)
    {
      if (i > whoami)
        { // In this case i is the receiver
          COTR[i].init(P, i, RNG, choicebits);
        }
      else if (i < whoami)
        { // In this case whoami is the receiver
          COTS[i].init(P, i, RNG);
        }
    }

  int i0= 0;
  if (whoami == 0)
    {
      i0= 1;
    }
  Delta= COTR[i0].get_Delta();
  aBit::set_nplayers(n, whoami, Delta);

  // This is the max size here
  aBits.resize(8 * max_l + OT_comp_sec);
  xvec.resize(8 * max_l + OT_comp_sec);

  default_l= 8;
  used= 0;
  status= EMPTY;

  printf("Finished Base-OTs\n");
}

/* Algorithm 16 Steps 1-4 of ePrint 2017/214 */
unsigned int aBitFactory::make_aBits(unsigned int l, Player &P)
{
  if (status != EMPTY)
    {
      throw OT_error();
    }
  unsigned int m= 8 * l, ldash= l + OT_comp_sec / 8;

  unsigned int n= P.nplayers(), whoami= P.whoami();

  /* Step 1 */
  sz_aBits= m + OT_comp_sec;
  xvec.randomize_at(0, ldash, G);
  for (unsigned int i= 0; i < sz_aBits; i++)
    {
      aBits[i].set_value(xvec.get_bit(i));
    }

  /* Step 2 */
  vector<vector<gf2n>> M(n);
  vector<vector<gf2n>> K(n);

  for (unsigned int i= 0; i < n; i++)
    {
      if (i < whoami)
        {
          COTS[i].next_checked_iteration(P, sz_aBits, aBits, xvec);
        }
      else if (i > whoami)
        {
          COTR[i].next_checked_iteration(P, sz_aBits, aBits);
        }
    }
  for (unsigned int i= 0; i < n; i++)
    {
      if (i > whoami)
        {
          COTS[i].next_checked_iteration(P, sz_aBits, aBits, xvec);
        }
      else if (i < whoami)
        {
          COTR[i].next_checked_iteration(P, sz_aBits, aBits);
        }
    }

  /* Step 4 */

  /* Step 4a and 4b */
  uint8_t seed[SEED_SIZE];
  AgreeRandom(P, seed, SEED_SIZE);
  PRNG G2;
  G2.SetSeed(seed);
  gf2n chi;
  aBit C, T;
  for (unsigned int i= 0; i < m; i++)
    {
      chi.randomize(G2);
      T.mult_by(chi, aBits[i]);
      C.add(T);
    }
  for (unsigned int h= 0; h < OT_comp_sec; h++)
    {
      chi.assign_zero();
      chi.set_bit(h, 1);
      T.mult_by(chi, aBits[m + h]);
      C.add(T);
    }

  /* Step 4d */
  vector<gf2n> CC(n);
  CC[0]= C.get_raw_value();
  for (unsigned int i= 1; i < n; i++)
    {
      CC[i].randomize(G);
      CC[0].sub(CC[i]);
    }
  // Now send CC[i] to player i and get the corresponding
  // value back
  vector<string> o(n);
  for (unsigned int i= 0; i < n; i++)
    {
      if (i != P.whoami())
        {
          stringstream ss;
          CC[i].output(ss);
          o[i]= ss.str();
        }
    }
  P.Send_Distinct_And_Receive(o);
  for (unsigned int i= 0; i < n; i++)
    {
      if (i != P.whoami())
        {
          istringstream ss(o[i]);
          CC[i].input(ss);
          CC[P.whoami()].add(CC[i]);
        }
    }
  // CC[P.whoami()] now holds the new local share of the value in C

  /* Step 4e */
  stringstream ss;
  CC[P.whoami()].output(ss);
  o[P.whoami()]= ss.str();
  P.Broadcast_Receive(o, true);
  for (unsigned int i= 0; i < n; i++)
    {
      if (i != P.whoami())
        {
          istringstream ss(o[i]);
          CC[i].input(ss);
          CC[P.whoami()].add(CC[i]);
        }
    }
  // CC[P.whoami()] now holds the reconstructed value of the shared value C

  /* Step 4f */
  // Compute Z_i^i
  gf2n Zi;
  Zi.add(CC[P.whoami()], C.get_raw_value());
  Zi.mul(Delta);
  for (unsigned int i= 0; i < n; i++)
    {
      if (i != P.whoami())
        {
          Zi.add(C.get_Key(i));
        }
    }
  vector<vector<gf2n>> data(n + 2, vector<gf2n>(n));
  data[0][P.whoami()]= C.get_raw_value();
  for (unsigned int i= 0; i < n; i++)
    {
      if (i != P.whoami())
        {
          data[i + 1][P.whoami()]= C.get_MAC(i);
        }
      else
        {
          data[i + 1][P.whoami()]= Zi;
        }
    }
  Commit_And_Open(data, P, true);

  /* Step 4g */
  // Check sum z_j^i
  gf2n zz;
  zz.assign_zero();
  for (unsigned int j= 0; j < n; j++)
    {
      zz.add(data[P.whoami() + 1][j]);
    }
  if (!zz.is_zero())
    {
      throw OT_error();
    }

  // Check the equation
  for (unsigned int j= 0; j < n; j++)
    {
      if (j != P.whoami())
        {
          zz.mul(data[0][j], Delta);
          zz.add(C.get_Key(j));
          zz.add(data[P.whoami() + 1][j]);
          if (!zz.is_zero())
            {
              throw OT_error();
            }
        }
    }

  // Do check of broadcasts
  P.Check_Broadcast();

  status= aShares;
  // These are the ones we want to save
  sz_aBits= m;
  used= 0;
  return m;
}

void aBitFactory::tune(Player &P, int verbose)
{
  unsigned int l= 16384; // Pick a large value here as sometimes have noisy results
  Timer T;
  if (verbose > 0)
    {
      cout << "Tuning aBitFactory\n";
    }
  double mx= 0;
  while (l <= max_l)
    {
      unsigned int sz= 0;
      T.reset();
      T.start();
      while (sz < 10 * l)
        {
          sz+= make_aBits(l, P);
          status= EMPTY;
        }
      T.stop();
      double e= sz / T.elapsed();
      if (verbose > 0)
        {
          cout << l << "\t" << e << "  " << sz << endl;
        }
      if (e > mx)
        {
          mx= e;
          default_l= l;
        }
      l= 2 * l;
    }
  /* Player 0 now needs to tell the other players what number to take */
  if (P.whoami() == 0)
    {
      stringstream ss;
      ss << default_l;
      P.send_all(ss.str());
    }
  else
    {
      string ss;
      P.receive_from_player(0, ss);
      istringstream is(ss);
      is >> default_l;
    }
  if (verbose > 0)
    {
      cout << "New default_l = " << default_l << endl;
    }
}

aBit aBitFactory::get_aShare(Player &P)
{
  if (used == sz_aBits)
    {
      status= EMPTY;
    }
  if (status == EMPTY)
    {
      make_aBits(P);
    }

  used++;
  return aBits[used - 1];
}
