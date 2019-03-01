/* Copyright (C) 2012-2017 IBM Corp.
 * This program is Licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */

/* Test_General.cpp - A general test program that uses a mix of operations over four ciphertexts.
 */
#include <NTL/ZZ.h>
#include <NTL/BasicThreadPool.h>
#include "FHE.h"
#include "timing.h"
#include "EncryptedArray.h"
#include <NTL/lzz_pXFactoring.h>
#include <random>
#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <iostream>
#include <fstream>

#include <cassert>
#include <cstdio>
#include <ctime>

NTL_CLIENT

//#define DEBUG_PRINTOUT

#ifdef DEBUG_PRINTOUT
#define debugCompare(ea,sk,p,c) {\
  PlaintextArray pp(ea);\
  ea.decrypt(c, sk, pp);\
  if (!equals(ea, pp, p)) { \
    std::cout << "oops:\n"; std::cout << p << "\n"; \
    std::cout << pp << "\n"; \
    exit(0); \
  }}
#else
#define debugCompare(ea,sk,p,c)
#endif

/**************

1. c1.multiplyBy(c0)
2. c0 += random constant
3. c2 *= random constant
4. tmp = c1
5. ea.shift(tmp, random amount in [-nSlots/2, nSlots/2])
6. c2 += tmp
7. ea.rotate(c2, random amount in [1-nSlots, nSlots-1])
8. c1.negate()
9. c3.multiplyBy(c2) 
10. c0 -= c3

**************/

int PORT_NUMBER = 8888;

static bool noPrint = true;

void  TestIt(long R, long p, long r, long d, long c, long k, long w, 
               long L, long m, const Vec<long>& gens, const Vec<long>& ords)
{
  double start, finish;

  NTL::SetSeed(conv<ZZ>((long) time(0)));
  // Have all ZZ_p's be modulo p.
  //ZZ_p::init(p); 
  char buffer[32];
  if (!noPrint) {
    std::cout << "\n\n******** TestIt" << (isDryRun()? "(dry run):" : ":");
    std::cout << " R=" << R 
	      << ", p=" << p
	      << ", r=" << r
	      << ", d=" << d
	      << ", c=" << c
	      << ", k=" << k
	      << ", w=" << w
	      << ", L=" << L
	      << ", m=" << m
	      << ", gens=" << gens
	      << ", ords=" << ords
	      << endl;
  }
  // What is the below gens1, ords1 thing for?
  vector<long> gens1, ords1;
  convert(gens1, gens);
  convert(ords1, ords);
  
  FHEcontext context(m, p, r, gens1, ords1);
  buildModChain(context, L, c);
  
  int size = context.zMStar.getNSlots();
  cout << "Plaintext slot size: " << size << endl;

  ZZX G;
  if (d == 0)
    G = context.alMod.getFactorsOverZZ()[0];
  else
    G = makeIrredPoly(p, d); 

  if (!noPrint) {
    context.zMStar.printout();
    std::cout << endl;

    std::cout << "security=" << context.securityLevel()<<endl;
    std::cout << "# small primes = " << context.smallPrimes.card() << "\n";
    std::cout << "# ctxt primes = " << context.ctxtPrimes.card() << "\n";
    std::cout << "# bits in ctxt primes = " 
	 << long(context.logOfProduct(context.ctxtPrimes)/log(2.0) + 0.5) << "\n";
    std::cout << "# special primes = " << context.specialPrimes.card() << "\n";
    std::cout << "# bits in special primes = " 
	 << long(context.logOfProduct(context.specialPrimes)/log(2.0) + 0.5) << "\n";
    std::cout << "G = " << G << "\n";
  }



  //FHESecKey secretKey(context);
  //const FHEPubKey& publicKey = secretKey;
  //secretKey.GenSecKey(); // A +-1/0 secret key
  //addSome1DMatrices(secretKey); // compute key-switching matrices that we need

  FHESecKey sk(context);
  sk.GenSecKey(64);
  std::ofstream of("pk.pk", std::ios::binary);
  FHEPubKey pk(sk);
  // pk.makeSymmetric();
  of << pk;
  of.close();

  

  using boost::asio::ip::tcp;
  tcp::iostream connect("127.0.0.1", "8888");
  if (!connect) {
          std::cerr << "Can not connect:" << 
               connect.error().message() << std::endl;
  return;
  }


  writeContextBase(connect, context);
  connect << context;

  // Start gen triple
  Ctxt c0(pk), c1(pk);
  //FHE_NTIMER_START(Circuit);
  start = clock();
  for (int i = 0; i < 100; i++) {

    /*
    long ptxt1;
    long ptxt2;
    ptxt1 = RandomBnd(p);
    ptxt2 = RandomBnd(p);
    pk.Encrypt(c0, to_ZZX(ptxt1));
    pk.Encrypt(c1, to_ZZX(ptxt2));
    */
    vector<long> ptxt1(size), ptxt2(size);
    for (int i = 0; i < size; i++) {
      ptxt1[i] = RandomBnd(p);
      ptxt2[i] = RandomBnd(p);
    }
    // Below is some stuff in Test_General.cpp
    /*
    PlaintextArray ptxt1(ea);
    PlaintextArray ptxt2(ea);
    random(ea, ptxt1);
    random(ea, ptxt2);
    ea.encrypt(c0, pk, ptxt1);
    ea.encrypt(c1, pk, ptxt2);
    */

    EncryptedArray ea(context, context.alMod);
    //cout << "Encrypted array size: " << ea.size() << endl;
    ZZX ptxt1_encoded, ptxt2_encoded;

    // encode plaintext vectors
    double encode_start = clock();
    ea.encode(ptxt1_encoded, ptxt1);
    ea.encode(ptxt2_encoded, ptxt2);
    double encode_finish = clock();
    cout << "Encode array takes " << (encode_finish - encode_start) / (2 * CLOCKS_PER_SEC) << endl;

    double encrypt_start = clock();
    pk.Encrypt(c0, ptxt1_encoded);
    pk.Encrypt(c1, ptxt2_encoded);
    double encrypt_finish = clock();
    cout << "Encrypt array takes " << (encrypt_finish - encrypt_start) / (CLOCKS_PER_SEC) << endl;

    
    connect << c0;
    connect << c1;
    connect.flush();




    // Receive d from Server
    Ctxt ciph_d(sk);
    connect >> ciph_d;
    ZZX dec_d;

    double decrypt_start = clock();
    sk.Decrypt(dec_d, ciph_d);
    double decrypt_finish = clock();
    cout << "Decrypt takes " << (decrypt_finish - decrypt_start) / (CLOCKS_PER_SEC) << endl;



    vector<long> res;
    double decode_start = clock();
    ea.decode(res, dec_d);
    double decode_finish = clock();
    cout << "Decode takes " << (decode_finish - decode_start) / (CLOCKS_PER_SEC) << endl;

    //connect.close();

    long d = res[0];
    long share_c = (ptxt1[0] * ptxt2[0] + d) % p;
    cout << "Share A: " << ptxt1[0] << " Share B: " << ptxt2[0] << " Share C: " << share_c << endl;

  }
  //FHE_NTIMER_STOP(Circuit);
  //printAllTimers();

  connect.close();
  //resetAllTimers();
  c0.cleanUp();
  c1.cleanUp();
  finish = clock();
  cout << "Took " << (finish - start) / CLOCKS_PER_SEC << " seconds for 100 iterations" << endl;

  if (!noPrint) {
    printAllTimers();
    std::cout << endl;
  }
}


/* A general test program that uses a mix of operations over four ciphertexts.
 * Usage: Test_General_x [ name=value ]...
 *   R       number of rounds  [ default=1 ]
 *   p       plaintext base  [ default=2 ]
 *   r       lifting  [ default=1 ]
 *   d       degree of the field extension  [ default=1 ]
 *              d == 0 => factors[0] defines extension
 *   c       number of columns in the key-switching matrices  [ default=2 ]
 *   k       security parameter  [ default=80 ]
 *   L       # of bits in the modulus chain  [ default=heuristic ]
 *   s       minimum number of slots  [ default=0 ]
 *   repeat  number of times to repeat the test  [ default=1 ]
 *   m       use specified value as modulus
 *   mvec    use product of the integers as  modulus
 *              e.g., mvec='[5 3 187]' (this overwrite the m argument)
 *   gens    use specified vector of generators
 *              e.g., gens='[562 1871 751]'
 *   ords    use specified vector of orders
 *              e.g., ords='[4 2 -4]', negative means 'bad'
 */

void run_server() {
  double start, finish;
  using boost::asio::ip::tcp;
  boost::asio::io_service ios;
  tcp::endpoint endpoint(tcp::v4(), 8888);
  tcp::acceptor acceptor(ios, endpoint);
  for (;;) {
    tcp::iostream client;
    boost::system::error_code err;
    acceptor.accept(*client.rdbuf(), err);
    if (!err) {
      std::cout << "connected" << std::endl;
      unsigned long m, p, r;
      std::vector<long> gens, ords;
      readContextBase(client, m, p, r, gens, ords);
      FHEcontext context(m, p, r, gens, ords);
      client >> context;
      std::ifstream ifs("pk.pk", std::ios::binary);
      FHEPubKey pk(context);
      ifs >> pk;
      ifs.close();

      int size = context.zMStar.getNSlots();
      cout << "Plaintext slot size: " << size << endl;
      Ctxt c0(pk), c1(pk), ran_num(pk), ctx1(pk), ctx2(pk);
      //FHE_NTIMER_START(Circuit);
      start = clock();
      for (int i = 0; i < 100; i++) {
        // Receive ciphertext of shares from client

        client >> ctx1; 
        client >> ctx2;
        /*
        long ptxt1;
        long ptxt2;
        long ran_num;

        ptxt1 = RandomBnd(p);
        ptxt2 = RandomBnd(p);
        ran_num = RandomBnd(p);

        Ctxt ran(pk);
        pk.Encrypt(ran, to_ZZX(ran_num));
        */

        vector<long> ptxt1(size), ptxt2(size), ran(size);
        for (int i = 0; i < size; i++) {
          ptxt1[i] = RandomBnd(p);
          ptxt2[i] = RandomBnd(p);
          ran[i] = RandomBnd(p);
        }

        EncryptedArray ea(context, context.alMod);
        ZZX ptxt1_encoded, ptxt2_encoded, ran_encoded;

        // encode plaintext vectors
        ea.encode(ptxt1_encoded, ptxt1);
        ea.encode(ptxt2_encoded, ptxt2);
        ea.encode(ran_encoded, ran);

        pk.Encrypt(c0, ptxt1_encoded);
        pk.Encrypt(c1, ptxt2_encoded);
        pk.Encrypt(ran_num, ran_encoded);

        Ctxt temp = ctx1;
        temp.multByConstant(ptxt2_encoded);

        Ctxt temp2 = ctx2;
        temp2.multByConstant(ptxt1_encoded);

        Ctxt sum = temp;
        sum += temp2;
        sum += ran_num;



        //cout << "Random value: " << ran_num << " " << endl;
        //cout << "Prime: " << p << endl;
        client << sum;
        //client.close();

        long share_c;
        share_c = (ptxt1[0] * ptxt2[0] - ran[0]) % p;
        cout << "Share A: " << ptxt1[0] << " Share B: " << ptxt2[0] << " Share C: " << share_c << endl; 
      }
      c0.cleanUp();
      c1.cleanUp();
      ran_num.cleanUp();
      ctx1.cleanUp();
      ctx2.cleanUp();
      finish = clock();
      cout << "Took " << (finish - start) / CLOCKS_PER_SEC << " seconds for 100 iterations" << endl;
      //FHE_NTIMER_STOP(Circuit);
      //printAllTimers();

      client.close();

      break;

    }


  }
}

int main(int argc, char **argv) 
{
  setTimersOn();

  ArgMapping amap;

  bool dry=false;
  amap.arg("dry", dry, "dry=1 for a dry-run");

  long R=1;
  amap.arg("R", R, "number of rounds");

  long p=2;
  amap.arg("p", p, "plaintext base");

  long r=1;
  amap.arg("r", r,  "lifting");

  long d=1;
  amap.arg("d", d, "degree of the field extension");
  amap.note("d == 0 => factors[0] defines extension");

  long c=2;
  amap.arg("c", c, "number of columns in the key-switching matrices");

  
  long k=80;
  amap.arg("k", k, "security parameter");

  long L=500;
  amap.arg("L", L, "# of bits in the modulus chain");

  long s=4096;
  amap.arg("s", s, "minimum number of slots");

  long repeat=1;
  amap.arg("repeat", repeat,  "number of times to repeat the test");

  long chosen_m=0;
  amap.arg("m", chosen_m, "use specified value as modulus", NULL);

  Vec<long> mvec;
  amap.arg("mvec", mvec, "use product of the integers as  modulus", NULL);
  amap.note("e.g., mvec='[5 3 187]' (this overwrite the m argument)");

  Vec<long> gens;
  amap.arg("gens", gens, "use specified vector of generators", NULL);
  amap.note("e.g., gens='[562 1871 751]'");

  Vec<long> ords;
  amap.arg("ords", ords, "use specified vector of orders", NULL);
  amap.note("e.g., ords='[4 2 -4]', negative means 'bad'");

  long seed=0;
  amap.arg("seed", seed, "PRG seed");

  long nt=1;
  amap.arg("nt", nt, "num threads");

  long server = 0;
  amap.arg("server", server, "0 if server, 1 if client");

  amap.arg("noPrint", noPrint, "suppress printouts");

  amap.parse(argc, argv);

  SetSeed(ZZ(seed));
  SetNumThreads(nt);
  

  long w = 64; // Hamming weight of secret key

  if (mvec.length()>0)
    chosen_m = computeProd(mvec);
  long m = FindM(k, L, c, p, d, s, chosen_m, !noPrint);
  cout << "Modulus " << m;
  setDryRun(dry);
  /*
  for (long repeat_cnt = 0; repeat_cnt < repeat; repeat_cnt++) {
    TestIt(R, p, r, d, c, k, w, L, m, gens, ords);
  }
  */
  if (server == 0) {
    cout << "Running server" << endl;
    run_server();
  } else {
    cout << "Running clienet" << endl;
    TestIt(R, p, r, d, c, k, w, L, m, gens, ords);
  }
}
