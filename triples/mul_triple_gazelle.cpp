/*
FV-SHE-Benchmarking: This code benchmarks add, sub, neg and mult-plain for FV

List of Authors:
Chiraag Juvekar, chiraag@mit.edu

License Information:
MIT License
Copyright (c) 2017, Massachusetts Institute of Technology (MIT)

*/




/* COMMAND USED TO COPY SHIT:
 cat ../mul_triple.cpp | ssh -i "ryandeng.pem" ubuntu@ec2-52-37-109-162.us-west-2.compute.amazonaws.com 'cat -> /home/ubuntu/gazelle_mpc/src/demo/ahe/mul_triple.cpp'
 */

#include <pke/gazelle.h>
#include <iostream>
#include <cassert>
#include <random>

#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Common/Timer.h>
#include <cryptoTools/Common/Log.h>

#include <cryptoTools/Network/Channel.h>
#include <cryptoTools/Network/Session.h>
#include <cryptoTools/Network/IOService.h>
#include <pke/gazelle.h>
#include <utils/backend.h>

#include "math/bit_twiddle.h"


using namespace std;
using namespace lbcrypto;
using namespace osuCrypto;


ui32 vec_size = 2048, window_size = 9;
ui32 num_rep = 100;
ui32 modulus_ = 256;



// Client is P0 in the ABY paper.
void ahe_client() {
    std::cout << "Client" << std::endl;
    

    DiscreteGaussianGenerator dgg = DiscreteGaussianGenerator(4.0);
    FVParams test_params {
        true,
        opt::q, opt::p, opt::logn, opt::phim,
        (opt::q/opt::p),
        OPTIMIZED, std::make_shared<DiscreteGaussianGenerator>(dgg),
        window_size
    };

    // get up the networking
    std::string addr = "54.149.186.160";
    IOService ios(0);
    Session sess(ios, addr, 1212, EpMode::Client);
    Channel chl = sess.addChannel();


    Timer time;
    chl.resetStats();
    time.setTimePoint("start");
    // KeyGen
    auto kp = KeyGen(test_params);
    // Send public key.
    printf("Public key a and b: %lu %lu \n", kp.pk.a[0], kp.pk.b[0]);

    chl.send(kp.pk.a);
    chl.send(kp.pk.b);
        
    // Ciphertexts to be sent from P0 to P1 which is just the encrypted ciphertexts.
    Ciphertext ct_pk_a(opt::phim);
    Ciphertext ct_pk_b(opt::phim);

    for(ui32 rep=0; rep<num_rep; rep++) {
        // Generate random values for client as shares.
        uv64 vec_c_a = get_dug_vector(vec_size, modulus_);
        uv64 vec_c_b = get_dug_vector(vec_size, modulus_);

        uv64 pt_a = packed_encode(vec_c_a, opt::p, opt::logn);
        ct_pk_a = Encrypt(kp.pk, pt_a, test_params);
        uv64 pt_b = packed_encode(vec_c_b, opt::p, opt::logn);
        ct_pk_b = Encrypt(kp.pk, pt_b, test_params);
        
        // Send ciphertexts to server.
        chl.send(ct_pk_a.a);
        chl.send(ct_pk_a.b);
        chl.send(ct_pk_b.a);
        chl.send(ct_pk_b.b);


        //Decrypt ciphertext
        uv64 inter_a(opt::phim);
        inter_a = Decrypt(kp.sk, ct_pk_a, test_params);
        uv64 inter_b(opt::phim);
        inter_b = Decrypt(kp.sk, ct_pk_b, test_params);

        auto decoded_a = packed_decode(inter_a, opt::p, opt::logn);
        auto decoded_b = packed_decode(inter_b, opt::p, opt::logn);

        // Receive ciphertext from server.
        Ciphertext ct_c_f(opt::phim);
        chl.recv(ct_c_f.a);
        chl.recv(ct_c_f.b);
        // Decode ciphertext
        auto vec_c_f = postprocess_client_share(kp.sk, ct_c_f, vec_size, test_params);


        uv64 client_c_shares(vec_size);
        for (ui32 i = 0; i < vec_size; i++) {
            client_c_shares[i] = (vec_c_a[i] * vec_c_b[i] + vec_c_f[i]) % modulus_;
        }
        // Compute share for c_0, work with 16 bit numbers for now.

        printf("Client Shares: %lu %lu %lu \n", vec_c_a[0], vec_c_b[0], client_c_shares[0]);
    }

    std::cout
        << "      Sent: " << chl.getTotalDataSent() << std::endl
        << "  received: " << chl.getTotalDataRecv() << std::endl << std::endl;

    chl.resetStats();

    time.setTimePoint("online");

    std::cout << time << std::endl;
    
    // Cleanup
    chl.close();
    sess.stop();
    ios.stop();
    return;
}

// Client is P1 in the ABY paper.
void ahe_server() {
    std::cout << "Server" << std::endl;
    
    DiscreteGaussianGenerator dgg = DiscreteGaussianGenerator(4.0);
    FVParams test_params {
        true,
        opt::q, opt::p, opt::logn, opt::phim,
        (opt::q/opt::p),
        OPTIMIZED, std::make_shared<DiscreteGaussianGenerator>(dgg),
        window_size
    };

    // get up the networking
    std::string addr = "172.31.19.227";
    IOService ios(0);
    Session sess(ios, addr, 1212, EpMode::Server);
    Channel chl = sess.addChannel();

    Timer time;
    


    // Ciphertext to be sent to client
    Ciphertext ct_pk(opt::phim);
    // Ciphertext that are received from the client
    Ciphertext ct_recv_a(opt::phim);
    Ciphertext ct_recv_b(opt::phim);
    Ciphertext ct_pk_a(opt::phim);
    Ciphertext ct_pk_b(opt::phim);


    

    PublicKey pk(opt::phim);
    SecretKey sk(opt::phim);
    chl.recv(pk.a);
    chl.recv(pk.b);
    time.setTimePoint("start");
    //chl.recv(sk.s);
    

    for(ui32 rep=0; rep<num_rep; rep++) {
        chl.recv(ct_recv_a.a);
        chl.recv(ct_recv_a.b);
        chl.recv(ct_recv_b.a);
        chl.recv(ct_recv_b.b);
        
        uv64 vec_s_a = get_dug_vector(vec_size, modulus_);
        uv64 vec_s_b = get_dug_vector(vec_size, modulus_);
        uv64 vec_ran = get_dug_vector(vec_size, modulus_);

        

        // Compute c shares in server
        uv64 vec(vec_size);
        for (ui32 n = 0; n < vec_size; n++) {
            vec[n] = (vec_s_a[n] * vec_s_b[n] - vec_ran[n]) % modulus_;
        }

        uv64 pt_a = packed_encode(vec_s_a, opt::p, opt::logn);
        uv64 pt_b = packed_encode(vec_s_b, opt::p, opt::logn);
        auto ct2_null_a = NullEncrypt(pt_a, test_params);
        auto ct2_null_b = NullEncrypt(pt_b, test_params);


        // Compute d to sent back to P0.
        auto ct_sum = EvalAdd(EvalMultPlain(ct_recv_a, ct2_null_b, test_params), EvalMultPlain(ct_recv_b, ct2_null_a, test_params), test_params);
        uv64 pt_ran = packed_encode(vec_ran, opt::p, opt::logn);
        auto ct_pk_ran = Encrypt(pk, pt_ran, test_params);
        ct_pk = EvalAdd(ct_sum, ct_pk_ran, test_params);       
        
        chl.send(ct_pk.a);
        chl.send(ct_pk.b);
        printf("Server Shares: %lu %lu %lu \n", vec_s_a[0], vec_s_b[0], vec[0]);
    }


    time.setTimePoint("online");

    std::cout << time << std::endl;
    // std::cout << input_bits << std::endl;
    // std::cout << extractedMap << std::endl;

    chl.close();
    sess.stop();
    ios.stop();
    return;
}


int main(int argc, char** argv) {
    // std::cin >> vec_size >> window_size;

    ftt_precompute(opt::z, opt::q, opt::logn);
    ftt_precompute(opt::z_p, opt::p, opt::logn);
    encoding_precompute(opt::p, opt::logn);
    //precompute_automorph_index(opt::phim);


    if (argc == 1)
    {
        std::vector<std::thread> thrds(2);
        thrds[0] = std::thread([]() { ahe_server(); });
        thrds[1] = std::thread([]() { ahe_client(); });

        for (auto& thrd : thrds)
            thrd.join();
    }
    else if(argc == 2)
    {
        int role = atoi(argv[1]); // 0: send, 1: recv
        role ? ahe_server() : ahe_client();
    }
    else
    {
        std::cout << "this program takes a runtime argument.\n\n"
            << "to run the AES GC, run\n\n"
            << "    gc-online [0|1]\n\n"
            << "the optional {0,1} argument specifies in which case the program will\n"
            << "run between two terminals, where each one was set to the opposite value. e.g.\n\n"
            << "    gc-online 0\n\n"
            << "    gc-online 1\n\n"
            << "These programs are fully networked and try to connect at localhost:1212.\n"
            << std::endl;


    }
}


