#ifndef ABIT_MP_H__
#define ABIT_MP_H__
#include <emp-tool/emp-tool.h>
#include <emp-ot/emp-ot.h>
#include "netmp.h"
#include "helper.h"

template<int nP>
class ABitMP { public:
	SHOTExtension<NetIO> *abit1[nP+1];
	SHOTExtension<NetIO> *abit2[nP+1];
	NetIOMP<nP> *io;
	ThreadPool * pool;
	int party;
	PRG prg;
	block Delta;
	Hash hash;
	block * pretable;
	ABitMP(NetIOMP<nP>* io, ThreadPool * pool, int party) {
		this->io = io;
		this->pool = pool;
		this->party = party;
		bool * tmp = new bool[128];
		prg.random_bool(tmp, 128);

		for(int i = 1; i <= nP; ++i) for(int j = 1; j <= nP; ++j) if(i < j) {
			if(i == party) {
				abit1[j] = new SHOTExtension<NetIO>(io->get(j, false));
				abit2[j] = new SHOTExtension<NetIO>(io->get(j, true));
			} else if (j == party) {
				abit2[i] = new SHOTExtension<NetIO>(io->get(i, false));
				abit1[i] = new SHOTExtension<NetIO>(io->get(i, true));
			}
		}

		vector<future<void>> res;//relic multi-thread problems...
		for(int i = 1; i <= nP; ++i) for(int j = 1; j <= nP; ++j) if(i < j) {
			if(i == party) {
				res.push_back(pool->enqueue([this, io, i, j]() {
					abit1[j]->setup_send();
					io->flush(j);
				}));
				res.push_back(pool->enqueue([this, io, i, j]() {
					abit2[j]->setup_recv();
					io->flush(j);
				}));
			} else if (j == party) {
				res.push_back(pool->enqueue([this, io, i, j]() {
					abit2[i]->setup_recv();
					io->flush(i);
				}));
				res.push_back(pool->enqueue([this, io, i, j]() {
					abit1[i]->setup_send();
					io->flush(i);
				}));
			}
		}
		joinNclean(res);

		Delta = bool_to128(tmp); 
		delete[] tmp;
	}
	~ABitMP() {
		for(int i = 1; i <= nP; ++i) if( i!= party ) {
			delete abit1[i];
			delete abit2[i];
		}
	}
	void compute(block * MAC[nP+1], block * KEY[nP+1], bool* data, int length) {
		vector<future<void>> res;
		
		block delta = Delta;
		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
			res.push_back(pool->enqueue([this, KEY, length, delta, party2]() {
				abit1[party2]->send_cot(KEY[party2], delta, length);
				io->flush(party2);
			}));
			res.push_back(pool->enqueue([this, MAC, data, length, party2]() {
				abit2[party2]->recv_cot(MAC[party2], data, length);
				io->flush(party2);
			}));
		}
		joinNclean(res);
	}
};
#endif //ABIT_MP_H__
