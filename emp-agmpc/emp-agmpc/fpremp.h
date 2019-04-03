#ifndef FPRE_MP_H__
#define FPRE_MP_H__
#include <emp-tool/emp-tool.h>
#include <thread>
#include "abitmp.h"
#include "netmp.h"
#include "cmpc_config.h"

using namespace emp;
template<int nP>
class FpreMP { public:
	ThreadPool *pool;
	int party;
	NetIOMP<nP> * io;
	ABitMP<nP>* abit;
	block Delta;
	PRP * prps;
	PRP * prps2;
	PRG * prgs;
	PRG prg;
	int ssp;
	FpreMP(NetIOMP<nP> * io[2], ThreadPool * pool, int party, int ssp = 40) {
		this->party = party;
		this->pool = pool;
		this->io = io[0];
		this ->ssp = ssp;
		abit = new ABitMP<nP>(io[1], pool, party);
		Delta = abit->Delta;
		prps = new PRP[nP+1];
		prps2 = new PRP[nP+1];
		prgs = new PRG[nP+1];
	}
	~FpreMP(){
		delete[] prps;
		delete[] prps2;
		delete[] prgs;
		delete abit;
	}
	int get_bucket_size(int size) {
		size = max(size, 320);
		int batch_size = ((size+2-1)/2)*2;
		if(batch_size >= 280*1000)
			return 3;
		else if(batch_size >= 3100)
			return 4;
		else return 5;
	}
	void compute(block * MAC[nP+1], block * KEY[nP+1], bool * r, int length) {
		int bucket_size = 1; //get_bucket_size(length);
		block * tMAC[nP+1];
		block * tKEY[nP+1];
		block * phi;

		bool *tr = new bool[length*bucket_size*3];
		phi = new block[length*bucket_size];
		bool *s[nP+1], *e = new bool[length*bucket_size];
		for(int i = 1; i <= nP; ++i) {
			tMAC[i] = new block[length*bucket_size*3];
			tKEY[i] = new block[length*bucket_size*3];
		}
		for(int i = 0; i <= nP; ++i) {
			s[i] = new bool[length*bucket_size];
			memset(s[i], 0, length*bucket_size);
		}
		prg.random_bool(tr, length*bucket_size*3);
		memset(tr, false, length*bucket_size*3);
		abit->compute(tMAC, tKEY, tr, length*bucket_size*3);
		vector<future<void>>	 res;

		for(int i = 1; i <= nP; ++i) for(int j = 1; j <= nP; ++j) if (i < j ) {
			if(i == party) {
				res.push_back(pool->enqueue([this, tKEY, tr, s, length, bucket_size, i, j]() {
					prgs[j].random_bool(s[j], length*bucket_size);
					for(int k = 0; k < length*bucket_size; ++k) {
						uint8_t data = garble(tKEY[j], tr, s[j], k, j);
						io->send_data(j, &data, 1);
						s[j][k] = (s[j][k] != (tr[3*k] and tr[3*k+1]));
					}
					io->flush(j);
				}));
			} else if (j == party) {
				res.push_back(pool->enqueue([this, tMAC, tr, s, length, bucket_size, i, j]() {
					for(int k = 0; k < length*bucket_size; ++k) {
						uint8_t data = 0;
						io->recv_data(i, &data, 1);
						bool tmp = evaluate(data, tMAC[i], tr, k, i);
						s[i][k] = (tmp != (tr[3*k] and tr[3*k+1]));
					}
				}));
			}
		}
		joinNclean(res);
		for(int k = 0; k < length*bucket_size; ++k) {
			s[0][k] = (tr[3*k] and tr[3*k+1]);
			for(int i = 1; i <= nP; ++i) 
				if (i != party) {
					s[0][k] = (s[0][k] != s[i][k]);
				}
			e[k] = (s[0][k] != tr[3*k+2]);
			tr[3*k+2] = s[0][k];
		}

		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
			res.push_back(pool->enqueue([this, e, length, bucket_size, party2]() {
				io->send_data(party2, e, length*bucket_size);
				io->flush(party2);
			}));
			res.push_back(pool->enqueue([this, tKEY, length, bucket_size, party2]() {
				bool * tmp = new bool[length*bucket_size];
				io->recv_data(party2, tmp, length*bucket_size);
				for(int k = 0; k < length*bucket_size; ++k) {
					if(tmp[k])
						tKEY[party2][3*k+2] = xorBlocks(tKEY[party2][3*k+2], Delta);
				}
				delete[] tmp;
			}));
		}
		joinNclean(res);
		
		//land -> and	
		for(int i = 0; i < length; ++i) {
			for(int j = 1; j <= nP; ++j) if (j!= party) {
				memcpy(MAC[j]+3*i, tMAC[j]+3*i, 3*sizeof(block));
				memcpy(KEY[j]+3*i, tKEY[j]+3*i, 3*sizeof(block));
			}
			memcpy(r+3*i, tr+3*i, 3);
		}
		
		delete[] tr;
		delete[] phi;
		delete[] e;
		for(int i = 1; i <= nP; ++i) {
			delete[] tMAC[i];
			delete[] tKEY[i];
			delete[] s[i];
		}
		delete[] s[0];
	}

	//TODO: change to justGarble
	uint8_t garble(block * KEY, bool * r, bool * r2, int i, int I) {
		uint8_t data = 0;
		block tmp[4], tmp2[4], tmpH[4];
		tmp[0] = KEY[3*i];
		tmp[1] = xorBlocks(tmp[0], Delta);
		tmp[2] = KEY[3*i+1];
		tmp[3] = xorBlocks(tmp[2], Delta);
		prps[I].Hn(tmp, tmp, 4*i, 4, tmpH);

		tmp2[0] = xorBlocks(tmp[0], tmp[2]);
		tmp2[1] = xorBlocks(tmp[1], tmp[2]);
		tmp2[2] = xorBlocks(tmp[0], tmp[3]);
		tmp2[3] = xorBlocks(tmp[1], tmp[3]);

		data = LSB(tmp2[0]);
		data |= (LSB(tmp2[1])<<1);
		data |= (LSB(tmp2[2])<<2);
		data |= (LSB(tmp2[3])<<3);
		if ( ((false != r[3*i] ) && (false != r[3*i+1])) != r2[i] )
			data= data ^ 0x1;
		if ( ((true != r[3*i] ) && (false != r[3*i+1])) != r2[i] )
			data = data ^ 0x2;
		if ( ((false != r[3*i] ) && (true != r[3*i+1])) != r2[i] )
			data = data ^ 0x4;
		if ( ((true != r[3*i] ) && (true != r[3*i+1])) != r2[i] )
			data = data ^ 0x8;
		return data;
	}
	bool evaluate(uint8_t tmp, block * MAC, bool * r, int i, int I) {
		block bH = xorBlocks(prps[I].H(MAC[3*i], 4*i + r[3*i]), prps[I].H(MAC[3*i+1], 4*i + 2 + r[3*i+1]));
		uint8_t res = LSB(bH);
		tmp >>= (r[3*i+1]*2+r[3*i]);
		return (tmp&0x1) != (res&0x1);
	}	
};
#endif// FPRE_H__
