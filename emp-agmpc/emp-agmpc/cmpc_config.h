#ifndef __CMPC_CONFIG
#define __CMPC_CONFIG
const static int abit_block_size = 1024;
const static int fpre_threads = 1;
#define NUM_PARTY_FOR_RUNNING 3
#define LOCALHOST
//#define __MORE_FLUSH
//#define __debug
//const static char *IP[] = {""
//,	"34.208.104.218"
//,	"3.18.32.232"
//,	"127.0.0.1"};

const static char *IP[] = { "",
	"127.0.0.1",
	"127.0.0.1",
	"127.0.0.1"};

const static bool lan_network = false;
#endif// __C2PC_CONFIG
