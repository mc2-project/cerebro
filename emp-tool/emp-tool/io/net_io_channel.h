#ifndef NETWORK_IO_CHANNEL
#define NETWORK_IO_CHANNEL

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include "emp-tool/io/io_channel.h"
using std::string;

#ifdef UNIX_PLATFORM

#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <sys/socket.h>

#ifdef NETIO_USE_TLS
#include <unistd.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/bio.h>
#endif

namespace emp {
/** @addtogroup IO
  @{
 */

class NetIO: public IOChannel<NetIO> { public:
	bool is_server;
	int mysocket = -1;
	int consocket = -1;
	FILE * stream = nullptr;
	char * buffer = nullptr;
	bool has_sent = false;
	string addr;
	int port;
	uint64_t counter = 0;

	#ifdef NETIO_USE_TLS
		bool is_openssl_initialized = 0;
		SSL_CTX *ctx = NULL;
		SSL *ssl = NULL;
	#endif

	NetIO(const char * address, int port, bool quiet = false) {
		this->port = port;
		is_server = (address == nullptr);
		if (address == nullptr) {
			struct sockaddr_in dest;
			struct sockaddr_in serv;
			socklen_t socksize = sizeof(struct sockaddr_in);
			memset(&serv, 0, sizeof(serv));
			serv.sin_family = AF_INET;
			serv.sin_addr.s_addr = htonl(INADDR_ANY); /* set our address to any interface */
			serv.sin_port = htons(port);           /* set the server port number */
			mysocket = socket(AF_INET, SOCK_STREAM, 0);
			int reuse = 1;
			setsockopt(mysocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
			if(bind(mysocket, (struct sockaddr *)&serv, sizeof(struct sockaddr)) < 0) {
				perror("error: bind");
				exit(1);
			}
			if(listen(mysocket, 1) < 0) {
				perror("error: listen");
				exit(1);
			}
			consocket = accept(mysocket, (struct sockaddr *)&dest, &socksize);
			close(mysocket);
		}
		else {
			addr = string(address);

			struct sockaddr_in dest;
			memset(&dest, 0, sizeof(dest));
			dest.sin_family = AF_INET;
			dest.sin_addr.s_addr = inet_addr(address);
			dest.sin_port = htons(port);

			while(1) {
				consocket = socket(AF_INET, SOCK_STREAM, 0);

				if (connect(consocket, (struct sockaddr *)&dest, sizeof(struct sockaddr)) == 0) {
					break;
				}

				close(consocket);
				usleep(1000);
			}
		}

		#ifdef NETIO_USE_TLS
			if(!quiet)
				std::cout << "connected (TCP)\n";

			set_nodelay();

			openssl_init();
			ssl = SSL_new(get_ssl_ctx());
			SSL_set_fd(ssl, consocket);

			if(is_server == true){
				SSL_set_accept_state(ssl);
			}else{
				SSL_set_connect_state(ssl);
			}

			int error = SSL_do_handshake(ssl);
			if(error != 1){
			  perror("Handshake failed");

			  printf("The error number returned by SSL is: %d\n", SSL_get_error(ssl, error));

			  char error_string[256];
			  int err_in_queue;
			  while(err_in_queue = ERR_get_error()){
			    printf("An error in the queue: %s\n", ERR_error_string(err_in_queue, error_string));
			  }
				exit(1);
			}

			if(check_peer_certificate_subject() == false){
				perror("error: certificate check failed");
				exit(1);
			}
		#else
			set_nodelay();
			stream = fdopen(consocket, "wb+");
			buffer = new char[NETWORK_BUFFER_SIZE];
			memset(buffer, 0, NETWORK_BUFFER_SIZE);
			setvbuf(stream, buffer, _IOFBF, NETWORK_BUFFER_SIZE);
			if(!quiet)
				std::cout << "connected\n";
		#endif
	}

	void sync() {
		int tmp = 0;
		if(is_server) {
			send_data(&tmp, 1);
			recv_data(&tmp, 1);
		} else {
			recv_data(&tmp, 1);
			send_data(&tmp, 1);
			flush();
		}
	}

	~NetIO(){
		flush();
		close(consocket);
		delete[] buffer;
	}

	void set_nodelay() {
		const int one = 1;
		setsockopt(consocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
	}

	void set_delay() {
		const int zero = 0;
		setsockopt(consocket, IPPROTO_TCP, TCP_NODELAY, &zero, sizeof(zero));
	}

	void flush() {
		#ifdef NETIO_USE_TLS
			BIO_flush(SSL_get_wbio(ssl));
		#else
			fflush(stream);
		#endif
	}

	void send_data(const void * data, int len) {
		counter += len;
		int sent = 0;
		while(sent < len) {
			int res;
			#ifdef NETIO_USE_TLS
				res = SSL_write(ssl, sent + (char*)data, len - sent);
			#else
				res = fwrite(sent + (char*)data, 1, len - sent, stream);
			#endif
			if (res >= 0)
				sent+=res;
			else{
				fprintf(stderr,"error: net_send_data %d\n", res);
				exit(1);
			}
		}
		has_sent = true;
	}

	void recv_data(void  * data, int len) {
		if(has_sent)
			flush();
		has_sent = false;
		int sent = 0;
		while(sent < len) {
			int res;
			#ifdef NETIO_USE_TLS
				res = SSL_read(ssl, sent + (char*)data, len - sent);
			#else
				res = fread(sent + (char*)data, 1, len - sent, stream);
			#endif
			if (res >= 0)
				sent += res;
			else{
				fprintf(stderr,"error: net_send_data %d\n", res);
				exit(1);
			}
		}
	}

	#ifdef NETIO_USE_TLS
		void openssl_init(){
			if(is_openssl_initialized == false){
				SSL_load_error_strings();
		    OpenSSL_add_ssl_algorithms();
				is_openssl_initialized = true;
			}
		}
	#endif

	#ifdef NETIO_USE_TLS
		SSL_CTX* get_ssl_ctx(){
			if(ctx != NULL){
				return ctx;
			}

			SSL_CTX *tmp_ctx;
			tmp_ctx = SSL_CTX_new(TLS_method());
			if(tmp_ctx == NULL){
				perror("Failed to create the SSL context object");
				exit(1);
			}

			SSL_CTX_set_min_proto_version(tmp_ctx, TLS1_3_VERSION);
			SSL_CTX_set_max_proto_version(tmp_ctx, TLS1_3_VERSION);

			if(!SSL_CTX_set_ciphersuites(tmp_ctx, "TLS_AES_128_GCM_SHA256")){
			       	perror("Failed to set cipher suite for TLS");
				exit(1);
			}

			SSL_CTX_set_verify(tmp_ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);
			#ifndef NETIO_MY_CERTIFICATE
				#define NETIO_MY_CERTIFICATE "./certificates/my_certificate.pem"
			#endif

			#ifndef NETIO_MY_PRIVATE_KEY
				#define NETIO_MY_PRIVATE_KEY "./certificates/my_private_key.key"
			#endif

			if(access(NETIO_MY_CERTIFICATE, R_OK) != 0){
				fprintf(stderr, "Failed to load this party's certificate file %s\n%s\n", NETIO_MY_CERTIFICATE, strerror(errno));
				exit(1);
			}

			if(access(NETIO_MY_PRIVATE_KEY, R_OK) != 0){
				fprintf(stderr, "Failed to load this party's private key file %s\n%s\n", NETIO_MY_PRIVATE_KEY, strerror(errno));
                                exit(1);
			}

			if(SSL_CTX_use_certificate_file(tmp_ctx, NETIO_MY_CERTIFICATE, SSL_FILETYPE_PEM) != 1){
				perror("Failed to set the party's certificate");
				exit(1);
			}
			
			if(SSL_CTX_use_PrivateKey_file(tmp_ctx, NETIO_MY_PRIVATE_KEY, SSL_FILETYPE_PEM) != 1){
				perror("Failed to set the party's private key");
                                exit(1);
			}

			#ifndef NETIO_CA_CERTIFICATE
				#define NETIO_CA_CERTIFICATE "./certificates/ca.pem"
			#endif

			if(access(NETIO_CA_CERTIFICATE, R_OK) != 0){
				perror("Failed to load the CA certificate");
				exit(1);
			}

			SSL_CTX_set_client_CA_list(tmp_ctx, SSL_load_client_CA_file(NETIO_CA_CERTIFICATE));
			if(SSL_CTX_load_verify_locations(tmp_ctx, NETIO_CA_CERTIFICATE, NULL) != 1){
				perror("Failed to set the CA certificate");
				exit(1);
			}

			ctx = tmp_ctx;
			return ctx;
		}
	#endif

	#ifdef NETIO_USE_TLS
		bool check_peer_certificate_subject(){
			if(ssl == NULL){
				perror("The SSL object has not been established");
				exit(1);
			}

			#ifdef NETIO_USE_TLS_NONAMECHECK
				return true;
			#endif

			struct sockaddr_in addr; socklen_t addrlen = sizeof(struct sockaddr_in);
			if(getpeername(consocket, (struct sockaddr*)&addr, &addrlen) != 0){
				perror("Failed to obtain the IP address of the other party, which is used to find the party's certificate");
				return false;
			}

			char sa_info[INET_ADDRSTRLEN];
			memset(sa_info, 0, INET_ADDRSTRLEN);

			if(inet_ntop(AF_INET, &(addr.sin_addr), sa_info, INET_ADDRSTRLEN) == NULL){
				perror("Failed to interpret the other party's IP address");
				return false;
			}

			X509 *peer_cert = SSL_get_peer_certificate(ssl);
			if(peer_cert == NULL){
				perror("Failed to obtain the peer's certificate");
				return false;
			}

			char peer_cert_common_name[256];
			if(X509_NAME_get_text_by_NID(X509_get_subject_name(peer_cert), NID_commonName, peer_cert_common_name, 255) == -1){
				perror("Failed to extract the common name from the certificate from the other party");
				return false;
			}

			if(strcmp(peer_cert_common_name, sa_info) != 0){
				perror("The common name in the party's certificate does not match the party's IP address");
				return false;
			}

			return true;
		}
	#endif
};
/**@}*/

}

#else  // not UNIX_PLATFORM

#include <boost/asio.hpp>
using boost::asio::ip::tcp;

namespace emp {

/** @addtogroup IO
  @{
 */
class NetIO: public IOChannel<NetIO> {
public:
	bool is_server;
	string addr;
	int port;
	uint64_t counter = 0;
	char * buffer = nullptr;
	int buffer_ptr = 0;
	int buffer_cap = NETWORK_BUFFER_SIZE;
	bool has_send = false;
	boost::asio::io_service io_service;
	tcp::socket s = tcp::socket(io_service);
	NetIO(const char * address, int port, bool quiet = false) {
		#ifdef NETIO_USE_TLS
		#error NetIO with TLS for boost library has not yet been implemented.
		fprintf(stderr, "error: NetIO with TLS for boost library has not yet been implemented.\n");
		exit(1);
		#endif

		this->port = port;
		is_server = (address == nullptr);
		if (address == nullptr) {
			tcp::acceptor a(io_service, tcp::endpoint(tcp::v4(), port));
			s = tcp::socket(io_service);
			a.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
			a.accept(s);
		} else {
			tcp::resolver resolver(io_service);
			tcp::resolver::query query(tcp::v4(), address, std::to_string(port).c_str());
			tcp::resolver::iterator iterator = resolver.resolve(query);

			s = tcp::socket(io_service);
			s.connect(*iterator);
		}
		s.set_option( boost::asio::socket_base::send_buffer_size( 65536 ) );
		buffer = new char[buffer_cap];
		set_nodelay();
		if(!quiet)
			std::cout << "connected\n";
	}
	void sync() {
		int tmp = 0;
		if(is_server) {
			send_data(&tmp, 1);
			recv_data(&tmp, 1);
		} else {
			recv_data(&tmp, 1);
			send_data(&tmp, 1);
			flush();
		}
	}

	~NetIO() {
		flush();
		delete[] buffer;
	}

	void set_nodelay() {
		s.set_option(boost::asio::ip::tcp::no_delay(true));
	}

	void set_delay() {
		s.set_option(boost::asio::ip::tcp::no_delay(false));
	}

	void flush() {
		boost::asio::write(s, boost::asio::buffer(buffer, buffer_ptr));
		buffer_ptr = 0;
	}

	void send_data(const void * data, int len) {
		counter += len;
		if (len >= buffer_cap) {
			if(has_send) {
				flush();
			}
			has_send = false;
			boost::asio::write(s, boost::asio::buffer(data, len));
			return;
		}
		if (buffer_ptr + len > buffer_cap)
			flush();
		memcpy(buffer + buffer_ptr, data, len);
		buffer_ptr += len;
		has_send = true;
	}

	void recv_data(void  * data, int len) {
		int sent = 0;
		if(has_send) {
			flush();
		}
		has_send = false;
		while(sent < len) {
			int res = s.read_some(boost::asio::buffer(sent + (char *)data, len - sent));
			if (res >= 0)
				sent += res;
			else
				fprintf(stderr,"error: net_send_data %d\n", res);
		}
	}
};

}

#endif  //UNIX_PLATFORM
#endif  //NETWORK_IO_CHANNEL
