import random
import resource
import time
import gmpy2
import phe.paillier as paillier
import sys
import socket                

HOST = "localhost"
PORT = 12345                


def bench_encrypt(pubkey, nums):
    for num in nums:
        pubkey.encrypt(num)


def bench_decrypt(prikey, nums):
    for num in nums:
        prikey.decrypt(num)


def bench_add(nums1, nums2):
    for num1, num2 in zip(nums1, nums2):
        num1 + num2


def bench_mul(nums1, nums2):
    for num1, num2 in zip(nums1, nums2):
        num1 * num2


def time_method(method, *args):
    start = time.time()
    method(*args)
    return time.time() - start




def party0(key_size, pubkey, prikey, bitlen, num_iterations=1000):
    # Send to party 1
    # s = socket.socket()
    # receive data from the server 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    for i in range(num_iterations):
        
        share_a, share_b = random.randrange(0, pow(2, bitlen)), random.randrange(0, pow(2, bitlen))
        enc_a, enc_b = pubkey.raw_encrypt(share_a), pubkey.raw_encrypt(share_b)
        
        msg_a = str(enc_a).rjust(key_size * 2, '0').encode()
        msg_b = str(enc_b).rjust(key_size * 2, '0').encode()
        s.sendall(msg_a)
        s.sendall(msg_b)
        rec = int(s.recv(key_size * 2).decode('utf-8'))
        d = paillier.EncryptedNumber(pubkey, rec)
        x = prikey.decrypt(d)
        share_c = (share_a * share_b + x) % pow(2, bitlen)
        #print(share_a, share_b, share_c)
    #return share_a, share_b, share_c
    s.close()



def party1(key_size, pubkey, bitlen, sigma, num_iterations=1000):
    prikey = paillier.PaillierPrivateKey(pubkey, p, q)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))         
    s.listen(10)
    client_socket, addr = s.accept()
    #enc_a, enc_b = paillier.EncryptedNumber(pubkey, enc_a), paillier.EncryptedNumber(pubkey, enc_b)
    for i in range(num_iterations):
        ran_num = random.randrange(0, pow(2, 2 * bitlen + sigma + 1))
        share_a, share_b = random.randrange(0, pow(2, bitlen)), random.randrange(0, pow(2, bitlen))
        share_c = (share_a * share_b - ran_num) % pow(2, bitlen)
        enc_a = client_socket.recv(key_size * 2)
        enc_b = client_socket.recv(key_size * 2)
        enc_a = paillier.EncryptedNumber(pubkey, int(enc_a.decode('utf-8')))
        enc_b = paillier.EncryptedNumber(pubkey, int(enc_b.decode('utf-8')))
        d = share_b * enc_a + share_a * enc_b + ran_num
        msg_d = str(d.ciphertext()).rjust(key_size * 2, "0").encode()
        #print(i)
        #print(share_a, share_b, share_c)
        client_socket.sendall(msg_d)
    
    client_socket.close()
    s.close()




n = 4037477155185516226725595957893750980511213094250968019634065687566236170699006711404262434260630571154019177756727787920456226956349059374205642839103726400508706696243791679280352691118587456595217277465138924400343022640991042955526614604442920236638593650650373841322757976125626137807330061182533526387602365132937151802841052801276722181725020535070981882225602899059205223649153825804750890004201662594424898685635987531504656306869732639728294590104815873157619550733603683309702324438943542890698564440348551010636472498719772084945013004710485040141284090077504283389254502452570992051761224643751263452164028462467293542176355133741814498359215872939232935156924417599179353676766441367660611676224938578732671647426519342811962687927021654425455475490618985138332807105240001753259945992603197367912242560395345288216509629076892028591600529560983542562475908858030720223827760554782376225539993500489764853035593
p = 1756567616486260686990588645215930235155833788355361852062119691150794841474575702910688493625441844672996789199685667154702631616982647079821675818131001687256428548882101905799431668373038223629729467609842762655833662424528321141219463831117234508159805588651303976261859851626572388762132489052219390731107668493487085026634712142251038933155886286221613851869333659304806982662447261705194456120071039701029347567376287513664830018137400494880830883303018859
q = 2298503693960759117996760058607098895545288240519069012796936895763559842603671641111827168910216944054878097805787073735044366547803326194128315821627081524339574951087159400208376015865723019661429114336853022999757367646117565139038069435146297632053430108112246616098319292097663708012894535347038736002338768656523960457619852628161213474949316168004496259154336512793588445227419718494407170497159399800323002949448701694682903442972338640026988127368729627
def main():
    key_size = 3072
    bitlen = 32
    sigma = 40
    num_iterations = 1000
    #pubkey, prikey = paillier.generate_paillier_keypair(n_length=key_size)
    pubkey = paillier.PaillierPublicKey(n)
    prikey = paillier.PaillierPrivateKey(pubkey, p, q)
    pubkey.max_int = 1 << (key_size * 4)
    start = time.time()
    if sys.argv[1] == "0":
        party0(key_size, pubkey, prikey, bitlen, num_iterations)
    else:
        party1(key_size, pubkey, bitlen, sigma, num_iterations)


    end = time.time()
    print("Total: {0} seconds, {1} seconds per iteration for {2} iterations".format((end - start), (end - start) / num_iterations, num_iterations))



main()




