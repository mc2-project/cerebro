import pickle
import hashlib
import binascii

MapNum = 11
MapSize = 80 + 24 * 64

Bound = 2 ** 64

for MapId in range(1, MapNum):
    MapArray = []
    for i in range(1, MapSize):
        m = hashlib.sha256()
        mstr = str(MapId) + "||" + str(i)
        m.update(mstr.encode())
        mres = m.digest()
        MapArray.append(int(binascii.hexlify(mres), 16) % Bound)
    with open("/home/ubuntu/mc2/Input_Data/hashmap_" + str(MapId), 'wb') as f:
        pickle.dump(MapArray, f)
