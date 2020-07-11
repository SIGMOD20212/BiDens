#coding:utf-8

import random
import mmh3
import math
import numpy as np
import copy
import time
import pickle

k = int(sys.argv[1])
gamma = float(sys.argv[2])
rounds = int(sys.argv[3])

_mersenne_prime = (1 << 29) - 1

rho, phi, pi, S_dict = dict(), dict(), dict(), dict()


def generate_sketch():
    E_Num = math.floor(gamma * k)
    sk = [1] * k
    for i in range(E_Num):
        sk[i] = -1
    random.shuffle(sk)
    return sk


def OPH(s, seed):
    sketch = [1.0] * k
    for item in s:
        rw = (mmh3.hash(str(item), seed) % _mersenne_prime) / _mersenne_prime
        iw = int(rw * k) # bucket
        sketch[iw] = min(sketch[iw], rw * k - iw)
    for i in range(k):
        if sketch[i] == 1.0:
            sketch[i] = -1
    return sketch


def OPTDens(sk, C):
    dsk = copy.deepcopy(sk)
    for i in range(k):
        if sk[i] == -1:
            j = 1
            i_next = (mmh3.hash(str((i << 32) + j), 1) % _mersenne_prime) % k
            while sk[i_next] == -1:
                j += 1
                i_next = (mmh3.hash(str((i << 32) + j), 1) % _mersenne_prime) % k
            dsk[i] = j*C + sk[i_next]
    return dsk


def FastDens(sk, C):
    Eu_lst = []
    Nu_lst = []
    for i in range(k):
        if sk[i] == -1:
            Eu_lst.append(i)
        else:
            Nu_lst.append(i)

    dsk = copy.deepcopy(sk)
    j = 1
    while len(Eu_lst) != 0:
        for i in Nu_lst:
            i_next = (mmh3.hash(str((i << 32) + j), 1) % _mersenne_prime) % k
            if dsk[i_next] == -1:
                dsk[i_next] = C*(j*k+i) + sk[i]
                Eu_lst.remove(i_next)
                if len(Eu_lst) == 0:
                    break
        j += 1
    return dsk


def OPT_FastDens(sk, C):
    Eu_lst = []
    Nu_lst = []
    for i in range(k):
        if sk[i] == -1:
            Eu_lst.append(i)
        else:
            Nu_lst.append(i)
    E_num = len(Eu_lst)
    N_num = len(Nu_lst)

    dsk = copy.deepcopy(sk)
    if E_num > N_num:
        j = 1
        while E_num > N_num:
            for i in Nu_lst:
                i_next = (mmh3.hash(str((i << 32) + j), 1) % _mersenne_prime) % k
                if dsk[i_next] == -1:
                    dsk[i_next] = C * (j * k + i) + sk[i]
                    Eu_lst.remove(i_next)
                    E_num -= 1
                    N_num += 1
                    if E_num <= N_num:
                        break
            j += 1
        for i in Eu_lst:
            j = 1
            i_next = (mmh3.hash(str((i << 32) + j), 1) % _mersenne_prime) % k
            while sk[i_next] == -1:
                j += 1
                i_next = (mmh3.hash(str((i << 32) + j), 1) % _mersenne_prime) % k
            dsk[i] = j * C + sk[i_next]
    else:
        for i in Eu_lst:
            j = 1
            i_next = (mmh3.hash(str((i << 32) + j), 1) % _mersenne_prime) % k
            while sk[i_next] == -1:
                j += 1
                i_next = (mmh3.hash(str((i << 32) + j), 1) % _mersenne_prime) % k
            dsk[i] = j * C + sk[i_next]
    return dsk


def h_fwd(i, j):
    key = tuple([i-1, j-1])
    rst = i * rho[key] % (k + 1)
    return rst


def h_inv(i, z):
    x_now = z*pi[i - 1] % (k + 1)
    key = tuple([i - 1, x_now-2])
    rst = phi[key]
    return rst


def BiDens(sk, C):
    Eu_lst = []
    Nu_lst = []
    densified = dict()
    dsk = copy.deepcopy(sk)

    for j in range(1, k+1):
        if sk[j-1] == -1:
            Eu_lst.append(j)
            densified[j] = False
        else:
            Nu_lst.append(j)
            densified[j] = True

    E_num = len(Eu_lst)
    N_num = len(Nu_lst)

    jF = 1
    if np.square(N_num) > k:
        while E_num > N_num:
            for i in Nu_lst:
                z = h_fwd(i, jF)
                if densified[z] == False:
                    dsk[z-1] = C*(jF*k+i) + sk[i-1]
                    densified[z] = True
                    E_num -= 1
            jF += 1
        while E_num != 0:
            for z in Eu_lst:
                if densified[z] == True:
                    continue
                jB = 1
                unfinish = True
                while unfinish:
                    key_tup = tuple([z, jB])
                    for i in S_dict[key_tup]:
                        if sk[i-1] > 0:
                            dsk[z-1] = C*(jB*k+i) + sk[i-1]
                            densified[z] = True
                            E_num -= 1
                            unfinish = False
                            break
                    jB += 1
    else:
        while E_num * N_num > k:
            for i in Nu_lst:
                z = h_fwd(i, jF)
                if densified[z] == False:
                    dsk[z-1] = C*(jF*k+i) + sk[i-1]
                    densified[z] = True
                    E_num -= 1
            jF += 1
        while E_num != 0:
            for z in Eu_lst:
                if densified[z] == True:
                    continue
                jB = k + 1
                S_min = []
                for i in Nu_lst:
                    if h_inv(i, z) < jB:
                        jB = h_inv(i, z)
                        S_min = [i]
                    if h_inv(i, z) == jB:
                        S_min.append(i)
                i_star = min(S_min)
                dsk[z-1] = C*(jB*k+i_star) + sk[i_star-1]
                densified[z] = True
                E_num -= 1
    return dsk


if __name__ == '__main__':
    rho = pickle.load(open('parameters/rho' + str(k) + '.txt', 'rb'))
    phi = pickle.load(open('parameters/phi' + str(k) + '.txt', 'rb'))
    pi = pickle.load(open('parameters/pi' + str(k) + '.txt', 'rb'))
    S_dict = pickle.load(open('parameters/s' + str(k) + '.txt', 'rb'))

    sketch = generate_sketch()
    t_OPTDens, t_FastDens, t_OPT_FastDens, t_BiDens = 0, 0, 0, 0

    for i in range(rounds):
        seed = np.random.randint(k)

        t_start1 = time.process_time()
        sketch_densified1 = OPTDens(sketch, 0)
        t_end1 = time.process_time()
        t_OPTDens += (t_end1 - t_start1)

        t_start2 = time.process_time()
        sketch_densified2 = FastDens(sketch, 0)
        t_end2 = time.process_time()
        t_FastDens += (t_end2 - t_start2)

        t_start3 = time.process_time()
        sketch_densified3 = OPT_FastDens(sketch, 0)
        t_end3 = time.process_time()
        t_OPT_FastDens += (t_end3 - t_start3)

        t_start4 = time.process_time()
        sketch_densified4 = BiDens(sketch, 0)
        t_end4 = time.process_time()
        t_BiDens += (t_end4 - t_start4)

    t_OPTDens /= rounds
    t_FastDens /= rounds
    t_OPT_FastDens /= rounds
    t_BiDens /= rounds

    print(str(t_OPTDens) + ' ' + str(t_FastDens) + ' ' + str(t_OPT_FastDens) + ' ' + str(t_BiDens))