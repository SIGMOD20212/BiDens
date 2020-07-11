#coding:utf-8

import sys
import mmh3
import numpy as np
import copy
import pickle

k = int(sys.argv[1])
rounds = int(sys.argv[2])

_mersenne_prime = (1 << 29) - 1

rho, phi, pi, S_dict = dict(), dict(), dict(), dict()


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


def OPTDens_FastDens(sk, C):
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


def exact_similarity(lst1, lst2):
    intersection = set(lst1).intersection(set(lst2))
    union = set(lst1).union(set(lst2))
    return float(len(intersection)) / len(union)


def estimated_similarity(sk1, sk2):
    N_mat = 0
    N_emp = 0
    for i in range(k):
        if sk1[i] == -1 and sk2[i] == -1:
            N_emp += 1
            continue
        else:
            if sk1[i] != -1 and sk2[i] != -1 and sk1[i] == sk2[i]:
                N_mat += 1
    return float(N_mat)/(k-N_emp)


if __name__ == '__main__':
    rho = pickle.load(open('parameters/rho' + str(k) + '.txt', 'rb'))
    phi = pickle.load(open('parameters/phi' + str(k) + '.txt', 'rb'))
    pi = pickle.load(open('parameters/pi' + str(k) + '.txt', 'rb'))
    S_dict = pickle.load(open('parameters/s' + str(k) + '.txt', 'rb'))

    dict_data = dict()
    freader = open('datasets/kddalgebra')
    index = 0
    for line in freader:
        lst_vector = line.strip().split(' ')[1:]
        lst_index = []
        for each in lst_vector:
            lst_index.append(each.split(':')[0])
        dict_data[index] = lst_index
        index += 1
    freader.close()

    MSE1, MSE2, MSE3, MSE4 = 0, 0, 0, 0
    num_computation = 0
    for _ in range(rounds):
        for i in range(index):
            for j in range(i+1, index):
                print(i, j)
                seed = np.random.randint(k)
                i_u = dict_data[i]
                i_v = dict_data[j]

                sketch1 = OPH(i_u, i)
                sketch2 = OPH(i_v, i)

                sketch1_densified1 = OPTDens(sketch1, 0)
                sketch2_densified1 = OPTDens(sketch2, 0)

                sketch1_densified2 = FastDens(sketch1, 0)
                sketch2_densified2 = FastDens(sketch2, 0)

                sketch1_densified3 = OPTDens_FastDens(sketch1, 0)
                sketch2_densified3 = OPTDens_FastDens(sketch2, 0)

                sketch1_densified4 = BiDens(sketch1, 0)
                sketch2_densified4 = BiDens(sketch2, 0)

                exact_jcd = exact_similarity(i_u, i_v)

                MSE1 += np.square(estimated_similarity(sketch1_densified1, sketch2_densified1) - exact_jcd)
                MSE2 += np.square(estimated_similarity(sketch1_densified2, sketch2_densified2) - exact_jcd)
                MSE3 += np.square(estimated_similarity(sketch1_densified3, sketch2_densified3) - exact_jcd)
                MSE4 += np.square(estimated_similarity(sketch1_densified4, sketch2_densified4) - exact_jcd)

                num_computation += 1

    MSE1 /= (rounds * num_computation)
    MSE2 /= (rounds * num_computation)
    MSE3 /= (rounds * num_computation)
    MSE4 /= (rounds * num_computation)

    print(str(k) + ' ' + str(MSE1) + ' ' + str(MSE2) + ' ' + str(MSE3) + ' ' + str(MSE4))