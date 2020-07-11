#coding:utf-8

import sys
import mmh3
import math
import numpy as np
import copy
import pickle
import operator

k = int(sys.argv[1])
c = int(sys.argv[2])
rounds = int(sys.argv[3])

_mersenne_prime = (1 << 29) - 1

rho, phi, pi, S_dict = dict(), dict(), dict(), dict()
lsh_table_opt, lsh_table_fast, lsh_table_opt_fast, lsh_table_bi = dict(), dict(), dict(), dict()
b = int(math.floor(k / c))
for i in range(b):
    lsh_table_opt[i] = dict()
    lsh_table_fast[i] = dict()
    lsh_table_opt_fast[i] = dict()
    lsh_table_bi[i] = dict()


class Object:
    def __init__(self, index, similarity):
        self.index = index
        self.similarity = similarity


def sort_by_attr(lst, attr):
    lst.sort(key=operator.attrgetter(attr), reverse=True)


def build_lsh(table, u, su):
    for i in range(b):
        su_i = ''
        for item in su[c*i: (i+1)*c]:
            su_i += str(item)+','
        if su_i in table[i]:
            table[i][su_i].append(u)
        else:
            table[i][su_i] = [u]


def retriv_lsh(table, su):
    rst = []
    for i in range(b):
        su_i = ''
        for item in su[c*i: (i+1)*c]:
            su_i += str(item)+','
        if su_i in table[i]:
            rst.extend(table[i][su_i])
        else:
            continue
    length_m = len(rst)
    rst = list(set(rst))
    return rst, length_m


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


def similarity(obj1, obj2):
    intersection = set(obj1).intersection(set(obj2))
    union = set(obj1).union(set(obj2))
    return float(len(intersection)) / len(union)


if __name__ == '__main__':
    rho = pickle.load(open('parameters/rho' + str(k) + '.txt', 'rb'))
    phi = pickle.load(open('parameters/phi' + str(k) + '.txt', 'rb'))
    pi = pickle.load(open('parameters/pi' + str(k) + '.txt', 'rb'))
    S_dict = pickle.load(open('parameters/s' + str(k) + '.txt', 'rb'))

    dict_data = dict()
    freader = open('datasets/kdd_algebra')
    index = 0
    for line in freader:
        lst_vector = line.strip().split(' ')[1:]
        lst_index = []
        for each in lst_vector:
            lst_index.append(int(each.split(':')[0]))
        dict_data[index] = lst_index
        index += 1
    freader.close()
    print('reading data finished!')

    for i in range(rounds):
        for key in dict_data:
            seed = np.random.randint(k)
            i_u = dict_data[key]
            sketch = OPH(i_u, i)

            sketch_densified1 = OPTDens(sketch, 0)
            build_lsh(lsh_table_opt, key, sketch_densified1)

            sketch_densified2 = FastDens(sketch, 0)
            build_lsh(lsh_table_fast, key, sketch_densified2)

            sketch_densified3 = OPT_FastDens(sketch, 0)
            build_lsh(lsh_table_opt_fast, key, sketch_densified3)

            sketch_densified4 = BiDens(sketch, 0)
            build_lsh(lsh_table_bi, key, sketch_densified4)

        for i in range(100):
            i_u = dict_data[i]
            sketch = OPH(i_u, i)

            sketch_densified1 = OPTDens(sketch, 0)
            rst_opt, len_opt = retriv_lsh(lsh_table_opt, sketch_densified1)
            rst_opt.remove(i)
            lst_tmp1 = []
            for each in rst_opt:
                i_v = dict_data[each]
                jcd = similarity(i_u, i_v)
                lst_tmp1.append(Object(each, jcd))
            sort_by_attr(lst_tmp1, 'similarity')
            lst_candidates1 = []
            for each in lst_tmp1:
                lst_candidates1.append(each.index)

            sketch_densified2 = FastDens(sketch, 0)
            rst_fast, len_fast = retriv_lsh(lsh_table_fast, sketch_densified2)
            rst_fast.remove(i)
            lst_tmp2 = []
            for each in rst_fast:
                i_v = dict_data[each]
                jcd = similarity(i_u, i_v)
                lst_tmp2.append(Object(each, jcd))
            sort_by_attr(lst_tmp2, 'similarity')
            lst_candidates2 = []
            for each in lst_tmp2:
                lst_candidates2.append(each.index)

            sketch_densified3 = OPT_FastDens(sketch, 0)
            rst_opt_fast, len_opt_fast = retriv_lsh(lsh_table_opt_fast, sketch_densified3)
            rst_opt_fast.remove(i)
            lst_tmp3 = []
            for each in rst_opt_fast:
                i_v = dict_data[each]
                jcd = similarity(i_u, i_v)
                lst_tmp3.append(Object(each, jcd))
            sort_by_attr(lst_tmp3, 'similarity')
            lst_candidates3 = []
            for each in lst_tmp3:
                lst_candidates3.append(each.index)

            sketch_densified4 = BiDens(sketch, 0)
            rst_bi, len_bi = retriv_lsh(lsh_table_bi, sketch_densified4)
            rst_bi.remove(i)
            lst_tmp4 = []
            for each in rst_bi:
                i_v = dict_data[each]
                jcd = similarity(i_u, i_v)
                lst_tmp4.append(Object(each, jcd))
            sort_by_attr(lst_tmp4, 'similarity')
            lst_candidates4 = []
            for each in lst_tmp4:
                lst_candidates4.append(each.index)