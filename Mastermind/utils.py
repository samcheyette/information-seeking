import numpy as np
import os
from itertools import combinations
import heapq
import pickle as pkl





sm = 1e-5
def normalize(ps):
    sum_ps = sum(ps)
    return [p/sum_ps for p in ps]

def get_KL(P, Q):
    KL = 0
    for i in range(len(P)):
        if P[i] == 0:

            assert(Q[i] < sm)
        else:
            if Q[i] > 0:
                KL += Q[i] * np.log2(Q[i]/P[i])
    return KL

def get_entropy(P):
    entropy = 0
    for i in range(len(P)):
        if P[i] > 0:
            entropy += -P[i] * np.log2(P[i])
    return entropy

def get_EIG(guess, options, prior):
    EIG = 0
    for i in range(len(options)):
        opt = options[i]
        if prior[i] > 0:
            test = (guess, get_overlap(opt, guess))
            lkhd = get_true_likelihoods(options,test)
            post = normalize([lkhd[k] * prior[k] for k in range(len(options))])
            KL = get_KL(prior, post)
            entropy = get_entropy(post)
            EIG += prior[i] * KL

    return EIG

def get_overlap(true_code, guess):

    letter_dict_code = {}
    letter_dict_guess = {}
    n_green = 0
    for i in range(len(guess)):
        if (not (true_code[i] in letter_dict_code)):
            letter_dict_code[true_code[i]] = 0
        if (not (guess[i] in letter_dict_guess)):
            letter_dict_guess[guess[i]] = 0    
        
        letter_dict_code[true_code[i]] += 1
        letter_dict_guess[guess[i]] += 1

        if (true_code[i] == guess[i]):
            n_green += 1

    n_yellow = 0
    for key in letter_dict_code:
        if (key in letter_dict_guess):
            n_yellow += min(letter_dict_guess[key], letter_dict_code[key])

    n_yellow -= n_green

    return (n_yellow, n_green)


def get_single_likelihood(option, test):
    guess, feedback = test
    n_yellow, n_green = feedback
    overlap = get_overlap(option, guess)

    if overlap == feedback:
        return 1
    else:
        return 0

def get_true_likelihoods(options,test):

    n_options = len(options)
    lkhds = []
    for i in range(len(options)):
        option = options[i]
        lkhds.append(get_single_likelihood(option, test))
    return lkhds





def get_feedback_options(guess, codes):
    n = len(guess)
    possible_feedback = set()
    for code in codes:
        possible_feedback.add(get_overlap(guess, code))
    return sorted(list(possible_feedback))



def generate_combinations(n, k_min=1, k_max=3):
    arrays = []

    for k in range(k_min,k_max+1):
        index_combinations = list(combinations(range(n), k))

        for indices in index_combinations:
            array = np.zeros(n, dtype=int)
            array[list(indices)] = 1
            arrays.append(array)
    return np.array(arrays)


def array_to_unique_value(arr, K):
    
    unique_value = 0
    for i, num in enumerate(reversed(arr)):
        unique_value += (num - 1) * (K ** i)
    return unique_value


def write_to_cache(cache_file, contents={}):
    with open(cache_file, "wb") as f:
        pkl.dump(contents, f)
    return


def load_from_cache(cache_file, default={}):

    if not os.path.exists(cache_file):
        with open(cache_file, "wb") as f:
            pkl.dump(default, f)
        return default
    else:
        with open(cache_file, "rb") as f:
            file = pkl.load(f)
        return file



class PriorityQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = []
        self.seen = set()
        self.index = 0  

    def add(self, variable, weight):
        str_v = str(variable)

        if str_v not in self.seen:
            self.seen.add(str_v)
            if len(self.queue) < self.capacity:
                heapq.heappush(self.queue, (weight, self.index, variable))
                self.index += 1
            elif weight > self.queue[0][0]:
                heapq.heappop(self.queue)
                heapq.heappush(self.queue, (weight, self.index, variable))
                self.index += 1

    def get_variables(self):
        return [var for _, var in self.queue]


    def get_all(self):
        return [(t[2],t[0]) for t in self.queue]
