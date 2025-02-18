from grammar import *
from utils import *
import numpy as np
from numpy import log, exp
import os
import time
import pickle as pkl
from scipy.special import softmax
from scipy.stats import wasserstein_distance
import csv
import functools
print = functools.partial(print, flush=True)


sm=1e-10



def stimuli_to_csv(dcts, csv_file):

    # Write the stimuli to a CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = None

        for dct in dcts:
            if writer is None:
                fieldnames =  list(dct.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(dct)

def evaluate_likelihood_filt(test,n_colors, output_cache,
                                 alpha=0.01, noise=0, beta=1):


    def norm(ps):
        return ps/np.sum(ps)

    guess, feedback = test
    guess_bin =  array_to_unique_value(guess, n_colors)
    codes = output_cache['codes']



    output_info = output_cache['output_info'][(guess_bin, feedback)]
    lkhd_true = output_info["true_likelihood"]

    filter_diff_above = output_info["filter_diff_above"]
    filter_diff_below = output_info["filter_diff_below"]

    priors = output_info["priors"]

    p_filters = norm(softmax((alpha * priors - filter_diff_above)/beta))

    outputs = output_info["filter_outputs"]

    p_output = p_filters.dot(outputs)

    p_output = p_output * (1-noise) + noise/2 

    return p_output 





def evaluate_likelihood_true(test, n_colors, output_cache, noise=0.01):
    guess, feedback = test
    guess_bin =  array_to_unique_value(guess, n_colors)
    codes = output_cache['codes']

    output_info = output_cache['output_info'][(guess_bin, feedback)]


    lkhd_true = output_info["true_likelihood"]


    lkhd = lkhd_true * (1-noise) + noise/2


    return lkhd


def get_consistent_codes(codes, guess, feedback):
    new_codes = []
    for i in range(len(codes)):
        if get_overlap(codes[i], guess) == feedback:
            new_codes.append(codes[i])

    return new_codes



def get_p_consistent(posterior, codes, guess, feedback):
    p = 0
    for i in range(len(codes)):
        if get_overlap(codes[i], guess) == feedback:
            p += posterior[i]

    return p



def get_EV(prior, codes, guess,n_colors, alpha, noise, lam, beta, output_cache):

    value= 0
    possibilities = get_feedback_options(guess, codes)

    for feedback in possibilities:
        test = (guess, feedback)
        p_consistent = get_p_consistent(prior, codes, guess, feedback)


        if p_consistent > sm:

            if alpha > 0:
                lkhd = evaluate_likelihood_filt(test, n_colors,output_cache, alpha=alpha, noise=noise, beta=beta)
            else:
                lkhd = evaluate_likelihood_true(test, n_colors,output_cache, noise=noise)

            
            posterior = normalize(prior * lkhd)
            KL = get_KL(prior, posterior)
            value += KL * p_consistent


    if len(guess) == len(codes[0]):
        value +=  lam * prior[codes.index(guess)]

    return value


def get_best_guess(prior, codes, all_guesses,n_colors, alpha, noise, lam, beta, output_cache):


    if get_entropy(prior) < 0.01:
        return codes[np.argmax(prior)]
    else:

        best_guess, best_value = None, 0
        idxs = [i for i in range(len(all_guesses))]
        #random.shuffle(idxs)
        for i in idxs:
            guess = all_guesses[i]
            value = get_EV(prior, codes, guess,n_colors,alpha, noise, lam, beta, output_cache)
            if value > best_value:  
                best_guess, best_value = guess, value
            elif value == best_value and random.random()> 0.5:
                best_guess, best_value = guess, value
        return best_guess




def random_guess(prior, codes, all_guesses,eps=0.1):

    if get_entropy(prior) > eps:
        return random.choice(all_guesses)

    else:
        return codes[np.argmax(prior)]






def get_EV_guesses(prior, codes, all_guesses, n_colors, alpha, noise, lam,beta, output_cache):

    values = np.zeros(len(all_guesses))

    for i in range(len(all_guesses)):
        guess = all_guesses[i]
        if alpha > 0:
            value = get_EV(prior, codes, guess, n_colors, alpha, noise, lam, beta, output_cache)
        else:
            value = get_EV(prior, codes, guess, n_colors, alpha, noise, lam, beta, output_cache)

        values[i] = value
    return values




def sample_guess(prior, codes, all_guesses, n_colors, alpha, noise, lam, temp, beta, output_cache):
    EVs = get_EV_guesses(prior, codes, all_guesses, n_colors, alpha, noise, lam, beta, output_cache)

    p_guesses = softmax(EVs/max(temp, sm))

    sample_idx = np.random.choice(len(p_guesses), p=p_guesses)

    return all_guesses[sample_idx]



def run_game(code, codes, all_guesses,n_colors, alpha, noise, lam, temp, beta, policy, output_cache):

    prior = normalize(np.ones(len(codes)))
    true_prior = normalize(np.ones(len(codes)))

    valid_codes = copy.deepcopy(codes)
    dcts = []
    for guess_number in range(15):
        n_codes_remaining_prev = len(valid_codes)
        if policy == "random":
            if len(valid_codes) > 1:
                guess = codes[np.random.randint(0,len(codes))]
            else:
                guess = valid_codes[0]

        elif policy == "random_valid":
            guess = valid_codes[np.random.randint(0,len(valid_codes))]

        elif policy == "maximize":
            guess = get_best_guess(prior, codes, all_guesses, n_colors, alpha, noise, lam, beta, output_cache)
        else:

            guess = sample_guess(prior, codes, all_guesses, n_colors, alpha, noise, lam, temp, beta, output_cache)



        EIG_filt = get_EV(prior, codes, guess,n_colors,alpha, noise, 0,beta, output_cache)
        EIG_true = get_EV(prior, codes, guess,n_colors, 0, 0, 0, 1, output_cache)
        feedback = get_overlap(code, guess)

        test=(guess, feedback)

        true_lkhd = evaluate_likelihood_true(test, n_colors,output_cache, 0)

        if alpha > 0:

            lkhd = evaluate_likelihood_filt(test, n_colors,output_cache, alpha, noise, beta)
        else:
            lkhd = evaluate_likelihood_true(test, n_colors,output_cache, noise)
        n_codes_remaining_prev = len(valid_codes)

        valid_codes = [code for code in valid_codes if get_overlap(code, guess) == feedback]

        true_posterior = normalize(true_prior *true_lkhd)
        posterior = normalize(prior * lkhd)

        entropy_prev = get_entropy(prior)
        entropy_post = get_entropy(posterior)
        n_codes_remaining_post = len(valid_codes)

        true_prior = copy.deepcopy(true_posterior)
        prior = copy.deepcopy(posterior)

        print("-"*15)
        print(f"\nN: {guess_number}, guess: {guess},  true_ent_prev: {round(np.log2(n_codes_remaining_prev),2)}, mod_ent_prev: {round(entropy_prev,2)}, EIG true: {round(EIG_true, 2)}, EIG filter: {round(EIG_filt,2)}, feedback: {feedback}\n")
        for i in range(len(codes)):
            if posterior[i] > 0.05 or true_posterior[i] > 0.05:
                print(policy, codes[i], np.round(true_posterior[i],2), np.round(posterior[i],2))
        print("")



        dct = {"guess_number":guess_number, "true_code":"".join([str(x) for x in code]), "guess": "".join([str(x) for x in guess]), "policy":policy,
                "alpha":alpha, "noise":noise, "lam": lam, "temp":temp, "beta": beta, "EIG":EIG_filt, "codes_remaining_prev":n_codes_remaining_prev, 
                "codes_remaining_post":n_codes_remaining_post, "entropy_prev":entropy_prev, "entropy_post":entropy_post}

        dcts.append(dct)


        if guess == code:
            print("game over")
            return dcts

    return dcts



def get_params(directory):


    files = os.listdir(directory)

    combined_filter = []
    combined_nofilter = []

    for file in files:
        d = pkl.load(open(f"{directory}/{file}", "rb"))


        if "nonoise" not in file:

            if "nofilter" in file:
                if "nonoise" not in file:
                    combined_nofilter.append(d)
            else:
                if "beta" not in file:
                    combined_filter.append(d)


    return combined_filter, combined_nofilter


