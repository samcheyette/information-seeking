from grammar import *
from utils import *
import numpy as np
from numpy import log, exp
import os
import time
import csv
import pickle as pkl
from scipy.special import softmax 

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


def get_all_guesses(n, k_min, k_max):
    all_combs = generate_combinations(n, k_min=k_min, k_max=k_max)
    guesses = []
    for i in range(len(all_combs)):
        guess = []
        for j in range(len(all_combs[i])):
            if all_combs[i][j] == 1:
                guess.append(j)

        guesses.append(guess)
    return guesses




def evaluate_likelihood_filt(test, output_cache,alpha, noise, beta=1):


    def norm(ps):
        return ps/np.sum(ps)

    guess, feedback = test
    guess_bin = binary_value(guess)
    codes = output_cache['codes']

    output_info = output_cache['output_info'][(guess_bin, feedback)]

    filter_diff_above = output_info["filter_diff_above"] 
    filter_diff_below = output_info["filter_diff_below"]

    priors = output_info["priors"]

    p_filters = norm(softmax((alpha * priors - filter_diff_above)/beta))

    outputs = output_info["filter_outputs"]

    p_output = p_filters.dot(outputs) 
    p_output = p_output * (1-noise) +  noise/2



    return p_output 







def evaluate_likelihood_true(test, output_cache, noise=0.01):
    guess, feedback = test
    guess_bin = binary_value(guess)
    codes = output_cache['codes']

    output_info = output_cache['output_info'][(guess_bin, feedback)]


    lkhd_true = output_info["true_likelihood"]

    lkhd = lkhd_true * (1-noise) +  noise/2

    return lkhd





def get_p_consistent(posterior, codes, guess, feedback):
    p = 0
    for i in range(len(codes)):
        if get_overlap(codes[i], guess) == feedback:
            p += posterior[i]

    return p



def get_EV(prior, codes, guess, alpha, noise, lam, beta, output_cache):

    value= 0
    code_length = len(codes[0])
    for k in range(min(code_length, len(guess)) + 1):
        test = (guess, k)
        p_consistent = get_p_consistent(prior, codes, guess, k)


        if p_consistent > sm:

            if alpha > 0:
                lkhd = evaluate_likelihood_filt(test, output_cache, alpha, noise, beta)
            else:
                lkhd = evaluate_likelihood_true(test, output_cache,noise)

            
            posterior = normalize(prior * lkhd)


            KL = get_KL(prior, posterior)
            value += KL * p_consistent


    if len(guess) == len(codes[0]):
        value +=  lam * prior[codes.index(guess)]

    return value




# def random_guess(prior, codes, all_guesses, eps=0.01):

#     if get_entropy(prior) > eps:
#         return random.choice(all_guesses)

#     else:
#         return codes[np.argmax(prior)]


# def get_best_guess(prior, codes, all_guesses, alpha, noise, lam,output_cache):


#     if get_entropy(prior) < 0.01:
#         return codes[np.argmax(prior)]
#     else:

#         best_guess, best_value = None, 0
#         idxs = [i for i in range(len(all_guesses))]
#         random.shuffle(idxs)
#         for i in idxs:
#             guess = all_guesses[i]
#             value = get_EV(prior, codes, guess, alpha, noise, lam,output_cache)
#             if value > best_value:  
#                 best_guess, best_value = guess, value
#             elif value == best_value and random.random()> 0.5:
#                 best_guess, best_value = guess, value


#         return best_guess




# def get_EV_guesses(prior, codes, all_guesses, alpha, noise, lam,output_cache):

#     values = np.zeros(len(all_guesses))

#     for i in range(len(all_guesses)):
#         guess = all_guesses[i]
#         value = get_EV(prior, codes, guess, alpha, noise, lam,output_cache)
#         values[i] = value
#     return values


# def sample_guess(prior, codes, all_guesses,alpha, noise, lam, temp, output_cache):
#     EVs = get_EV_guesses(prior, codes, all_guesses, alpha, noise, lam,output_cache)

#     p_guesses = softmax(EVs/max(temp, sm))

#     sample_idx = np.random.choice(len(p_guesses), p=p_guesses)

#     return all_guesses[sample_idx]



# def run_game(code, codes, all_guesses, alpha, noise, lam, temp, output_cache):

#     prior = normalize(np.ones(len(codes)))
#     true_prior = normalize(np.ones(len(codes)))


#     for guess_number in range(15):

#         guess = sample_guess(prior, codes, all_guesses, alpha, noise, lam, temp, output_cache)


#         EIG_filt = get_EV(prior, codes, guess, alpha, noise, 0,output_cache)
#         EIG_true = get_EV(prior, codes, guess, 0, 0, 0, output_cache)

#         feedback = get_overlap(code, guess)

#         test=(guess, feedback)

#         lkhd = evaluate_likelihood_filt(test, output_cache, alpha, noise)
#         true_lkhd = evaluate_likelihood_true(test, output_cache, 0)

#         true_posterior = normalize(true_prior *true_lkhd)
#         posterior = normalize(prior * lkhd)
#         print("-"*15)
#         print(f"\nN: {guess_number}, guess: {guess}, EIG true: {round(EIG_true, 2)}, EIG filter: {round(EIG_filt,2)}, feedback: {feedback}\n")
#         for i in range(len(codes)):
#             if posterior[i] > 0.05 or true_posterior[i] > 0.05:
#                 print(codes[i], np.round(true_posterior[i],2), np.round(posterior[i],2))
#         print("")

#         if guess == code:
#             print("game over")
#             return

#         true_prior = copy.deepcopy(true_posterior)
#         prior = copy.deepcopy(posterior)

def random_guess(prior, codes, all_guesses,eps=0.1):

    if get_entropy(prior) > eps:
        return random.choice(all_guesses)

    else:
        return codes[np.argmax(prior)]




def get_best_guess(prior, codes, all_guesses, alpha, noise, lam, beta, output_cache):


    if get_entropy(prior) < 0.01:
        return codes[np.argmax(prior)]
    else:

        best_guess, best_value = None, 0
        idxs = [i for i in range(len(all_guesses))]
        random.shuffle(idxs)
        for i in idxs:
            guess = all_guesses[i]
            value = get_EV(prior, codes, guess,alpha, noise, lam, beta, output_cache)
            if value > best_value:  
                best_guess, best_value = guess, value
            elif value == best_value and random.random()> 0.5:
                best_guess, best_value = guess, value
        return best_guess



def get_EV_guesses(prior, codes, all_guesses, alpha, noise, lam, beta, output_cache):

    values = np.zeros(len(all_guesses))

    for i in range(len(all_guesses)):
        guess = all_guesses[i]
        if alpha > 0:
            value = get_EV(prior, codes, guess, alpha, noise, lam, beta, output_cache)
        else:
            value = get_EV(prior, codes, guess, alpha, noise, lam, beta, output_cache)

        values[i] = value
    return values




def sample_guess(prior, codes, all_guesses, alpha, noise, lam, temp,beta,  output_cache):
    EVs = get_EV_guesses(prior, codes, all_guesses, alpha, noise, lam,beta, output_cache)

    p_guesses = softmax(EVs/max(temp, sm))

    sample_idx = np.random.choice(len(p_guesses), p=p_guesses)

    return all_guesses[sample_idx]



def run_game(code, codes, all_guesses, alpha, noise, lam, temp, beta, policy, output_cache):

    prior = normalize(np.ones(len(codes)))
    true_prior = normalize(np.ones(len(codes)))

    valid_codes = copy.deepcopy(codes)
    dcts = []
    for guess_number in range(15):
        n_codes_remaining_prev = len(valid_codes)
        if policy == "random":
            if len(valid_codes) > 2:
                guess = all_guesses[np.random.randint(0,len(all_guesses))]
            else:
                guess = valid_codes[0]

        elif policy == "random_valid":
            guess = valid_codes[np.random.randint(0,len(valid_codes))]

        elif policy == "maximize":
            guess = get_best_guess(prior, codes, all_guesses, alpha, noise, lam, beta,  output_cache)
        else:

            guess = sample_guess(prior, codes, all_guesses, alpha, noise, lam, temp,beta,  output_cache)



        EIG_filt = get_EV(prior, codes, guess,alpha, noise, 0, beta, output_cache)
        EIG_true = get_EV(prior, codes, guess, 0, 0, 0, 0, output_cache)
        feedback = get_overlap(code, guess)

        test=(guess, feedback)

        true_lkhd = evaluate_likelihood_true(test,output_cache, 0)

        if alpha > 0:

            lkhd = evaluate_likelihood_filt(test, output_cache, alpha, noise, beta)
        else:
            lkhd = evaluate_likelihood_true(test,output_cache, noise)

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


        dct = {"guess_number":guess_number, "true_code":" ".join([str(x) for x in code]), "guess": " ".join([str(x) for x in guess]), "policy":policy,
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


    print(len(combined_filter))
    return combined_filter, combined_nofilter


if __name__ == "__main__":



    n_positions, set_size = 12,3
    #alpha, noise, lam, temp = 0, 0.0, 0.5, None


    policy = "play"
    filt, betafit = True, False

    output_cache_file = f"cache_files/outputs_{n_positions}_{set_size}.pkl"
    with open(output_cache_file, "rb") as f:
        output_cache = pkl.load(f)



    options = output_cache['codes']
    all_guesses = get_all_guesses(n_positions, 1, n_positions//2)



    params_filter, params_no_filter = get_params("cache_files/params/")


    params = params_filter if filt else params_no_filter

    output_dcts = []




    r_id = 0
    for i in range(10):
        for j in range(len(params)):
            code = options[np.random.randint(0, len(options))]

            beta = params[j]["beta"] if ("beta" in params[j] and betafit) else 1

            alpha = params[j]["alpha"] if filt else 0
            lam = params[j]["lam"]
            temp = params[j]["temp"]
            noise = params[j]["noise"]




            print("="*50)
            print("")

            print(params[j])
            if random.random() < 0.2:
                print("~")

                alpha,lam, temp, noise  = 9, 4, 0.01, 0.1
            print(code)




            dcts = run_game(code, options, all_guesses, alpha, noise,lam, temp, beta, policy, output_cache)


            ents = []
            for d in dcts:
                d["alpha"] = params[j]["alpha"] if filt else 0
                d["lam"] = params[j]["lam"]
                d["temp"] = params[j]["temp"]
                d["noise"] =  params[j]["noise"]
                d["beta"] = params[j]["beta"] if ("beta" in params[j] and betafit) else 1

                d["r_id"] = r_id
                ents.append(np.log2(d["codes_remaining_prev"]))



            output_dcts += dcts

            if filt:
                stimuli_to_csv(output_dcts, f"model_files/filter_sim_{n_positions}_{set_size}.csv")
                stimuli_to_csv(output_dcts, f"experiments/basic_game/data/filter_sim_{n_positions}_{set_size}.csv")
            else:
                stimuli_to_csv(output_dcts, f"model_files/no_filter_sim_{n_positions}_{set_size}.csv")
                stimuli_to_csv(output_dcts, f"experiments/basic_game/data/no_filter_sim_{n_positions}_{set_size}.csv")

            r_id += 1


    # output_dcts = []
    # noise, lam, temp, beta  = 0.0, 1, 0.0, 1
    # policy = "play"
    # alphas = [0,0.1,0.3,1,3,9]
    # #alphas = [0,1,2,4,8,16]

    # r_id = 0
    # for i in range(500):
    #     code = options[np.random.randint(0, len(options))]
    #     #for policy in ["random", "maximize"]:
    #     for alpha in alphas:
    #         print("="*50)
    #         print(alpha, noise, lam, temp, beta)
    #         dcts = run_game(code, options, all_guesses, alpha, noise,lam, temp, beta, policy,output_cache)
    #         for d in dcts:
    #             d["r_id"] = r_id
    #         output_dcts += dcts
    #         #stimuli_to_csv(output_dcts, f"model_files/basemodels_{n_positions}_{set_size}.csv")
    #         stimuli_to_csv(output_dcts, f"model_files/games_{n_positions}_{set_size}.csv")


    #         r_id += 1
    #         print("")