import numpy as np
from itertools import combinations
from grammar import *
from utils import *


likelihood_cache = {}




def get_EIG_filters(guess, options, prior, noise, n_samples = 250, n_chains=2,thin=10, p_sample= 0.1):
    EIG = 0

    gt0 = np.sum([x > 0 for x in prior])

    indices = np.random.choice(len(options), size=min(round(p_sample*len(options)), gt0), replace=False, p=prior)
    total_p = 0.
    for idx in range(len(indices)):
        k = indices[idx]
        opt = options[k]
        if prior[k] > 0:


            test = (guess, get_overlap(opt, guess))
            true_lkhd = get_true_likelihoods(options,test)

            filt_lkhd =  run_mcmc(options, copy.deepcopy(grammar),test, noise, 
                    steps=n_samples, thin=thin, chains=chains, verbose=False)

            post = normalize([prior[i] * filt_lkhd[i] for i in range(len(prior))])

            KL = get_KL(prior, post)
            entropy = get_entropy(post)
            EIG += prior[k] * KL
            total_p += prior[k]
    return EIG/total_p








def run_mcmc(options, grammar, test, noise,
                chains=1, steps=10000, burnin=0.1, thin=100, verbose=False ):

    true_lkhds = get_true_likelihoods(options, test)

    posterior_predictive, samples = [0 for _ in range(len(options))], 0

    possible_sets = find_possible_sets(test[0].execute({}), test[1])
    hypothesis = find_disjunctive_representation(possible_sets, grammar)

    for chain in range(chains):
        #hypothesis = sample(grammar)


        h_lkhd, h_evals = get_filter_likelihood(hypothesis, options, test, true_lkhds)

        h_prior = hypothesis.log_probability
        h_posterior = h_prior + h_lkhd


        for step in range(steps):


            proposal = resample_random_subtree(hypothesis, grammar)
            proposal_lkhd, proposal_evals = get_filter_likelihood(proposal, options, test, 
                                            true_lkhds, noise=noise, likelihood_limit=h_posterior)
            proposal_prior = proposal.log_probability
            proposal_posterior = proposal_prior + proposal_lkhd

            if proposal_posterior - h_posterior > log(np.random.random()):
                hypothesis = copy.deepcopy(proposal)
                h_lkhd, h_prior, h_posterior = proposal_lkhd, proposal_prior, proposal_posterior
                h_evals = copy.deepcopy(proposal_evals)


            if verbose and (step % round(steps*0.1) == 0):
                print(test)
                print(chain, step, hypothesis, np.round(h_prior), 
                            np.round(h_lkhd), np.round(h_posterior))
                print("")



            if step > burnin*steps and step % thin == 0 and h_evals != None:
                samples += 1
                #sum_h = sum(h_evals)
                #if sum_h > 0:
                for i in range(len(options)):
                    posterior_predictive[i] += h_evals[i] * exp(h_posterior)


        #hypothesis = sample(grammar)


    norm = sum(posterior_predictive)
    posterior_predictive = list(map(lambda n: n/norm, posterior_predictive))
    
    return posterior_predictive




def sample_guess(grammar, guess_grammar, options, prior, noise, epsilon = 1e-1, samples=15, use_filters=False, filter_mcmc=100):
    if get_entropy(prior) < epsilon:
        return options[np.argmax(prior)], 0

    else:
        best_guess_expr = sample(guess_grammar)
        best_guess = best_guess_expr.evaluate({})
        if Number(None) not in best_guess:
            best_guess = List(*[Number(x) for x in list(set(best_guess))])
            if use_filters:
                best_EIG = get_EIG_filters(best_guess, options, prior, noise, n_samples=filter_mcmc)
            else:
                best_EIG = get_EIG(best_guess, options, prior )
            print(best_guess, np.round(best_EIG,2), np.round(get_EIG(best_guess, options, prior ),2))

        else:
            best_guess = random.choice(options)
            best_EIG = -float("inf")

        for i in range(samples):

            proposed_guess_expr = resample_random_subtree(best_guess_expr, guess_grammar)
            proposed_guess = proposed_guess_expr.evaluate({})
            if Number(None) not in proposed_guess:
                proposed_guess =  List(*[Number(x) for x in list(set(proposed_guess))])
                if use_filters:
                    proposed_EIG = get_EIG_filters(proposed_guess, options, prior, noise, n_samples=filter_mcmc)
                else:
                    proposed_EIG = get_EIG(proposed_guess, options, prior )
            else:
                proposed_EIG = -float("inf")

            if proposed_EIG >= best_EIG:

                if use_filters:
                    #resample so that we don't get bad estimates of EIG
                    best_EIG = get_EIG_filters(proposed_guess, options, prior, noise, n_samples=filter_mcmc)
                else:
                    best_EIG = proposed_EIG
                best_guess_expr = copy.deepcopy(proposed_guess_expr)
                best_guess = copy.deepcopy(proposed_guess)

                print(best_guess, np.round(best_EIG,2), np.round(get_EIG(best_guess, options, prior ),2))

    return best_guess, best_EIG




if __name__ == "__main__":



    n_positions = 8
    code_length = 3
    noise = 0.01
    chains, steps, burnin, thin = 2, 10000, 0.1, 100

    positions = [Number(i) for i in range(n_positions)]
    options = get_all_sets(n_positions, code_length)

    true_posterior = [1/len(options) for _ in range(len(options))]
    filtered_posterior = [1/len(options) for _ in range(len(options))]


    true_code = options[0]
    #guess=sample_guess(positions, filtered_posterior, grammar)

    grammar = make_grammar(n_positions)
    guess_grammar = make_guess_grammar(n_positions)


    guess_number = 0


    while get_entropy(filtered_posterior) > 0.01:

        #guess = List(positions[0], positions[1], positions[2], positions[3])

        guess, EIG = sample_guess(grammar, guess_grammar, options, 
                            filtered_posterior, noise, epsilon = 1e-1, samples=25, use_filters=False, filter_mcmc = 100)
        feedback = get_overlap(true_code, guess)
        test = (guess, feedback)


        print("")
        print(guess_number, true_code, guess, feedback, np.round(EIG,1))
        print("")




        posterior_predictive = run_mcmc(options, copy.deepcopy(grammar), test,noise,
                         chains, steps, burnin, thin,
                             verbose=True)

        true_lkhds = get_true_likelihoods(options, test)
        true_lkhds = list(map(lambda n: n/sum(true_lkhds), true_lkhds))

        filtered_posterior = normalize([posterior_predictive[i] * filtered_posterior[i] for i in range(len(posterior_predictive))])
        true_posterior = normalize([true_lkhds[i] * true_posterior[i] for i in range(len(posterior_predictive))])

        diff = 0
        for i in range(len(posterior_predictive)):
            if (true_posterior[i] > 2/len(options) or filtered_posterior[i] > 2/len(options) ):
                print(options[i], round(true_posterior[i],2),
                                 round(filtered_posterior[i],2))

            diff += np.abs(filtered_posterior[i] - true_posterior[i])

        print("")

        print(np.round(diff,2),np.round(get_entropy(true_posterior),2), 
                    np.round(get_entropy(filtered_posterior),2))
        print("")

        guess_number += 1