# %% [markdown]
# ### CHANGE reward to 1 (reward is always 1)

# %%
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
# import lanfactory
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import norm
import pickle
# import os, sys, math



# %%
model_config_rswg = {
    "RSWG": {
        "doc": "RSWG",
        "params": ["alpha+", "alpha-", "beta"],
        "param_bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 50.0]],
    }
}

# %%
def softmax(q_val, beta):
    q_val = np.array(q_val)*beta
    q_val = q_val - np.max(q_val)
    q_val = np.exp(q_val)
    q_val = q_val / np.sum(q_val)
    return q_val

# %%
def sample_uniform_starting_pts():

    alpha_plus_low = model_config_rswg['RSWG']['param_bounds'][0][0]
    alpha_plus_high = model_config_rswg['RSWG']['param_bounds'][1][0]

    alpha_minus_low = model_config_rswg['RSWG']['param_bounds'][0][1]
    alpha_minus_high = model_config_rswg['RSWG']['param_bounds'][1][1]

    beta_low = model_config_rswg['RSWG']['param_bounds'][0][2]
    beta_high = model_config_rswg['RSWG']['param_bounds'][1][2]

    alpha_plus = random.uniform(alpha_plus_low, alpha_plus_high)

    alpha_minus = random.uniform(alpha_minus_low, alpha_minus_high)

    beta = random.uniform(beta_low, beta_high)

    starting_pts = [alpha_plus, alpha_minus, beta]
    
    return starting_pts

# %%
def RSWG(params, actions, rewards, tr):

    a_p = params[0]
    a_n = params[1]
    beta = params[2]
    
    # subj data - 200 * 2
    n_trials = 20
    subj_ll = 0
    q_RL = np.zeros((1,5))
    
    # for tr in np.arange(n_trials):
    q_RL = np.zeros((1,5))
    sample_ll = 0
    for i in range(500):
        state = 0
        a_t = np.argmax(actions[tr][i])
        reward = rewards[tr][i]
        # import ipdb; ipdb.set_trace()

        like = softmax(q_RL[state, :], beta)[a_t]
        ll = np.log(like)
        sample_ll += ll

        if (reward - q_RL[state, a_t]) >= 0:
            alpha = a_p
            q_RL[state, a_t] = q_RL[state, a_t] + alpha * (reward - q_RL[state, a_t])
        else:
            alpha = a_n
            q_RL[state, a_t] = q_RL[state, a_t] + alpha * (reward - q_RL[state, a_t])
    sample_ll/=500
    subj_ll += sample_ll
    #print("=> ", params, subj_ll)
    return -subj_ll

def RSWG_offline(params, actions, rewards):

    a_p = params[0]
    a_n = params[1]
    beta = params[2]
    
    # subj data - 200 * 2
    n_trials = actions.shape[0]
    subj_ll = 0
    q_RL = np.zeros((1,5))
    
    for tr in np.arange(n_trials):
        q_RL = np.zeros((1,5))
        sample_ll = 0
        state = 0
        a_t = np.argmax(actions[tr])
        reward = rewards[tr]
        # import ipdb; ipdb.set_trace()

        like = softmax(q_RL[state, :], beta)[a_t]
        ll = np.log(like)
        subj_ll += ll

        if (reward - q_RL[state, a_t]) >= 0:
            alpha = a_p
            q_RL[state, a_t] = q_RL[state, a_t] + alpha * (reward - q_RL[state, a_t])
        else:
            alpha = a_n
            q_RL[state, a_t] = q_RL[state, a_t] + alpha * (reward - q_RL[state, a_t])
    #print("=> ", params, subj_ll)
    return -subj_ll

if __name__ == '__main__':
    # %%
    # Load pickled data


    datasets = ['online_DPT_meta_info_sine.npz','offline_DPT_meta_info_sine.npz']
    eval = {'offline': 1, 'online': 0}
    trs = list(set(np.random.randint(0, 200, size=25)))
    best_parameters = {'online':[], 'offline':[]}
    for tr in trs:
        for key, value in eval.items():
            dataset = np.load(datasets[value])
            n_restarts_r1 = 5

            # refer to model config and set the bounds
            lower_bound, upper_bound = model_config_rswg['RSWG']['param_bounds']
            bnds = tuple(zip(lower_bound, upper_bound))

            n_r = n_restarts_r1
            best_negLL = np.inf

            if key == 'offline':
                with open('./datasets/trajs_bandit_envs100000_hists1_samples1_H500_d5_var0.3_cov0.0_test.pkl', 'rb') as f:
                    data = pickle.load(f)
                data  = data[:200]
                context_actions = np.zeros((200, 500, 5))
                context_rewards = np.zeros((200, 500))
                for i in range(200):
                    context_actions[i][:500] = data[i]['context_actions']
                    context_actions[i][499] = dataset['context_actions'][i]

                    context_rewards[i][:500] = data[i]['context_rewards']
                    context_rewards[i][499] = dataset['context_rewards'][i]

                args = (context_actions, context_rewards, tr)
            else:
                args = (dataset['context_actions'], dataset['context_rewards'], tr)

            while n_r >= 0:
                print("\tn_r = %d " % (n_r) )
                x0 = sample_uniform_starting_pts()
                res = minimize(RSWG, x0, args=args, method='Nelder-Mead', bounds=bnds, options={'disp': True, 'maxiter': 5000})
                m_negLL = RSWG(res.x, args[0], args[1], args[2])

                if m_negLL == np.inf:
                    print("np.inf")
                else:
                    n_r -= 1

                if m_negLL < best_negLL:
                    print("\t\tbest_negLL: ", best_negLL, "   | best_x: ", res.x.copy(), "   | m_negLL: ", m_negLL)
                    best_x = res.x.copy()
                    best_negLL = m_negLL

            best_parameters[key].append(best_x)
            print(f'{key} DPT RSWG parameters')
            print("recoverd_param after R 1: ", best_x)
    
    print(best_parameters)
    import json
    with open('RW.json', 'w') as file:
        json.dump(best_parameters, file)



