import gym
import numpy as np
import torch
from scipy import signal

from envs.base_env import BaseEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(dim, H, var, type='uniform'):
    if type == 'uniform':
        means = np.random.uniform(0, 1, dim)
    elif type == 'bernoulli':
        means = np.random.beta(1, 1, dim)
    else:
        raise NotImplementedError
    env = BanditEnv(means, H, var=var, type=type)
    return env


def sample_linear(arms, H, var):
    lin_d = arms.shape[1]
    theta = np.random.normal(0, 1, lin_d) / np.sqrt(lin_d)
    env = LinearBanditEnv(theta, arms, H, var=var)
    return env


class BanditEnv(BaseEnv):
    def __init__(self, means, H, var=0.0, type='uniform'):
        opt_a_index = np.argmax(means)
        self.initial_means = means.copy()
        self.means = means
        self.opt_a_index = opt_a_index
        self.opt_a = np.zeros(means.shape)
        self.opt_a[opt_a_index] = 1.0
        self.dim = len(means)
        self.observation_space = gym.spaces.Box(low=1, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.dim,))
        self.state = np.array([1])
        self.processs_var = 0.005
        self.var = var
        self.dx = 1
        self.du = self.dim
        self.topk = False
        self.type = type

        # some naming issue here
        self.H_context = H
        # possibly wrong as this is the length of an episode
        # which means it is usually set to 1
        self.H = 1

    def get_arm_value(self, u):
        return np.sum(self.means * u)

    def reset(self):
        self.current_step = 0
        self.means = self.initial_means.copy()
        return self.state

    def transit(self, x, u):
        a = np.argmax(u)

        ## Reward 1: Restless Bandit
        self.means = self.means + np.random.normal(0, self.processs_var, self.dim)

        # update to calculate the optimal arm by ground truth mean
        self.opt_a_index = np.argmax(self.means)
        self.opt_a = np.zeros(self.means.shape)
        self.opt_a[self.opt_a_index] = 1.0

        gt_mean = self.means[a]
        r = gt_mean + np.random.normal(0, self.var)

        ## Reward 2: 
        # r = self.means[a] + np.random.normal(0, self.var)

        ## Reward 3
        # random_step = np.random.randint(0, 500) 
        # mu = 0.5 * np.sin(2 * np.pi * random_step / 500)
        # r = self.means[a] + np.random.normal(mu, self.var)
            
        return self.state.copy(), r

    def step(self, action):
        if self.current_step >= self.H:
            raise ValueError("Episode has already ended")

        _, r = self.transit(self.state, action)
        self.current_step += 1
        done = (self.current_step >= self.H)

        return self.state.copy(), r, done, {}

    def deploy_eval(self, ctrl):
        # No variance during evaluation
        tmp = self.var
        self.var = 0.0
        res = self.deploy(ctrl)
        self.var = tmp
        return res


class BanditEnvVec(BaseEnv):
    """
    Vectorized bandit environment.
    """
    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self.dx = envs[0].dx
        self.du = envs[0].du

    def reset(self):
        return [env.reset() for env in self._envs]

    def step(self, actions):
        next_obs, rews, dones = [], [], []
        for action, env in zip(actions, self._envs):
            next_ob, rew, done, _ = env.step(action)
            next_obs.append(next_ob)
            rews.append(rew)
            dones.append(done)
        return next_obs, rews, dones, {}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def envs(self):
        return self._envs

    def deploy_eval(self, ctrl):
        # evaluating through the end of episodes of all environments here
        single_step = False
        # No variance during evaluation
        tmp = [env.var for env in self._envs]
        for env in self._envs:
            env.var = 0.0
        res = self.deploy(ctrl, single_step)
        for env, var in zip(self._envs, tmp):
            env.var = var
        return res

    def deploy(self, ctrl, single_step):
        x = self.reset()

        xs = []
        xps = []
        us = []
        rs = []
        done = False

        while not done:
            u = ctrl.act_numpy_vec(x)

            xs.append(x)
            us.append(u)

            x, r, done, _ = self.step(u)
            done = all(done)

            rs.append(r)
            xps.append(x)

        xs = np.concatenate(xs)
        us = np.concatenate(us)
        xps = np.concatenate(xps)
        rs = np.concatenate(rs)
        
        return xs, us, xps, rs

    def get_arm_value(self, us):
        values = [np.sum(env.means * u) for env, u in zip(self._envs, us)]
        return np.array(values)




class LinearBanditEnv(BanditEnv):
    def __init__(self, theta, arms, H, var=0.0):
        self.theta = theta
        self.arms = arms
        self.means = arms @ theta
        self.opt_a_index = np.argmax(self.means)
        self.opt_a = np.zeros(self.means.shape)
        self.opt_a[self.opt_a_index] = 1.0
        self.dim = len(self.means)
        self.observation_space = gym.spaces.Box(low=1, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.dim,))
        self.state = np.array([1])
        self.var = var
        self.dx = 1
        self.du = self.dim
        
        self.H_context = H
        self.H = 1

    def get_arm_value(self, u):
        return np.sum(self.means * u)

    def reset(self):
        self.current_step = 0
        return self.state

    def transit(self, x, u):
        a = np.argmax(u)
        r = self.means[a] + np.random.normal(0, self.var)
        return self.state.copy(), r

    def step(self, action):
        if self.current_step >= self.H:
            raise ValueError("Episode has already ended")

        _, r = self.transit(self.state, action)
        self.current_step += 1
        done = (self.current_step >= self.H)

        return self.state.copy(), r, done, {}