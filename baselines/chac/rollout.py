import numpy as np
import time
import sys
import pickle
from baselines.chac.utils import store_args
from tqdm import tqdm
from collections import deque

class Rollout:
    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            use_target_net (boolean): whether or not to use the target net for rollouts
            noise_eps (float): scale of the additive Gaussian noise
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """

        self.policy = policy
        self.dims = dims
        self.logger = logger
        self.render = render
        self.T = T
        self.rollout_batch_size = rollout_batch_size

        self.policy_action_params = kwargs['policy_action_params']

        self.envs = [make_env() for _ in range(rollout_batch_size)]
        self.first_env = self.envs[0]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.custom_histories = []

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self, return_states=False):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        if return_states:
            mj_states = [[] for _ in range(self.rollout_batch_size)]

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # hold custom histories through out the iterations
        other_histories = []

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        for t in range(self.T):
            if return_states:
                for i in range(self.rollout_batch_size):
                    mj_states[i].append(self.envs[i].env.sim.get_state())

            if self.policy_action_params:
                policy_output = self.policy.get_actions(o, ag, self.g, **self.policy_action_params)
            else:
                policy_output = self.policy.get_actions(o, ag, self.g)

            if isinstance(policy_output, np.ndarray):
                u = policy_output  # get the actions from the policy output since actions should be the first element
            else:
                u = policy_output[0]
                other_histories.append(policy_output[1:])
            try:
                if u.ndim == 1:
                    # The non-batched case should still have a reasonable shape.
                    u = u.reshape(1, -1)
            except:
                self.logger.warn('Action "u" is not a Numpy array.')
            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except Exception as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        if return_states:
            for i in range(self.rollout_batch_size):
                mj_states[i].append(self.envs[i].env.sim.get_state())

        self.initial_o[:] = o
        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if other_histories:
            for history_index in range(len(other_histories[0])):
                self.custom_histories.append(deque(maxlen=self.history_len))
                self.custom_histories[history_index].append([x[history_index] for x in other_histories])
        self.n_episodes += self.rollout_batch_size

        if return_states:
            ret = convert_episode_to_batch_major(episode), mj_states
        else:
            ret = convert_episode_to_batch_major(episode)
        return ret

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.custom_histories.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)

    def generate_rollouts_update(self, n_cycles, n_batches):
        updated_policy = self.policy
        dur_total = 0
        dur_ro = 0
        dur_train = 0
        time_durations = (dur_total, dur_ro, dur_train)
        return updated_policy, time_durations

    def logs(self, prefix='worker'):
        raise NotImplemented
class RolloutWorker(Rollout):
    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
        history_len=100, render=False, **kwargs):
        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size,
                history_len=history_len, render=render, **kwargs)

        self.env = self.policy.env
        self.env.visualize = render
        self.env.graph = kwargs['graph']
        self.time_scales = np.array([int(t) for t in kwargs['time_scales'].split(',')])
        self.eval_data = {}

    def train_policy(self, n_train_rollouts, n_train_batches):
        dur_train = 0
        dur_ro = 0

        for episode in tqdm(range(n_train_rollouts), file=sys.__stdout__, desc='Train Rollout'):
            ro_start = time.time()
            success, self.eval_data, train_duration = self.policy.train(self.env, episode, self.eval_data, n_train_batches)
            dur_train += train_duration
            self.success_history.append(1.0 if success else 0.0)
            self.n_episodes += 1
            dur_ro += time.time() - ro_start - train_duration

        return dur_train, dur_ro

    def generate_rollouts_update(self, n_train_rollouts, n_train_batches):
        dur_start = time.time()
        self.policy.set_train_mode()
        dur_train, dur_ro = self.train_policy(n_train_rollouts, n_train_batches)
        dur_total = time.time() - dur_start
        time_durations = (dur_total, dur_ro, dur_train)
        updated_policy = self.policy
        return updated_policy, time_durations

    def generate_rollouts(self, return_states=False):
        self.reset_all_rollouts()
        self.policy.set_test_mode()
        success, self.eval_data, _ = self.policy.train(self.env, self.n_episodes, self.eval_data, None)
        self.success_history.append(1.0 if success else 0.0)
        self.n_episodes += 1
        return self.eval_data

    def logs(self, prefix=''):
        eval_data = self.eval_data

        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('episodes', self.n_episodes)]

        # Get metrics for all layers of the hierarchy
        for i in range(self.policy.n_levels):
            layer_prefix = '{}_{}/'.format(prefix, i)

            subg_succ_prefix = '{}subgoal_succ'.format(layer_prefix)
            if subg_succ_prefix in eval_data.keys():
                if len(eval_data[subg_succ_prefix]) > 0:
                    logs += [(subg_succ_prefix + '_rate',
                              np.mean(eval_data[subg_succ_prefix]))]
                else:
                    logs += [(subg_succ_prefix + '_rate', 0.0)]

            for postfix in ["n_subgoals", "fw_loss", "fw_bonus", "reward", "q_loss", "q_grads",
                    "q_grads_std", "target_q", "next_q", "current_q", "mu_loss", "mu_grads",
                    "mu_grads_std", "reward_-0.0_frac", "reward_-1.0_frac",
                    "reward_-{}.0_frac".format(self.time_scales[i])]:
                metric_key = "{}{}".format(layer_prefix, postfix)
                if metric_key in eval_data.keys():
                    logs += [(metric_key, eval_data[metric_key])]

            q_prefix = "{}q".format(layer_prefix)
            if q_prefix in eval_data.keys():
                if len(eval_data[q_prefix]) > 0:
                    logs += [("{}avg_q".format(layer_prefix), np.mean(eval_data[q_prefix]))]
                else:
                    logs += [("{}avg_q".format(layer_prefix), 0.0)]

        if prefix != '' and not prefix.endswith('/'):
            new_logs = []
            for key, val in logs:
                if not key.startswith(prefix):
                    new_logs += [((prefix + '/' + key, val))]
                else:
                    new_logs += [(key, val)]
            logs = new_logs

        return logs

    def clear_history(self):
        self.success_history.clear()
        self.custom_histories.clear()
        if hasattr(self, 'eval_data'):
            self.eval_data.clear()
