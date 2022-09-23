import numpy as np
import torch
from src import logger
from src.chac.representer import RepresentationNetwork
from src.chac.experience_buffer import ExperienceBuffer
from src.chac.actor import Actor
from src.chac.critic import Critic
from src.chac.forward_model import ForwardModel

import copy


class Layer:
    def __init__(self, level, env, agent_params):
        self.level = level
        self.n_levels = agent_params["n_levels"]
        self.time_scale = agent_params["time_scales"][level]
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]
        self.fw = agent_params["fw"]

        # Set time limit for each layer. If agent uses only 1 layer, time limit
        # is the max number of low-level actions allowed in the episode
        # (i.e, env.max_actions) to ensure that the policies that are learned
        # are limited in length.
        if self.n_levels > 1:
            self.time_limit = self.time_scale
        else:
            self.time_limit = env.max_actions

        self.current_state = None
        self.goal = None

        self.random_action_perc = agent_params["random_action_perc"]

        # Ceiling on buffer size
        self.buffer_size_ceiling = 10**7

        # Number of full episodes stored in replay buffer
        self.episodes_to_store = agent_params["buffer_size"]

        # Set number of transitions to serve as replay goals during goal replay
        self.num_replay_goals = 2

        # Number of the transitions created for each attempt (i.e, action replay + goal replay + subgoal testing)
        if self.level == 0:
            self.trans_per_attempt = (
                1 + self.num_replay_goals
            ) * self.time_limit
        else:
            self.trans_per_attempt = (
                1 + self.num_replay_goals
            ) * self.time_limit + int(self.time_limit * self.subgoal_test_perc)

        # Buffer size = transitions per attempt * # attempts per episode * num of episodes stored
        self.buffer_size = min(
            self.trans_per_attempt
            * self.time_limit ** (self.n_levels - 1 - self.level)
            * self.episodes_to_store,
            self.buffer_size_ceiling,
        )

        act_dim = goal_dim = env.subgoal_dim
        if self.level == 0:
            # Actions of lowest layer are real actions in environment
            act_dim = env.action_dim
        if self.level == self.n_levels - 1:
            # Goals of highest layer are real goals of environment
            goal_dim = env.end_goal_dim

        self.replay_buffer = ExperienceBuffer(
            self.buffer_size,
            agent_params["batch_size"],
            env.state_dim,
            act_dim,
            goal_dim,
        )

        # Create buffer to store not yet finalized goal replay transitions
        self.temp_goal_replay_storage = []

        logger.info(
            "\nHierarchy Level {} with time scale {}".format(
                self.level, self.time_scale
            )
        )
        # Initialize networks
        self.actor = Actor(env, self.level, self.n_levels)
        logger.info(self.actor)
        # Initialize networks

        self.critic = Critic(env, self.level, self.n_levels, self.time_scale)
        logger.info(self.critic)

        if self.fw:
            self.state_predictor = ForwardModel(
                env, self.level, agent_params["fw_params"], self.buffer_size
            )
            logger.info(self.state_predictor)

        # Parameter determines degree of noise added to actions during training
        self.noise_perc = (
            agent_params["atomic_noise"]
            if self.level == 0
            else agent_params["subgoal_noise"]
        )

        # Create flag to indicate when layer has ran out of attempts to achieve goal.
        # This will be important for subgoal testing
        self.maxed_out = False
        self.subgoal_penalty = agent_params["subgoal_penalties"][level]
        self.surprise_history = []
        self.q_values = []

        # regularization
        self.reg = agent_params["reg"]
        self.old_subgoals = []
        # self.subgoals_range = env.subgoal_bounds
        self.phi_interval = 100
        self.stable_coeff = 0.001

        # prioritize sampling
        self.high_ratio = 0.5
        self.low_ratio = 0.3
        self.candidate_idxs = np.array(
            [i for i in range(self.buffer_size // self.time_limit)]
        )

        # goal_array stores goal for each layer of agent.
        self.goal_array = [None] * self.n_levels
        self.current_state = None
        self.steps_taken = 0
        self.total_steps = 0
        self.abs_range = 20

        self.representer = RepresentationNetwork(
            env, 2, self.abs_range, env.subgoal_dim
        )
        self.representer_old = copy.deepcopy(self.representer)

    def sample_prioritized_data(self, states_array):
        """TODO: add documentations"""
        episode_num = self.replay_buffer.size

        p = states_array[: self.time_limit + 1, :]
        p_argsorted = np.argsort(p.reshape(-1))

        high_p = p_argsorted[-int(len(p_argsorted) * self.high_ratio):]
        low_p = p_argsorted[int(len(p_argsorted) * self.low_ratio):]

        random_idx = np.random.randint(len(high_p), size=self.buffer_size)
        selected = high_p[random_idx]

        random_idx_new = np.random.randint(len(low_p), size=self.buffer_size)
        selected_new = low_p[random_idx_new]
        cur_candidate_idxs = self.candidate_idxs[
            : episode_num * (self.time_limit + 1)
        ]
        selected_idx_new = cur_candidate_idxs[selected_new]
        t_samples_new = selected_idx_new[0]

        reg_states = states_array[t_samples_new, :]
        selected_idx = cur_candidate_idxs[selected]

        t_samples = selected_idx[0]

        high_states = states_array[t_samples:]
        high_states_next = states_array[t_samples:]

        states = high_states
        next_state = states_array[t_samples:]
        train_data = np.array(
            [states, next_state, high_states, high_states_next]
        )
        return train_data, selected_idx, reg_states

    # Add noise to provided action
    def add_noise(self, action, env):
        # Noise added will be percentage of range
        action_bounds = (
            env.action_bounds
            if self.level == 0
            else env.subgoal_bounds_symmetric
        )
        action_offset = (
            env.action_offset if self.level == 0 else env.subgoal_bounds_offset
        )

        assert len(action) == len(
            action_bounds
        ), "Action bounds must have same dimension as action"
        assert len(action) == len(
            self.noise_perc
        ), "Noise percentage vector must have same dimension as action"

        # Add noise to action and ensure remains within bounds$
        for i in range(len(action)):
            action[i] += np.random.normal(
                0, self.noise_perc[i] * action_bounds[i]
            )
            action[i] = max(
                min(action[i], action_bounds[i] + action_offset[i]),
                -action_bounds[i] + action_offset[i],
            )

        return action

    def get_random_action(self, env):
        action = np.zeros(
            (env.action_dim if self.level == 0 else env.subgoal_dim)
        )

        # Each dimension of random action should take some value in the dimension's range
        for i in range(len(action)):
            if self.level == 0:
                action[i] = np.random.uniform(
                    -env.action_bounds[i] + env.action_offset[i],
                    env.action_bounds[i] + env.action_offset[i],
                )
            else:
                action[i] = np.random.uniform(
                    env.subgoal_bounds[i][0], env.subgoal_bounds[i][1]
                )

        return action

    # Function selects action using an epsilon-greedy policy
    @torch.no_grad()
    def choose_action(self, agent, env, subgoal_test):
        current_state_tensor = torch.FloatTensor(self.current_state).view(1, -1)
        goal_tensor = torch.FloatTensor(self.goal).view(1, -1)
        # If testing mode or testing subgoals, action is output of actor network without noise
        if agent.test_mode or subgoal_test:
            action = (
                self.actor(current_state_tensor, goal_tensor)[0].cpu().numpy()
            )
            action_type = "Policy"
            next_subgoal_test = subgoal_test
        else:
            if np.random.random_sample() > self.random_action_perc:
                action = self.add_noise(
                    self.actor(current_state_tensor, goal_tensor)[0]
                    .cpu()
                    .numpy(),
                    env,
                )
                action_type = "Noisy Policy"
            else:
                action = self.get_random_action(env)
                action_type = "Random"

            # Determine whether to test upcoming subgoal to penalize the subgoals that weren't achieved
            if np.random.random_sample() < self.subgoal_test_perc:
                next_subgoal_test = True
            else:
                next_subgoal_test = False

        return action, action_type, next_subgoal_test

    def perform_action_replay(self, hindsight_action, next_state, goal_status):
        """Create action replay transition by evaluating hindsight action given original goal
        Determine reward (0 if goal achieved, -1 otherwise) and finished boolean"""
        # The finished boolean is used for determining the target for Q-value updates
        finished = goal_status[self.level]
        reward = 0.0 if finished else -1.0
        # Transition will take the form [old state, hindsight_action, reward, next_state, goal, terminate boolean]
        transition = np.array(
            [
                self.current_state,
                hindsight_action,
                reward,
                next_state,
                self.goal,
                finished,
                None,
            ],
            dtype=object,
        )
        self.replay_buffer.add(transition)

    def create_prelim_goal_replay_trans(
        self, hindsight_action, next_state, env
    ):
        """Create initial goal replay transitions
        Create transition evaluating hindsight action for some goal to be determined in future.
        Goal will be ultimately be selected from states layer has traversed through.
        Transition will be in the form [old state, hindsight action, reward = None, next state,
            goal = None, finished = None, next state projeted to subgoal/end goal space]"""

        if self.level == self.n_levels - 1:
            hindsight_goal = env.project_state_to_end_goal(next_state)
        else:
            hindsight_goal = env.project_state_to_sub_goal(next_state)

        transition = np.array(
            [
                self.current_state,
                hindsight_action,
                None,
                next_state,
                None,
                None,
                hindsight_goal,
            ],
            dtype=object,
        )
        self.temp_goal_replay_storage.append(transition)

    def get_reward(self, new_goal, hindsight_goal, goal_thresholds):
        """Returns 0 if all distances are smaller than the goal thresholds else -1"""
        assert (
            len(new_goal) == len(hindsight_goal) == len(goal_thresholds)
        ), "Goal, hindsight goal, and goal thresholds do not have same dimensions"
        achieved = np.all(
            np.absolute(new_goal - hindsight_goal) < goal_thresholds, axis=-1
        )
        return achieved - 1.0

    def finalize_goal_replay(self, goal_thresholds):
        """Finalize goal replay by filling in goal, reward, and finished boolean
        for the preliminary goal replay transitions created before"""
        # Choose transitions to serve as goals during goal replay.  The last transition will always be used
        num_trans = len(self.temp_goal_replay_storage)
        if num_trans == 0:
            return None
        num_replay_goals = self.num_replay_goals
        # If fewer transitions in the ordinary number of replay goals, lower number of replay goals
        if num_trans < self.num_replay_goals:
            num_replay_goals = num_trans

        indices = np.zeros((num_replay_goals))
        indices[: num_replay_goals - 1] = np.random.randint(
            num_trans, size=num_replay_goals - 1
        )
        indices[num_replay_goals - 1] = num_trans - 1
        indices = np.sort(indices)

        # For each selected transition, update the goal dimension of the selected transition and all prior transitions
        # by using the next state of the selected transition as the new goal.
        # Given new goal, update the reward and finished boolean as well.
        for i in range(len(indices)):
            trans_copy = np.copy(self.temp_goal_replay_storage)

            new_goal = trans_copy[int(indices[i])][6]
            for index in range(num_trans):
                # Update goal to new goal
                trans_copy[index][4] = new_goal

                # Update reward
                trans_copy[index][2] = self.get_reward(
                    new_goal, trans_copy[index][6], goal_thresholds
                )

                # Update finished boolean based on reward
                if trans_copy[index][2] == 0:
                    trans_copy[index][5] = True
                else:
                    trans_copy[index][5] = False

                self.replay_buffer.add(trans_copy[index])

        # Clear storage for preliminary goal replay transitions at end of goal replay
        self.temp_goal_replay_storage = []

    def penalize_subgoal(self, subgoal, next_state, test_fail=True):
        """Create transition penalizing subgoal if necessary. The target Q-value when this transition is used will ignore
        next state as the finished boolean = True. Change the finished boolean to False, if you would like the subgoal
        penalty to depend on the next state."""
        transition = np.array(
            [
                self.current_state,
                subgoal,
                self.subgoal_penalty if test_fail else 0.0,
                next_state,
                self.goal,
                True,
                None,
            ],
            dtype=object,
        )
        self.replay_buffer.add(transition)

    # Determine whether layer is finished training
    def return_to_higher_level(
        self, max_lay_achieved, agent, env, attempts_made
    ):
        """
        Return to higher level if
        (i) a higher level goal has been reached,
        (ii) maxed out episode time steps (env.max_actions)
        (iii) not testing and layer is out of attempts, and
        (iv) testing, layer is not the highest level, and layer is out of attempts.
        -----------------------------------------------------------------------------------
        NOTE: during testing, highest level will continue to output subgoals until either
        (i) the maximum number of episode time steps or (ii) the end goal has been achieved.
        """

        assert (
            env.step_ctr == agent.steps_taken
        ), "Step counter of env and agent should be equal"
        # Return to previous level when any higher level goal achieved.
        # NOTE: if not testing and agent achieves end goal, training will continue until
        # out of time (i.e., out of time steps or highest level runs out of attempts).
        # This will allow agent to experience being around the end goal.
        if max_lay_achieved is not None and max_lay_achieved >= self.level:
            return True

        # Return when out of time
        elif env.step_ctr >= env.max_actions:
            return True

        # Return when layer has maxed out attempts
        elif not agent.test_mode and attempts_made >= self.time_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to achieve goal
        elif (
            agent.test_mode
            and self.level < self.n_levels - 1
            and attempts_made >= self.time_limit
        ):
            return True

        else:
            return False

    def train(
        self, agent, env, subgoal_test=False, episode_num=None, eval_data={}
    ):
        """Learn to achieve goals with actions belonging to appropriate time scale.
        "goal_array" contains the goal states for the current layer and all higher layers"""
        train_test_prefix = (
            "test_{}/".format(self.level)
            if agent.test_mode
            else "train_{}/".format(self.level)
        )
        if self.level > 0:
            if "{}subgoal_succ".format(train_test_prefix) not in eval_data:
                eval_data["{}subgoal_succ".format(train_test_prefix)] = []
            if "{}n_subgoals".format(train_test_prefix) not in eval_data:
                eval_data["{}n_subgoals".format(train_test_prefix)] = 0

        if "{}q".format(train_test_prefix) not in eval_data:
            eval_data["{}q".format(train_test_prefix)] = []

        # Set layer's current state and new goal state
        self.goal = agent.goal_array[self.level]
        self.current_state = agent.current_state

        # Reset flag indicating whether layer has ran out of attempts.  This will be used for subgoal testing.
        self.maxed_out = False

        # Display all subgoals if visualizing training and current layer is bottom layer
        if self.level == 0 and agent.env.visualize and self.n_levels > 1:
            env.display_subgoals(agent.goal_array)

        # Current layer has self.time_limit attempts to each its goal state.
        attempts_made = 0

        # to update the subgaol
        counter = 0
        while True:
            # Select action to achieve goal state using epsilon-greedy policy or greedy policy if in test mode
            action, action_type, next_subgoal_test = self.choose_action(
                agent, env, subgoal_test
            )
            current_state_tensor = torch.FloatTensor(self.current_state).view(
                1, -1
            )
            goal_tensor = torch.FloatTensor(self.goal).view(1, -1)
            action_tensor = torch.FloatTensor(action).view(1, -1)
            with torch.no_grad():
                q_val = self.critic(
                    current_state_tensor, goal_tensor, action_tensor
                )
            eval_data["{}q".format(train_test_prefix)] += [q_val[0].item()]

            if agent.env.graph and not self.fw and agent.env.visualize:
                self.q_values += [q_val[0].item()]

            # If next layer is not bottom level, propose subgoal for next layer to achieve and determine
            # whether that subgoal should be tested
            if self.level > 0:
                # make sure we already have added transitions to the buffer
                if self.reg:
                    subgoal = (
                        self.representer(current_state_tensor)[0]
                        .detach()
                        .numpy()
                    )
                    # if any value is nan choose the regularisaton
                    if np.isnan(subgoal).any():
                        print(
                            f"The representer network produced NaNs, choose the non regularised version"
                        )
                        agent.goal_array[self.level - 1] = action
                    else:
                        agent.goal_array[self.level - 1] = subgoal
                else:
                    agent.goal_array[self.level - 1] = action


                goal_status, eval_data, max_lay_achieved = agent.layers[
                    self.level - 1
                ].train(agent, env, next_subgoal_test, episode_num, eval_data)

                eval_data["{}subgoal_succ".format(train_test_prefix)] += [
                    1.0 if goal_status[self.level - 1] else 0.0
                ]
                eval_data["{}n_subgoals".format(train_test_prefix)] += 1

            # If layer is bottom level, execute low-level action
            else:
                # move to next state
                agent.current_state = env.execute_action(action)
                agent.steps_taken += 1

                if agent.verbose and env.step_ctr >= env.max_actions:
                    print("Out of actions (Steps: %d)" % env.step_ctr)

                # Determine whether any of the goals from any layer was achieved
                # and, if applicable, the highest layer whose goal was achieved
                goal_status, max_lay_achieved = agent.check_goals(env)

            attempts_made += 1

            if (
                agent.env.graph
                and self.fw
                and agent.env.visualize
                and self.state_predictor.err_list
            ):
                agent_state_tensor = torch.FloatTensor(self.current_state).view(
                    1, -1
                )
                surprise = self.state_predictor.pred_bonus(
                    action_tensor, current_state_tensor, agent_state_tensor
                )
                self.surprise_history += surprise.tolist()

            # Print if goal from current layer has been achieved
            if agent.verbose and goal_status[self.level]:

                if self.level < self.n_levels - 1:
                    print("SUBGOAL ACHIEVED")

                print(
                    "\nEpisode %d, Layer %d, Attempt %d Goal Achieved"
                    % (episode_num, self.level, attempts_made)
                )
                print("Goal: ", self.goal)

                if self.level == self.n_levels - 1:
                    print(
                        "Hindsight Goal: ",
                        env.project_state_to_end_goal(agent.current_state),
                    )
                else:
                    print(
                        "Hindsight Goal: ",
                        env.project_state_to_sub_goal(agent.current_state),
                    )

            # Perform hindsight learning using action actually executed (low-level action or hindsight subgoal)
            if self.level == 0:
                hindsight_action = action
            else:
                # If subgoal action was achieved by layer below, use this as hindsight action
                if goal_status[self.level - 1]:
                    hindsight_action = action
                # Otherwise, use subgoal that was achieved in hindsight
                else:
                    hindsight_action = env.project_state_to_sub_goal(
                        agent.current_state
                    )

            # Next, create hindsight transitions if not testing
            if not agent.test_mode:
                # Create action replay transition by evaluating hindsight action given current goal
                self.perform_action_replay(
                    hindsight_action, agent.current_state, goal_status
                )

                # Create preliminary goal replay transitions.  The goal and reward in these transitions will be
                # finalized when this layer has run out of attempts or the goal has been achieved.
                self.create_prelim_goal_replay_trans(
                    hindsight_action, agent.current_state, env
                )

                # Penalize subgoals if subgoal testing and subgoal was missed by lower layers after maximum number of attempts
                test_fail = agent.layers[self.level - 1].maxed_out
                if self.level > 0 and next_subgoal_test and test_fail:
                    self.penalize_subgoal(
                        action, agent.current_state, test_fail
                    )

            # Print summary of transition
            if agent.verbose:
                print(
                    "\nEpisode %d, Level %d, Attempt %d"
                    % (episode_num, self.level, attempts_made)
                )
                print("Old State: ", self.current_state)
                print("Hindsight Action: ", hindsight_action)
                print("Original Action: ", action)
                print("Next State: ", agent.current_state)
                print("Goal: ", self.goal)

                if self.level == self.n_levels - 1:
                    print(
                        "Hindsight Goal: ",
                        env.project_state_to_end_goal(agent.current_state),
                    )
                else:
                    print(
                        "Hindsight Goal: ",
                        env.project_state_to_sub_goal(agent.current_state),
                    )

                print("Goal Status: ", goal_status, "\n")
                print("All Goals: ", agent.goal_array)

            # Update state of current layer
            self.current_state = agent.current_state

            if (
                (
                    max_lay_achieved is not None
                    and max_lay_achieved >= self.level
                )
                or env.step_ctr >= env.max_actions
                or attempts_made >= self.time_limit
            ):

                if agent.verbose and self.level == self.n_levels - 1:
                    print("HL Attempts Made: ", attempts_made)

                # If goal was not achieved after max number of attempts, set maxed out flag to true
                if (
                    attempts_made >= self.time_limit
                    and not goal_status[self.level]
                ):
                    self.maxed_out = True

                # If not testing, finish goal replay by filling in missing goal and reward values before returning to
                # prior level.
                if not agent.test_mode:
                    if self.level == self.n_levels - 1:
                        goal_thresholds = env.end_goal_thresholds
                    else:
                        goal_thresholds = env.sub_goal_thresholds

                    self.finalize_goal_replay(goal_thresholds)

                # Under certain circumstances, the highest layer will not seek a new end goal
                if self.return_to_higher_level(
                    max_lay_achieved, agent, env, attempts_made
                ):
                    return goal_status, eval_data, max_lay_achieved

    def learn(self, num_updates, counter):
        """Update networks for num_updates"""
        representation_loss = 0
        learn_history = {}
        learn_history["reward"] = []
        if self.fw:
            learn_history["fw_bonus"] = []
            learn_history["fw_loss"] = []

        learn_summary = {}

        if self.replay_buffer.size <= 250:
            return learn_summary

        self.critic.train()
        self.actor.train()
        if self.fw:
            self.state_predictor.train()
        
        if self.reg:
            self.representer.train()

        for __ in range(num_updates):
            (
                old_states,
                actions,
                rewards,
                new_states,
                goals,
                done,
            ) = self.replay_buffer.get_batch()

            # use forward model to update reward with curiosity
            if self.fw:
                bonus = self.state_predictor.pred_bonus(
                    actions, old_states, new_states
                ).unsqueeze(1)
                eta = self.state_predictor.eta
                rewards = rewards * eta + (1 - eta) * bonus
                learn_history["fw_bonus"].append(bonus.mean().item())


            learn_history["reward"] += rewards.cpu().numpy().tolist()


            if self.reg and self.level > 0:
                states_array = old_states.numpy()
                train_data, __, reg_states = self.sample_prioritized_data(
                    states_array
                )
                reg_states_tensor = torch.tensor(
                    reg_states, dtype=torch.float32
                ).to("cpu")
                # If regularization is enabled
                reg_feature_old = self.representer_old(reg_states_tensor)
                reg_feature_new = self.representer(reg_states_tensor)

                stable_loss = (
                    (reg_feature_new - reg_feature_old).pow(2).mean()
                )

                # prioritize sampling
                train_data = torch.tensor(
                    train_data, dtype=torch.float32
                ).to("cpu")

                state, next_state = self.representer(
                    train_data[0]
                ), self.representer(train_data[1])

                min_dist = torch.clamp(
                    (state - next_state).pow(2).mean(dim=1), min=0.0
                )

                high_state, high_next_state = self.representer(
                    train_data[2]
                ), self.representer(train_data[3])

                max_dist = torch.clamp(
                    1 - (high_state - high_next_state).pow(2).mean(dim=1),
                    min=0.0,
                )

                ini_representation_loss = (min_dist + max_dist).mean()

                # if counter > self.phi_interval:
                #     representation_loss = (
                #         stable_loss * self.stable_coeff
                #         + ini_representation_loss 
                #     )
                #     counter = 0
                # else:
                #     representation_loss = ini_representation_loss 
                #     counter += 1
                q_update = self.critic.update(
                    old_states,
                    actions,
                    rewards,
                    new_states,
                    goals,
                    self.representer(new_states).detach(),
                    done,
                )
            else:
                q_update = self.critic.update(
                    old_states,
                    actions,
                    rewards,
                    new_states,
                    goals,
                    self.actor(new_states, goals).detach(),
                    done,
                )

            for k, v in q_update.items():
                if k not in learn_history.keys():
                    learn_history[k] = []
                learn_history[k].append(v)
            if self.reg and self.level > 0:
                mu_loss = -self.critic(
                    old_states, goals, self.representer(old_states)
                ).mean()
                if counter > self.phi_interval:
                    representation_loss = (
                        stable_loss * self.stable_coeff
                        + mu_loss 
                    )
                    counter = 0
                else:
                    representation_loss = mu_loss 
                    counter += 1
                mu_update = self.representer.update(representation_loss)
            else:
                mu_loss = -self.critic(
                    old_states, goals, self.actor(old_states, goals)
                ).mean()
                mu_update = self.actor.update(mu_loss)

            for k, v in mu_update.items():
                if k not in learn_history.keys():
                    learn_history[k] = []
                learn_history[k].append(v)

            if self.fw:
                learn_history["fw_loss"].append(
                    self.state_predictor.update(old_states, actions, new_states)
                )

        r_vals = [-0.0, -1.0]

        if self.level != 0:
            r_vals.append(float(-self.time_scale))

        for reward_val in r_vals:
            learn_history["reward_{}_frac".format(reward_val)] = float(
                np.sum(np.isclose(learn_history["reward"], reward_val))
            ) / len(learn_history["reward"])

        for k, v in learn_history.items():
            learn_summary[k] = (
                v.mean() if isinstance(v, torch.Tensor) else np.mean(v)
            )

        return learn_summary
