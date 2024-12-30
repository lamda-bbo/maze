import numpy as np
from baselines.common.runners import AbstractEnvRunner
from collections import deque

MAX_ENT = -np.log(1/6)

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, partner=None, agent_population=None, partner_population=None,
                 ent_pool_coef=0.0, ent_version=3, history_len=10, div_version=0):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.agent_population = agent_population
        self.partner_population = partner_population
        self.partner = partner
        self.ent_pool_coef = ent_pool_coef
        self.entropy_pop_delta_history = deque(maxlen=history_len)
        self.entropy_pop_new_history = deque(maxlen=history_len)
        self.neg_logp_pop_new_history = deque(maxlen=history_len)
        self.neg_logp_pop_delta_history = deque(maxlen=history_len)
        self.entropy_pop_delta_mean = 0.0
        self.entropy_pop_new_mean = 0.0
        self.neg_logp_pop_new_mean = 0.0
        self.neg_logp_pop_delta_mean = 0.0
        self.ent_version = ent_version
        self.div_version = div_version
        print(f'self.ent_version {self.ent_version} self.ent_pool_coef {self.ent_pool_coef}')

    def run(self):
        # Here, we init the lists that will contain the mb and partner of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states # state of the model, not the env (=None)
        partner_obs, partner_rewards, partner_actions, partner_values, partner_dones, partner_neglogpacs = [], [], [], [], [], []
        epinfos = []
        # For n in range number of steps

        import time
        tot_time = time.time()
        int_time = 0
        num_envs = len(self.curr_state)
        # False
        if self.env.trajectory_sp:
            # Selecting which environments should run fully in self play
            sp_envs_bools = np.random.random(num_envs) < self.env.self_play_randomization
            print("SP envs: {}/{}".format(sum(sp_envs_bools), num_envs))

        other_agent_simulation_time = 0

        from overcooked_ai_py.mdp.actions import Action

        def other_agent_action():
            if self.env.use_action_method:
                other_agent_actions = self.env.other_agent.actions(self.curr_state, self.other_agent_idx)
                return [Action.ACTION_TO_INDEX[a] for a in other_agent_actions]
            else:
                other_agent_actions = self.env.other_agent.direct_policy(self.obs1)
                return other_agent_actions

        def entropy(action_probs, eps=1e-4):
            """
            action_probs shape: (num_examples, num_classes)
            output shape: (num_examples)
            """
            assert action_probs.shape[1] == 6, 'action_probs.shape[1] == 6'
            neg_p_logp = - action_probs * np.log(action_probs)
            entropy = np.sum(neg_p_logp, axis=1)
            # assert np.max(entropy) <= MAX_ENT+1e5, 'entropy_max <= MAX_ENT'
            return entropy

        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            overcooked = 'env_name' in self.env.__dict__.keys() and self.env.env_name == "Overcooked-v0"
            if overcooked:
                if self.agent_population:
                    pop_len = len(self.agent_population)
                    action_probs_np = np.zeros((pop_len, self.obs0.shape[0], 6)) ## 6 is the action_dim
                    actions_np = np.zeros((pop_len, self.obs0.shape[0]))
                    agent_i = 0
                    agent_entropy = []
                    for agent in self.agent_population:
                        actions, _, _, _, action_probs = agent.model.step(self.obs0, S=self.states, M=self.dones)
                        actions_np[agent_i] = actions.copy()
                        action_probs_np[agent_i] = action_probs.copy()
                        agent_entropy.append(entropy(action_probs))
                        agent_i += 1
                    action_probs_pop_np = np.mean(action_probs_np, axis=0)
                    # entropy_pop = entropy(action_probs_pop_np)

                actions_agent, values_agent, self.states, neglogpacs_agent, action_probs_agent0 = self.model.step(self.obs0, S=self.states, M=self.dones)

                if self.agent_population:
                    action_probs_np_new = np.append(action_probs_np, np.expand_dims(action_probs_agent0, axis=0), axis=0)
                    action_probs_pop_np_new = np.mean(action_probs_np_new, axis=0)
                    entropy_pop = entropy(action_probs_pop_np)    # average policy except current agent
                    entropy_pop_new = entropy(action_probs_pop_np_new) # average policy including current agent
                    entropy_pop_delta = entropy_pop_new - entropy_pop

                    sampled_action_prob_pop_np = np.take(action_probs_pop_np, actions_agent)
                    neg_logp_pop = - np.log(sampled_action_prob_pop_np)
                    sampled_action_prob_pop_np_new = np.take(action_probs_pop_np_new, actions_agent)
                    neg_logp_pop_new = - np.log(sampled_action_prob_pop_np_new)
                    neg_logp_pop_delta = neg_logp_pop_new - neg_logp_pop

                import time
                current_simulation_time = time.time()

                # jump over this if part, only look at else part
                # Randomize at either the trajectory level or the individual timestep level
                if self.env.trajectory_sp:

                    # If there are environments selected to not run in SP, generate actions
                    # for the other agent, otherwise we skip this step.
                    if sum(sp_envs_bools) != num_envs:
                        other_agent_actions_bc = other_agent_action()

                    # If there are environments selected to run in SP, generate self-play actions
                    if sum(sp_envs_bools) != 0:
                        other_agent_actions_sp, _, _, _ = self.model.step(self.obs1, S=self.states, M=self.dones)

                    # Select other agent actions for each environment depending on whether it was selected
                    # for self play or not
                    other_agent_actions = []
                    for i in range(num_envs):
                        if sp_envs_bools[i]:
                            sp_action = other_agent_actions_sp[i]
                            other_agent_actions.append(sp_action)
                        else:
                            bc_action = other_agent_actions_bc[i]
                            other_agent_actions.append(bc_action)
                
                else:

                    if self.env.self_play_randomization < 1:
                        # Get actions through the action method of the agent

                        if self.partner_population is not None:
                            pop_len = len(self.partner_population)
                            action_probs_np_partner = np.zeros((pop_len, self.obs0.shape[0], 6))  ## 6 is the action_dim
                            actions_np_partner = np.zeros((pop_len, self.obs0.shape[0]))
                            partner_i = 0
                            partner_entropy = []
                            for p in self.partner_population:
                                actions, _, _, _, action_probs = p.model.step(self.obs1, S=self.states, M=self.dones)
                                actions_np_partner[partner_i] = actions.copy()
                                action_probs_np_partner[partner_i] = action_probs.copy()
                                partner_entropy.append(entropy(action_probs))
                                partner_i += 1
                            action_probs_pop_np_partner = np.mean(action_probs_np_partner, axis=0)
                            # entropy_pop = entropy(action_probs_pop_np)

                        actions_partner, values_partner, self.states_partner, neglogpacs_partner, action_probs_agent1 = self.partner.model.step(self.obs1,
                                                                                                        S=self.states,
                                                                                                        M=self.dones)

                        if self.partner_population is not None:
                            action_probs_np_new_partner = np.append(action_probs_np_partner,
                                                            np.expand_dims(action_probs_agent1, axis=0), axis=0)
                            action_probs_pop_np_new_partner = np.mean(action_probs_np_new_partner, axis=0)
                            entropy_pop_partner = entropy(action_probs_pop_np_partner)  # average policy except current agent
                            entropy_pop_new_partner = entropy(action_probs_pop_np_new_partner)  # average policy including current agent
                            entropy_pop_delta_partner = entropy_pop_new_partner - entropy_pop_partner

                            sampled_action_prob_pop_np_partner = np.take(action_probs_pop_np_partner, actions_partner)
                            neg_logp_pop_partner = - np.log(sampled_action_prob_pop_np_partner)
                            sampled_action_prob_pop_np_new_partner = np.take(action_probs_pop_np_new_partner, actions_partner)
                            neg_logp_pop_new_partner = - np.log(sampled_action_prob_pop_np_new_partner)
                            neg_logp_pop_delta_partner = neg_logp_pop_new_partner - neg_logp_pop_partner

                    # Naive non-parallelized way of getting actions for other
                    if self.env.self_play_randomization > 0:
                        self_play_actions, _, _, _ = self.model.step(self.obs1, S=self.states, M=self.dones)
                        self_play_bools = np.random.random(num_envs) < self.env.self_play_randomization

                        for i in range(num_envs):
                            is_self_play_action = self_play_bools[i]
                            if is_self_play_action:
                                other_agent_actions[i] = self_play_actions[i]

                # NOTE: This has been discontinued as now using .other_agent_true takes about the same amount of time
                # elif self.env.other_agent_bc:
                #     # Parallelise actions with direct action, using the featurization function
                #     featurized_states = [self.env.mdp.featurize_state(s, self.env.mlp) for s in self.curr_state]
                #     player_featurizes_states = [s[idx] for s, idx in zip(featurized_states, self.other_agent_idx)]
                #     other_agent_actions = self.env.other_agent.direct_policy(player_featurizes_states, sampled=True, no_wait=True)

                other_agent_simulation_time += time.time() - current_simulation_time

                joint_action = [(actions_agent[i], actions_partner[i]) for i in range(len(actions))]

                mb_obs.append(self.obs0.copy())
                partner_obs.append(self.obs1.copy())
            else:
                actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
                mb_obs.append(self.obs.copy())

            mb_actions.append(actions_agent)
            mb_values.append(values_agent)
            mb_neglogpacs.append(neglogpacs_agent)
            mb_dones.append(self.dones)

            partner_actions.append(actions_partner)
            partner_values.append(values_partner)
            partner_neglogpacs.append(neglogpacs_partner)
            partner_dones.append(self.dones.copy())

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            if overcooked:
                obs, rewards, self.dones, infos = self.env.step(joint_action)
                rewards_np = np.array(rewards)
                rewards_partner = rewards_np.copy()
                if self.agent_population:
                    if self.ent_version == 3:
                        # 0: no diversity
                        # 1: mep-sample diversity
                        # 2: JSD diversity
                        if self.div_version == 0:
                            rewards_np = rewards_np
                            rewards_partner = rewards_partner
                        elif self.div_version == 1:
                            rewards_np = rewards_np + self.ent_pool_coef * neg_logp_pop_new
                            rewards_partner = rewards_partner + self.ent_pool_coef * neg_logp_pop_new_partner  # here check again
                        elif self.div_version == 2:
                            rewards_np = rewards_np + self.ent_pool_coef * (entropy_pop_new - np.mean(agent_entropy, axis=0)/(pop_len+1))    # here check again
                            rewards_partner = rewards_partner + self.ent_pool_coef * (
                                        entropy_pop_new_partner - np.mean(partner_entropy, axis=0) / (
                                            pop_len + 1))  # here check again
                        else:
                            print(f"div_version {self.div_version} is unknown.")
                    else:
                        print(f"ent_version {self.ent_version} is unknown.")
                        exit()

                    self.entropy_pop_delta_history.append(np.mean(entropy_pop_delta))  
                    self.entropy_pop_new_history.append(np.mean(entropy_pop_new))  
                    self.neg_logp_pop_new_history.append(np.mean(neg_logp_pop_new))
                    self.neg_logp_pop_delta_history.append(np.mean(neg_logp_pop_delta))

                    self.entropy_pop_delta_mean = np.mean(self.entropy_pop_delta_history)
                    self.entropy_pop_new_mean = np.mean(self.entropy_pop_new_history)
                    self.neg_logp_pop_new_mean = np.mean(self.neg_logp_pop_new_history)
                    self.neg_logp_pop_delta_mean = np.mean(self.neg_logp_pop_delta_history)

                rewards = rewards_np.tolist()
                rewards_partner = rewards_partner.tolist()
                both_obs = obs["both_agent_obs"]
                self.obs0[:] = both_obs[:, 0, :, :]
                self.obs1[:] = both_obs[:, 1, :, :]
                self.curr_state = obs["overcooked_state"]
                self.other_agent_idx = obs["other_agent_env_idx"]
            else:
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            partner_rewards.append(rewards_partner)

        print("Other agent actions took", other_agent_simulation_time, "seconds")
        tot_time = time.time() - tot_time
        print("Total simulation time for {} steps: {} \t Other agent action time: {} \t {} steps/s".format(self.nsteps, tot_time, int_time, self.nsteps / tot_time))
        
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs0, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)  # Q_function_value   # mb_values: state_function_value
        mb_advs = np.zeros_like(mb_rewards) # adv_function_value
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        # batch of steps to batch of rollouts
        partner_obs = np.asarray(partner_obs, dtype=self.obs.dtype)
        partner_rewards = np.asarray(partner_rewards, dtype=np.float32)
        partner_actions = np.asarray(partner_actions)
        partner_values = np.asarray(partner_values, dtype=np.float32)
        partner_neglogpacs = np.asarray(partner_neglogpacs, dtype=np.float32)
        partner_dones = np.asarray(partner_dones, dtype=np.bool)
        last_values_partner = self.partner.model.value(self.obs1, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        partner_returns = np.zeros_like(partner_rewards)  # Q_function_value   # mb_values: state_function_value
        partner_advs = np.zeros_like(partner_rewards)  # adv_function_value
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values_partner
            else:
                nextnonterminal = 1.0 - partner_dones[t + 1]
                nextvalues = partner_values[t + 1]
            delta = partner_rewards[t] + self.gamma * nextvalues * nextnonterminal - partner_values[t]
            partner_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        partner_returns = partner_advs + partner_values
        partner_epinfos = epinfos.copy()
        result_1 = (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), mb_states, epinfos)
        result_2 = (*map(sf01, (partner_obs, partner_returns, partner_dones, partner_actions, partner_values, partner_neglogpacs)), mb_states)
        # return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
        #     mb_states, epinfos), (*map(sf01, (partner_obs, partner_returns, partner_dones, partner_actions, partner_values, partner_neglogpacs)),
        #     mb_states, epinfos)
        return result_1, result_2

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
