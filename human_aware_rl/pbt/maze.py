import copy
import os
import time
import sys
import random
import itertools
import numpy as np
import pandas as pd
import heapq
import tensorflow.compat.v1 as tf
import gym
from collections import defaultdict
from memory_profiler import profile
from tensorflow.python.saved_model import simple_save
from datetime import datetime
from sklearn.cluster import KMeans

from sacred import Experiment
from sacred.observers import FileStorageObserver

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

import logz
from overcooked_ai_py.utils import (
    profile,
    load_pickle,
    save_pickle,
    save_dict_to_file,
    load_dict_from_file,
)
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair

from human_aware_rl.utils import (
    create_dir_if_not_exists,
    delete_dir_if_exists,
    reset_tf,
    set_global_seed,
)
from human_aware_rl.baselines_utils import (
    create_model,
    get_vectorized_gym_env,
    my_update_model,
    get_agent_from_model,
    save_baselines_model,
    overwrite_model,
    load_baselines_model,
    LinearAnnealer,
)

PBT_DATA_DIR = "MAZE/"
ex = Experiment("MAZE")
ex.observers.append(FileStorageObserver.create(PBT_DATA_DIR + "pbt_exps"))


class PBTAgent(object):
    """An agent that can be saved and loaded and all and the main data it contains is the self.model

    Goal is to be able to pass in save_locations or PBTAgents to workers that will load such agents
    and train them together.
    """

    def __init__(
        self, agent_name, start_params, start_logs=None, model=None, gym_env=None
    ):
        self.params = start_params
        self.logs = (
            start_logs
            if start_logs is not None
            else {
                "agent_name": agent_name,
                "avg_rew_per_step": [],
                "params_hist": defaultdict(list),
                "num_ppo_runs": 0,
                "reward_shaping": [],
            }
        )
        self.logs["agent_name"] = (
            agent_name if agent_name is not None else self.logs["agent_name"]
        )
        with tf.device("/device:GPU:{}".format(self.params["GPU_ID"])):
            self.model = (
                model
                if model is not None
                else create_model(gym_env, agent_name, **start_params)
            )

    @property
    def num_ppo_runs(self):
        return self.logs["num_ppo_runs"]

    @property
    def agent_name(self):
        return self.logs["agent_name"]

    def get_agent(self):
        return get_agent_from_model(self.model, self.params["sim_threads"])

    def update(
        self,
        gym_env,
        partner=None,
        agent_population=None,
        partner_population=None,
        ent_version=1,
        div_version=0,
        add_ent_to_loss=1,
        train_experience=None,
    ):
        with tf.device("/device:GPU:{}".format(self.params["GPU_ID"])):

            train_info, experience = my_update_model(
                gym_env,
                self.model,
                partner,
                agent_population,
                partner_population,
                ent_version,
                div_version,
                add_ent_to_loss,
                train_experience=train_experience,
                **self.params,
            )

            for k, v in train_info.items():
                if k not in self.logs.keys():
                    self.logs[k] = []
                self.logs[k].extend(v)
            self.logs["num_ppo_runs"] += 1
            return experience

    def update_avg_rew_per_step_logs(self, avg_rew_per_step_stats):
        self.logs["avg_rew_per_step"] = avg_rew_per_step_stats

    def save(self, save_folder):
        """Save agent model, logs, and parameters"""
        create_dir_if_not_exists(save_folder)
        save_baselines_model(self.model, save_folder)
        save_dict_to_file(dict(self.logs), save_folder + "logs")
        save_dict_to_file(self.params, save_folder + "params")

    @staticmethod
    def from_dir(load_folder, agent_name):
        logs = load_dict_from_file(load_folder + "logs.txt")
        params = load_dict_from_file(load_folder + "params.txt")
        model = load_baselines_model(load_folder[0:-1], agent_name, params)
        return PBTAgent(agent_name, params, start_logs=logs, model=model)

    @staticmethod
    def update_from_files(file0, file1, gym_env, save_dir):
        pbt_agent0 = PBTAgent.from_dir(file0)
        pbt_agent1 = PBTAgent.from_dir(file1)
        gym_env.other_agent = pbt_agent1
        pbt_agent0.update(gym_env)
        return pbt_agent0

    def save_predictor(self, save_folder):
        """Saves easy-to-load simple_save tensorflow predictor for agent"""
        tf.compat.v1.saved_model.simple_save(
            tf.get_default_session(),
            save_folder,
            inputs={"obs": self.model.act_model.X},
            outputs={
                "action": self.model.act_model.action,
                "value": self.model.act_model.vf,
                "action_probs": self.model.act_model.action_probs,
            },
        )

    def update_pbt_iter_logs(self):
        for k, v in self.params.items():
            self.logs["params_hist"][k].append(v)
        self.logs["params_hist"] = dict(self.logs["params_hist"])

    def explore_from(self, best_training_agent):
        a = overwrite_model(best_training_agent.model, self.model)
        self.logs["num_ppo_runs"] = best_training_agent.num_ppo_runs
        self.params = best_training_agent.params.copy()
        return a

    def deep_copy(self, agent_name, gym_env=None):
        agent_copy = PBTAgent(
            agent_name, self.params.copy(), self.logs.copy(), gym_env=gym_env
        )
        overwrite_model(self.model, agent_copy.model)


@ex.config
def my_config():

    ##################
    # GENERAL PARAMS #
    ##################

    # my newly added params
    # 0: no diversity
    # 1: mep diversity
    # 2: JSD diversity
    DIV_VERSION = 2

    # if add entropy to the loss function (as PPO2 baseline does)
    ADD_ENT_TO_LOSS = True

    # iteration iterval to repair
    PAIR_INTERVAL = 5

    # pair method
    # 0: random 1: max 2:min 3:mcc
    PAIR_MODE = 0

    # archive size
    ARCHIVE_SIZE = 5

    POPULATION_SIZE = 2

    TIMESTAMP_DIR = True
    EX_NAME = "pbt_random0"

    # Network params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3

    now = datetime.now()  # current date and time

    if TIMESTAMP_DIR:
        SAVE_DIR = (
            PBT_DATA_DIR
            + now.strftime("%Y_%m_%d-%H_%M_%S_")
            + EX_NAME
            + "_div"
            + str(DIV_VERSION)
            + "_entloss"
            + str(ADD_ENT_TO_LOSS)
            + "_inter"
            + str(PAIR_INTERVAL)
            + "_mode"
            + str(PAIR_MODE)
            + "_net"
            + str(SIZE_HIDDEN_LAYERS)
            + "/"
        )
    else:
        SAVE_DIR = (
            PBT_DATA_DIR
            + EX_NAME
            + "_div"
            + str(DIV_VERSION)
            + "_entloss"
            + str(ADD_ENT_TO_LOSS)
            + "/"
        )

    print("Saving data to ", SAVE_DIR)

    RUN_TYPE = "pbt"

    # Reduce parameters to be able to run locally to test for simple bugs
    LOCAL_TESTING = False

    # GPU id to use
    GPU_ID = 0

    # List of seeds to run
    SEEDS = [9015]

    # Number of parallel environments used for simulating rollouts
    sim_threads = 50 if not LOCAL_TESTING else 2

    ##############
    # PBT PARAMS #
    ##############

    TOTAL_STEPS_PER_AGENT = (
        1.1e7 if not LOCAL_TESTING else 1e4
    )  # We keep this parameters same across all the methods

    ITER_PER_SELECTION = POPULATION_SIZE  # How many pairings and model training updates before the worst model is overwritten

    RESAMPLE_PROB = 0.33
    MUTATION_FACTORS = [0.75, 1.25]
    HYPERPARAMS_TO_MUTATE = [
        "LAM",
        "CLIPPING",
        "LR",
        "STEPS_PER_UPDATE",
        "ENTROPY",
        "VF_COEF",
    ]

    NUM_SELECTION_GAMES = 2 if not LOCAL_TESTING else 2

    ##############
    # PPO PARAMS #
    ##############

    # Total environment timesteps for the PPO run
    PPO_RUN_TOT_TIMESTEPS = 40000 if not LOCAL_TESTING else 1000
    NUM_PBT_ITER = int(
        TOTAL_STEPS_PER_AGENT
        * POPULATION_SIZE
        // (ITER_PER_SELECTION * PPO_RUN_TOT_TIMESTEPS)
    )  # Numer of main loops

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    TOTAL_BATCH_SIZE = 10000 if not LOCAL_TESTING else 1000

    # Number of minibatches we divide up each batch into before
    # performing gradient steps
    MINIBATCHES = 5 if not LOCAL_TESTING else 1

    BATCH_SIZE = TOTAL_BATCH_SIZE // sim_threads

    # Number of gradient steps to perform on each mini-batch
    STEPS_PER_UPDATE = 8 if not LOCAL_TESTING else 1

    # Learning rate
    LR = 8e-4

    # Entropy bonus coefficient
    ENTROPY = 0.5

    # Entropy bonus coefficient for the model pool
    ENTROPY_POOL = 0.04

    # Version of calculating the entropy
    ENT_VERSION = 3

    # Paths of the member agents in the model pool
    LOAD_FOLDER_LST = ""

    # Value function coefficient
    VF_COEF = 0.5

    # Gamma discounting factor
    GAMMA = 0.99

    # Lambda advantage discounting factor
    LAM = 0.98

    # Max gradient norm
    MAX_GRAD_NORM = 0.1

    # PPO clipping factor
    CLIPPING = 0.05

    # 0 is default value that does no annealing
    REW_SHAPING_HORIZON = 5e6

    ##################
    # NETWORK PARAMS #
    ##################

    # Network type used
    NETWORK_TYPE = "conv_and_mlp"

    ##################
    # MDP/ENV PARAMS #
    ##################

    # Mdp params
    layout_name = "random0"
    start_order_list = None

    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0.015,
        "POT_DISTANCE_REW": 0.03,
        "SOUP_DISTANCE_REW": 0.1,
    }

    # Env params
    horizon = 400

    #########
    # OTHER #
    #########

    # For non fixed MDPs
    mdp_generation_params = {
        "padded_mdp_shape": (11, 7),
        "mdp_shape_fn": ([5, 11], [5, 7]),
        "prop_empty_fn": [0.6, 1],
        "prop_feats_fn": [0, 0.6],
    }

    # Approximate info stats
    GRAD_UPDATES_PER_AGENT = (
        STEPS_PER_UPDATE
        * MINIBATCHES
        * (PPO_RUN_TOT_TIMESTEPS // TOTAL_BATCH_SIZE)
        * ITER_PER_SELECTION
        * NUM_PBT_ITER
        // POPULATION_SIZE
    )

    print("Total steps per agent", TOTAL_STEPS_PER_AGENT)
    print("Grad updates per agent", GRAD_UPDATES_PER_AGENT)

    params = {
        "LOCAL_TESTING": LOCAL_TESTING,
        "RUN_TYPE": RUN_TYPE,
        "EX_NAME": EX_NAME,
        "SAVE_DIR": SAVE_DIR,
        "GPU_ID": GPU_ID,
        "mdp_params": {
            "layout_name": layout_name,
            "start_order_list": start_order_list,
            "rew_shaping_params": rew_shaping_params,
        },
        "env_params": {"horizon": horizon},
        "PPO_RUN_TOT_TIMESTEPS": PPO_RUN_TOT_TIMESTEPS,
        "NUM_PBT_ITER": NUM_PBT_ITER,
        "ITER_PER_SELECTION": ITER_PER_SELECTION,
        "POPULATION_SIZE": POPULATION_SIZE,
        "RESAMPLE_PROB": RESAMPLE_PROB,
        "MUTATION_FACTORS": MUTATION_FACTORS,
        "mdp_generation_params": mdp_generation_params,  # NOTE: currently not used
        "HYPERPARAMS_TO_MUTATE": HYPERPARAMS_TO_MUTATE,
        "REW_SHAPING_HORIZON": REW_SHAPING_HORIZON,
        "ENTROPY": ENTROPY,
        "ENTROPY_POOL": ENTROPY_POOL,
        "ENT_VERSION": ENT_VERSION,
        "LOAD_FOLDER_LST": LOAD_FOLDER_LST.split(":"),
        "GAMMA": GAMMA,
        "sim_threads": sim_threads,
        "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR": LR,
        "VF_COEF": VF_COEF,
        "STEPS_PER_UPDATE": STEPS_PER_UPDATE,
        "MINIBATCHES": MINIBATCHES,
        "CLIPPING": CLIPPING,
        "LAM": LAM,
        "NETWORK_TYPE": NETWORK_TYPE,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "SEEDS": SEEDS,
        "NUM_SELECTION_GAMES": NUM_SELECTION_GAMES,
        "total_steps_per_agent": TOTAL_STEPS_PER_AGENT,
        "grad_updates_per_agent": GRAD_UPDATES_PER_AGENT,
        "DIV_VERSION": DIV_VERSION,
        "ADD_ENT_TO_LOSS": ADD_ENT_TO_LOSS,
        "PAIR_INTERVAL": PAIR_INTERVAL,
        "PAIR_MODE": PAIR_MODE,
        "ARCHIVE_SIZE": ARCHIVE_SIZE,
    }


@ex.named_config
def fixed_mdp():
    LOCAL_TESTING = False
    layout_name = "simple"

    sim_threads = 30 if not LOCAL_TESTING else 2
    PPO_RUN_TOT_TIMESTEPS = 36000 if not LOCAL_TESTING else 1000
    TOTAL_BATCH_SIZE = 12000 if not LOCAL_TESTING else 1000

    STEPS_PER_UPDATE = 5
    MINIBATCHES = 6 if not LOCAL_TESTING else 2

    LR = 5e-4


@ex.named_config
def fixed_mdp_rnd_init():
    # NOTE: Deprecated
    LOCAL_TESTING = False
    fixed_mdp = True
    layout_name = "scenario2"

    sim_threads = 10 if LOCAL_TESTING else 50
    PPO_RUN_TOT_TIMESTEPS = 24000
    TOTAL_BATCH_SIZE = 8000

    STEPS_PER_UPDATE = 4
    MINIBATCHES = 4

    LR = 5e-4


@ex.named_config
def padded_all_scenario():
    # NOTE: Deprecated
    LOCAL_TESTING = False
    fixed_mdp = ["scenario2", "simple", "schelling_s", "unident_s"]
    PADDED_MDP_SHAPE = (10, 5)

    sim_threads = 10 if LOCAL_TESTING else 60
    PPO_RUN_TOT_TIMESTEPS = 40000 if not LOCAL_TESTING else 1000
    TOTAL_BATCH_SIZE = 20000 if not LOCAL_TESTING else 1000

    STEPS_PER_UPDATE = 8
    MINIBATCHES = 4

    LR = 5e-4
    REW_SHAPING_HORIZON = 1e7


def pbt_one_run(params, seed):
    print(params)
    # Iterating noptepochs over same batch data but shuffled differently
    # dividing each batch in `nminibatches` and doing a gradient step for each one
    create_dir_if_not_exists(params["SAVE_DIR"])  # 'PBT_DATA_DIR/pbt_simple/seed_9015/'
    save_dict_to_file(
        params, params["SAVE_DIR"] + "config"
    )  # 'PBT_DATA_DIR/pbt_simple/seed_9015/config'

    logz.configure_output_dir(params["SAVE_DIR"])

    mdp = OvercookedGridworld.from_layout_name(**params["mdp_params"])
    overcooked_env = OvercookedEnv(mdp, **params["env_params"])
    overcooked_env.reset()

    gym_env = get_vectorized_gym_env(
        overcooked_env,
        "Overcooked-v0",
        agent_idx=0,
        featurize_fn=lambda x: mdp.lossless_state_encoding(x),
        **params,
    )
    gym_env.update_reward_shaping_param(1.0)  # Start reward shaping from 1

    annealer = LinearAnnealer(horizon=params["REW_SHAPING_HORIZON"])

    # AGENT POPULATION INITIALIZATION
    population_size = params["POPULATION_SIZE"]
    assert params["ITER_PER_SELECTION"] == population_size
    agent_population = []  # always denote as 0
    partner_population = []  # always denote as 1
    partner_archive = []
    agent_names = ["agent" + str(i) for i in range(population_size)]
    partner_names = ["partner" + str(i) for i in range(population_size)]
    archive_partner_names = [
        "partner" + str(i)
        for i in range(population_size, population_size + params["ARCHIVE_SIZE"])
    ]
    for agent_name in agent_names:
        agent = PBTAgent(agent_name, params, gym_env=gym_env)
        print(f"Initialized {agent_name}")
        agent_population.append(agent)
    for partner_name in partner_names:
        partner = PBTAgent(partner_name, params, gym_env=gym_env)
        print(f"Initialized {partner_name}")
        partner_population.append(partner)
    for archive_partner_name in archive_partner_names:
        archive_partner = PBTAgent(archive_partner_name, params, gym_env=gym_env)
        partner_archive.append(archive_partner)

    def pbt_training():
        best_agent_sparse_rew_avg = [-np.Inf] * population_size
        best_partner_sparse_rew_avg = [-np.Inf] * population_size
        partner_idx = list(range(population_size))
        rewards_pd = []
        select_pd = []
        pair_pd = []
        copy_pd = []
        select_idx = [-1 for i in range(population_size)]
        num_partner_name = population_size + params["ARCHIVE_SIZE"]
        iter_begin = params["PAIR_INTERVAL"]
        iter_mid = (
            params["NUM_PBT_ITER"]
            // params["PAIR_INTERVAL"]
            // 2
            * params["PAIR_INTERVAL"]
        )
        iter_end = (
            params["NUM_PBT_ITER"] // params["PAIR_INTERVAL"] * params["PAIR_INTERVAL"]
        )

        print(f"Total iterations {params["NUM_PBT_ITER"]}")
        # Main training loop
        for pbt_iter in range(1, params["NUM_PBT_ITER"] + 1):

            pairs_to_train = list(zip(range(population_size), partner_idx))

            # Step 1: Update
            update_begin = time.time()
            for sel_iter in range(params["ITER_PER_SELECTION"]):

                pair_idx = np.random.choice(len(pairs_to_train))
                idx_agent, idx_partner = pairs_to_train.pop(pair_idx)
                pbt_agent, pbt_partner = (
                    agent_population[idx_agent],
                    partner_population[idx_partner],
                )
                # print(f"Training agent {idx_agent} with partner {idx_partner} fixed ")

                agent_env_steps = (
                    pbt_agent.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
                )
                reward_shaping_param = annealer.param_value(agent_env_steps)
                pbt_agent.logs["reward_shaping"].append(reward_shaping_param)
                gym_env.update_reward_shaping_param(reward_shaping_param)

                gym_env.other_agent = pbt_partner.get_agent()
                gym_env.venv.remote_set_agent_idx(0)
                agent_population_rest = agent_population.copy()
                agent_population_rest.pop(idx_agent)
                partner_population_rest = partner_population.copy()
                partner_population_rest.pop(idx_partner)
                if pbt_iter >= 4 * params["PAIR_INTERVAL"]:
                    experience = pbt_agent.update(
                        gym_env,
                        partner=pbt_partner,
                        agent_population=agent_population_rest,
                        partner_population=partner_population_rest,
                        ent_version=params["ENT_VERSION"],
                        div_version=params["DIV_VERSION"],
                        add_ent_to_loss=params["ADD_ENT_TO_LOSS"],
                    )

                    print(
                        f"Training partner {idx_partner} with agent {idx_agent} fixed "
                    )

                    agent_env_steps = (
                        pbt_partner.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
                    )
                    reward_shaping_param = annealer.param_value(agent_env_steps)
                    print(
                        "Current reward shaping:",
                        reward_shaping_param,
                        "\t Save_dir",
                        params["SAVE_DIR"],
                    )
                    pbt_partner.logs["reward_shaping"].append(reward_shaping_param)
                    gym_env.update_reward_shaping_param(reward_shaping_param)

                    pbt_partner.update(
                        gym_env,
                        ent_version=params["ENT_VERSION"],
                        div_version=params["DIV_VERSION"],
                        add_ent_to_loss=params["ADD_ENT_TO_LOSS"],
                        train_experience=experience,
                    )
            assert len(pairs_to_train) == 0
            update_end = time.time()

            # Step2: Selecetion
            # not do the selection at the beginning, stable the training
            if pbt_iter <= 3 * params["PAIR_INTERVAL"]:
                # pair wise reward of current partner population
                pair_reward_sparse = np.zeros((population_size, population_size))
                pair_reward_dense = np.zeros((population_size, population_size))
                pair_reward_per_step = np.zeros((population_size, population_size))
                # partner_idx[i] is the partner index of agent i

                # get pair_wise reward
                for i in range(population_size):
                    for j in range(population_size):
                        # Pairs each agent with itself in assessing generalization performance
                        print(f"Evaluating agent {i} and partner {j}")
                        agent_pair = AgentPair(
                            agent_population[i].get_agent(),
                            partner_population[j].get_agent(),
                        )
                        trajs = overcooked_env.get_rollouts(
                            agent_pair,
                            params["NUM_SELECTION_GAMES"],
                            reward_shaping=reward_shaping_param,
                        )
                        dense_rews, sparse_rews, lens = (
                            trajs["ep_returns"],
                            trajs["ep_returns_sparse"],
                            trajs["ep_lengths"],
                        )
                        pair_reward_sparse[i][j] = np.mean(sparse_rews)
                        pair_reward_dense[i][j] = np.mean(dense_rews)
                        pair_reward_per_step[i][j] = np.sum(dense_rews) / np.sum(lens)

                for i in range(population_size):
                    # Saving each agent model at the end of the pbt iteration
                    agent_population[i].update_pbt_iter_logs()

                    partner_population[i].update_pbt_iter_logs()

                    if (
                        (pbt_iter == iter_begin)
                        or (pbt_iter == iter_mid)
                        or (pbt_iter == iter_end)
                    ):
                        save_folder = (
                            params["SAVE_DIR"] + agent_population[i].agent_name + "/"
                        )
                        agent_population[i].save_predictor(
                            save_folder + "pbt_iter{}/".format(pbt_iter)
                        )
                        agent_population[i].save(
                            save_folder + "pbt_iter{}/".format(pbt_iter)
                        )

                    agent_population[i].update_avg_rew_per_step_logs(
                        np.mean(pair_reward_per_step[i])
                    )
                    avg_sparse_rew_agent = np.mean(pair_reward_sparse[i])
                    if avg_sparse_rew_agent > best_agent_sparse_rew_avg[i]:
                        best_agent_sparse_rew_avg[i] = avg_sparse_rew_agent
                        best_save_folder = (
                            params["SAVE_DIR"]
                            + agent_population[i].agent_name
                            + "/best/"
                        )
                        delete_dir_if_exists(best_save_folder, verbose=True)
                        agent_population[i].save_predictor(best_save_folder)
                        agent_population[i].save(best_save_folder)

                logz.log_tabular("Iteration", pbt_iter)
                for k in range(population_size):
                    logz.log_tabular(
                        f"Agent{k}SparseReward",
                        format(np.mean(pair_reward_sparse[k]), ".5f"),
                    )
                logz.log_tabular(
                    "MinSparseReward",
                    format(
                        np.min(
                            [
                                np.mean(pair_reward_sparse[i])
                                for i in range(population_size)
                            ]
                        ),
                        ".5f",
                    ),
                )
                logz.log_tabular(
                    "MaxSparseReward",
                    format(
                        np.max(
                            [
                                np.mean(pair_reward_sparse[i])
                                for i in range(population_size)
                            ]
                        ),
                        ".5f",
                    ),
                )
                logz.log_tabular(
                    "AverageSparseReward",
                    format(
                        np.mean(
                            [
                                np.mean(pair_reward_sparse[i])
                                for i in range(population_size)
                            ]
                        ),
                        ".5f",
                    ),
                )
                logz.log_tabular(
                    "StdSparseReward",
                    format(
                        np.std(
                            [
                                np.mean(pair_reward_sparse[i])
                                for i in range(population_size)
                            ]
                        ),
                        ".5f",
                    ),
                )
                logz.log_tabular("updateTime", format(update_end - update_begin, ".5f"))
                logz.log_tabular("selectTime", format(0, ".5f"))
                logz.log_tabular("pairTime", format(0, ".5f"))
                logz.dump_tabular()

                if pbt_iter == 3 * params["PAIR_INTERVAL"]:
                    if_copy = []
                    pair_reward_sparse_T = pair_reward_sparse.T
                    # add the first partmer into the archive directly
                    partner_archive[0].explore_from(partner_population[0])
                    archive_len = 1
                    pair_reward_sparse_archive_T = np.array([pair_reward_sparse_T[0]])
                    # add the rest partner
                    # get the initial thred
                    archive_distances = [0 for i in range(population_size)]
                    dists = np.zeros(
                        (population_size, population_size), dtype=np.float64
                    )
                    for i in range(population_size):
                        for j in range(population_size):
                            dists[i][j] = np.linalg.norm(
                                pair_reward_sparse_T[i] - pair_reward_sparse_T[j]
                            )
                        archive_distances[i] = np.sum(heapq.nsmallest(2, dists[i]))
                    thred_init = np.mean(archive_distances) / 2
                    for k in range(1, population_size):
                        if archive_len == 1:
                            if (
                                np.linalg.norm(
                                    pair_reward_sparse_T[k] - pair_reward_sparse_T[0]
                                )
                                > thred_init
                            ):
                                # add the k-th partner into the archive
                                partner_archive[1].explore_from(partner_population[k])
                                archive_len += 1
                                pair_reward_sparse_archive_T = np.vstack(
                                    (
                                        pair_reward_sparse_archive_T,
                                        pair_reward_sparse_T[k],
                                    )
                                )
                            else:
                                random_num = random.randint(0, 1)
                                if random_num == 0:
                                    # replace the 0-th partner in the archive with the k-th partner
                                    partner_archive[0].explore_from(
                                        partner_population[k]
                                    )
                                    pair_reward_sparse_archive_T[0] = (
                                        pair_reward_sparse_T[k]
                                    )
                        else:
                            # get the thred
                            nearest_distance = [0 for i in range(archive_len)]
                            archive_distances = [0 for i in range(archive_len)]
                            dists = np.zeros(
                                (archive_len, archive_len), dtype=np.float64
                            )
                            for i in range(archive_len):
                                for j in range(archive_len):
                                    dists[i][j] = np.linalg.norm(
                                        pair_reward_sparse_archive_T[i]
                                        - pair_reward_sparse_archive_T[j]
                                    )
                                archive_distances[i] = np.sum(
                                    heapq.nsmallest(2, dists[i])
                                )
                            thred = np.mean(archive_distances) / 2
                            print(f"the thred: {thred}")
                            for i in range(archive_len):
                                nearest_distance[i] = np.linalg.norm(
                                    pair_reward_sparse_archive_T[i]
                                    - pair_reward_sparse_T[k]
                                )
                            a = np.min(nearest_distance)
                            print(f"the nearest: {a}")
                            b = np.argmin(nearest_distance)
                            print(f"the index: {b}")
                            if a > thred:
                                iff = partner_archive[archive_len].explore_from(
                                    partner_population[k]
                                )
                                if_copy.append(iff)
                                archive_len += 1
                                pair_reward_sparse_archive_T = np.vstack(
                                    (
                                        pair_reward_sparse_archive_T,
                                        pair_reward_sparse_T[k],
                                    )
                                )
                            else:
                                random_num = random.randint(0, 1)
                                if random_num == 0:
                                    # replace the b-th partner in the archive with the k-th partner
                                    partner_archive[b].explore_from(
                                        partner_population[k]
                                    )
                                    pair_reward_sparse_archive_T[b] = (
                                        pair_reward_sparse_T[k]
                                    )

            if (
                pbt_iter > 3 * params["PAIR_INTERVAL"]
                and pbt_iter % params["PAIR_INTERVAL"] == 0
            ):
                if_copy = []
                # Get some reward statistics information
                pair_reward_sparse_archive = np.zeros((population_size, archive_len))
                pair_reward_dense_archive = np.zeros((population_size, archive_len))
                pair_reward_per_step_archive = np.zeros((population_size, archive_len))
                select_begin = time.time()
                for i in range(population_size):
                    for k in range(archive_len):
                        agent_pair = AgentPair(
                            agent_population[i].get_agent(),
                            partner_archive[k].get_agent(),
                        )
                        trajs = overcooked_env.get_rollouts(
                            agent_pair,
                            params["NUM_SELECTION_GAMES"],
                            reward_shaping=reward_shaping_param,
                        )
                        dense_rews, sparse_rews, lens = (
                            trajs["ep_returns"],
                            trajs["ep_returns_sparse"],
                            trajs["ep_lengths"],
                        )
                        pair_reward_sparse_archive[i][k] = np.mean(sparse_rews)
                        pair_reward_dense_archive[i][k] = np.mean(dense_rews)
                        pair_reward_per_step_archive[i][k] = np.sum(
                            dense_rews
                        ) / np.sum(lens)

                # pair wise reward of current partner population
                pair_reward_sparse = np.zeros((population_size, population_size))
                pair_reward_dense = np.zeros((population_size, population_size))
                pair_reward_per_step = np.zeros((population_size, population_size))
                # partner_idx[i] is the partner index of agent i
                agent_to_pair = list(range(population_size))
                # get pair_wise reward
                for i in range(population_size):
                    for j in range(population_size):
                        # Pairs each agent with itself in assessing generalization performance
                        print(f"Evaluating agent {i} and partner {j}")
                        agent_pair = AgentPair(
                            agent_population[i].get_agent(),
                            partner_population[j].get_agent(),
                        )
                        trajs = overcooked_env.get_rollouts(
                            agent_pair,
                            params["NUM_SELECTION_GAMES"],
                            reward_shaping=reward_shaping_param,
                        )
                        dense_rews, sparse_rews, lens = (
                            trajs["ep_returns"],
                            trajs["ep_returns_sparse"],
                            trajs["ep_lengths"],
                        )
                        pair_reward_sparse[i][j] = np.mean(sparse_rews)
                        pair_reward_dense[i][j] = np.mean(dense_rews)
                        pair_reward_per_step[i][j] = np.sum(dense_rews) / np.sum(lens)

                pair_reward_sparse_archive_T = pair_reward_sparse_archive.T
                pair_reward_sparse_T = pair_reward_sparse.T

                # get the initial thred
                archive_distances = [0 for i in range(population_size)]
                dists = np.zeros((population_size, population_size), dtype=np.float64)
                for i in range(population_size):
                    for j in range(population_size):
                        dists[i][j] = np.linalg.norm(
                            pair_reward_sparse_T[i] - pair_reward_sparse_T[j]
                        )
                    archive_distances[i] = np.sum(heapq.nsmallest(2, dists[i]))
                thred_init = np.mean(archive_distances) / 2

                for k in range(population_size):
                    if archive_len == 1:
                        if (
                            np.linalg.norm(
                                pair_reward_sparse_T[k] - pair_reward_sparse_T[0]
                            )
                            > thred_init
                        ):
                            # add the k-th partner into the archive
                            iff = partner_archive[1].explore_from(partner_population[k])
                            if_copy.append(iff)
                            archive_len += 1
                            pair_reward_sparse_archive_T = np.vstack(
                                (pair_reward_sparse_archive_T, pair_reward_sparse_T[k])
                            )
                        else:
                            random_num = random.randint(0, 1)
                            if random_num == 0:
                                # replace the 0-th partner in the archive with the k-th partner
                                iff = partner_archive[0].explore_from(
                                    partner_population[k]
                                )
                                if_copy.append(iff)
                                pair_reward_sparse_archive_T[0] = pair_reward_sparse_T[
                                    k
                                ]
                    else:
                        # get the threshold (thred)
                        nearest_distance = [0 for i in range(archive_len)]
                        archive_distances = [0 for i in range(archive_len)]
                        dists = np.zeros((archive_len, archive_len), dtype=np.float64)
                        for i in range(archive_len):
                            for j in range(archive_len):
                                dists[i][j] = np.linalg.norm(
                                    pair_reward_sparse_archive_T[i]
                                    - pair_reward_sparse_archive_T[j]
                                )
                            archive_distances[i] = np.sum(heapq.nsmallest(2, dists[i]))
                        thred = np.mean(archive_distances) / 2
                        print(f"the thred: {thred}")
                        for i in range(archive_len):
                            nearest_distance[i] = np.linalg.norm(
                                pair_reward_sparse_archive_T[i]
                                - pair_reward_sparse_T[k]
                            )
                        a = np.min(nearest_distance)
                        print(f"the nearest: {a}")
                        b = np.argmin(nearest_distance)
                        print(f"the index: {b}")
                        if a > thred:
                            if archive_len < params["ARCHIVE_SIZE"]:
                                iff = partner_archive[archive_len].explore_from(
                                    partner_population[k]
                                )
                                if_copy.append(iff)
                                archive_len += 1
                                pair_reward_sparse_archive_T = np.vstack(
                                    (
                                        pair_reward_sparse_archive_T,
                                        pair_reward_sparse_T[k],
                                    )
                                )
                            else:
                                # first in first out queue
                                new_partner = PBTAgent(
                                    "partner" + str(num_partner_name),
                                    params,
                                    gym_env=gym_env,
                                )
                                iff = new_partner.explore_from(partner_population[k])
                                if_copy.append(iff)
                                partner_archive.append(new_partner)
                                partner_archive.pop(0)
                                pair_reward_sparse_archive_T = np.vstack(
                                    (
                                        pair_reward_sparse_archive_T,
                                        pair_reward_sparse_T[k],
                                    )
                                )
                                pair_reward_sparse_archive_T = np.delete(
                                    pair_reward_sparse_archive_T, 0, axis=0
                                )
                                num_partner_name += 1
                        else:
                            random_num = random.randint(0, 1)
                            if random_num == 0:
                                # replace the b-th partner in the archive with the k-th partner
                                partner_archive[b].explore_from(partner_population[k])
                                pair_reward_sparse_archive_T[b] = pair_reward_sparse_T[
                                    k
                                ]

                if archive_len == population_size:
                    select_idx = [i for i in range(population_size)]
                    for i in range(population_size):
                        iff = partner_population[i].explore_from(partner_archive[i])
                        if_copy.append(iff)
                elif archive_len > population_size:
                    kmeans = KMeans(n_clusters=population_size, random_state=9).fit(
                        pair_reward_sparse_archive_T
                    )
                    for i in range(population_size):
                        indexes = np.where(kmeans.labels_ == i)
                        indexes = np.concatenate(indexes, axis=0)
                        select_idx[i] = np.random.choice(indexes)
                        iff = partner_population[i].explore_from(
                            partner_archive[select_idx[i]]
                        )
                        if_copy.append(iff)

                select_end = time.time()

                # Step 3: Pairing
                pair_begin = time.time()
                pair_reward_sparse_T = np.array(
                    [
                        pair_reward_sparse_archive_T[select_idx[i]]
                        for i in range(population_size)
                    ]
                )
                pair_reward_sparse = pair_reward_sparse_T.T
                agent_to_pair = list(range(population_size))
                # random pair
                if params["PAIR_MODE"] == 0:
                    partner_idx = np.random.permutation(population_size)
                # max pair
                elif params["PAIR_MODE"] == 1:
                    pair_reward_sparse_copy = pair_reward_sparse.copy()
                    # for each agent, find the partner with max reward
                    for _ in range(population_size):
                        k = np.random.choice(agent_to_pair)
                        agent_to_pair.remove(k)
                        p = np.argmax(pair_reward_sparse_copy[k])
                        partner_idx[k] = p
                        pair_reward_sparse_copy[:, p] = -1000
                # min pair
                elif params["PAIR_MODE"] == 2:
                    pair_reward_sparse_copy = pair_reward_sparse.copy()
                    # for each agent, find the partner with max reward
                    for _ in range(population_size):
                        k = np.random.choice(agent_to_pair)
                        agent_to_pair.remove(k)
                        p = np.argmin(pair_reward_sparse_copy[k])
                        partner_idx[k] = p
                        pair_reward_sparse_copy[:, p] = 10000

                pair_end = time.time()

                for i in range(population_size):
                    # Saving each agent model at the end of the pbt iteration
                    agent_population[i].update_pbt_iter_logs()

                    partner_population[i].update_pbt_iter_logs()

                    if (
                        (pbt_iter == iter_begin)
                        or (pbt_iter == iter_mid)
                        or (pbt_iter == iter_end)
                    ):
                        save_folder = (
                            params["SAVE_DIR"] + agent_population[i].agent_name + "/"
                        )
                        agent_population[i].save_predictor(
                            save_folder + "pbt_iter{}/".format(pbt_iter)
                        )
                        agent_population[i].save(
                            save_folder + "pbt_iter{}/".format(pbt_iter)
                        )

                        # save_folder = params["SAVE_DIR"] + partner_population[i].agent_name + '/'
                        # partner_population[i].save_predictor(save_folder + "pbt_iter{}/".format(pbt_iter))
                        # partner_population[i].save(save_folder + "pbt_iter{}/".format(pbt_iter))

                    # agent_population[i].update_avg_rew_per_step_logs(np.mean(pair_reward_per_step[i]))
                    # partner_population[i].update_avg_rew_per_step_logs(np.mean(pair_reward_per_step[:,i]))
                    avg_sparse_rew_agent = np.mean(pair_reward_sparse_archive_T[:, i])
                    # avg_sparse_rew_partner = np.mean(pair_reward_sparse[:,i])
                    if avg_sparse_rew_agent > best_agent_sparse_rew_avg[i]:
                        best_agent_sparse_rew_avg[i] = avg_sparse_rew_agent
                        # print("New best avg sparse rews {} for agent {}, saving...".format(best_agent_sparse_rew_avg, agent_name))
                        best_save_folder = (
                            params["SAVE_DIR"]
                            + agent_population[i].agent_name
                            + "/best/"
                        )
                        delete_dir_if_exists(best_save_folder, verbose=True)
                        agent_population[i].save_predictor(best_save_folder)
                        agent_population[i].save(best_save_folder)
                    # if avg_sparse_rew_partner > best_partner_sparse_rew_avg[i]:
                    #     best_partner_sparse_rew_avg[i] = avg_sparse_rew_partner
                    #     # print("New best avg sparse rews {} for agent {}, saving...".format(best_agent_sparse_rew_avg, agent_name))
                    #     best_save_folder = params["SAVE_DIR"] + partner_population[i].agent_name + '/best/'
                    #     delete_dir_if_exists(best_save_folder, verbose=True)
                    #     partner_population[i].save_predictor(best_save_folder)
                    #     partner_population[i].save(best_save_folder)

                # np.savez(f'{params["SAVE_DIR"]}pair.npz', reward=pair_reward_sparse, pair=np.array(partner_idx))
                rewards_pd.append(pair_reward_sparse_archive_T)
                select_pd.append(select_idx.copy())
                pair_pd.append(partner_idx.copy())
                copy_pd.append(if_copy.copy())
                out = pd.DataFrame(
                    {
                        "Reward": rewards_pd,
                        "Select": select_pd,
                        "Pair": pair_pd,
                        "Copy": copy_pd,
                    }
                )

                out.to_csv(f'{params["SAVE_DIR"]}pair.csv', index=False)

                logz.log_tabular("Iteration", pbt_iter)
                # for k in range(population_size):
                #     logz.log_tabular(f"Agent{k}DenseReward", format(np.mean(pair_reward_dense[k]), '.5f'))
                for k in range(population_size):
                    logz.log_tabular(
                        f"Agent{k}SparseReward",
                        format(np.mean(pair_reward_sparse_archive_T[:, k]), ".5f"),
                    )
                logz.log_tabular(
                    "MinSparseReward",
                    format(
                        np.min(
                            [
                                np.mean(pair_reward_sparse_archive_T[:, i])
                                for i in range(population_size)
                            ]
                        ),
                        ".5f",
                    ),
                )
                logz.log_tabular(
                    "MaxSparseReward",
                    format(
                        np.max(
                            [
                                np.mean(pair_reward_sparse_archive_T[:, i])
                                for i in range(population_size)
                            ]
                        ),
                        ".5f",
                    ),
                )
                logz.log_tabular(
                    "AverageSparseReward",
                    format(
                        np.mean(
                            [
                                np.mean(pair_reward_sparse_archive_T[:, i])
                                for i in range(population_size)
                            ]
                        ),
                        ".5f",
                    ),
                )
                logz.log_tabular(
                    "StdSparseReward",
                    format(
                        np.std(
                            [
                                np.mean(pair_reward_sparse_archive_T[:, i])
                                for i in range(population_size)
                            ]
                        ),
                        ".5f",
                    ),
                )
                logz.log_tabular("updateTime", format(update_end - update_begin, ".5f"))
                logz.log_tabular("selectTime", format(select_end - select_begin, ".5f"))
                logz.log_tabular("pairTime", format(pair_end - pair_begin, ".5f"))
                logz.dump_tabular()

    pbt_training()
    reset_tf()
    print(params["SAVE_DIR"])


@ex.automain
def run_pbt(params):
    create_dir_if_not_exists(params["SAVE_DIR"])
    print(f'save dir: {params["SAVE_DIR"]}')
    save_dict_to_file(params, params["SAVE_DIR"] + "config")
    for seed in params["SEEDS"]:
        set_global_seed(seed)
        curr_seed_params = params.copy()
        curr_seed_params["SAVE_DIR"] += "seed_{}/".format(seed)
        pbt_one_run(curr_seed_params, seed)
