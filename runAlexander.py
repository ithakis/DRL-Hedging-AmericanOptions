import os
from multiprocessing import cpu_count
import tensorflow as tf
from datetime import datetime

##############################################################################################################################
from pathlib import Path
import json
from typing import Mapping, Sequence

# import tensorflow as tf
import acme
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.tf.savers import make_snapshot
import acme.utils.loggers as log_utils
import dm_env
import numpy as np
# increase the number of rows and columns to display in numpy
np.set_printoptions(edgeitems=30, linewidth=100000)
import sonnet as snt
import pandas as pd
pd.set_option('display.max_rows', 500)

from environment.EnvironmentAlexander import TradingEnv
from environment.utilsAlexander import Utils
import agent.distributional as ad

from absl import app
from absl import flags

FLAGS = flags.FLAGS
# Utility Function Flags
flags.DEFINE_float('S0',                100.0,      'Initial spot price (Default 100.0)')
flags.DEFINE_float('K',                 100.0,      'Initial strike price (Default 100.0)')
flags.DEFINE_integer('init_ttm',        60,         'Number of days in one episode (Default 60)')
flags.DEFINE_float('r',                 0.0,        'Risk-Free-Rate (Default 0.0)')  # Corrected default
flags.DEFINE_float('spread',            0.0,        'Hedging transaction cost (Default 0.0)')
flags.DEFINE_list('liab_ttms',          ['60'],     'List of maturities selected for new adding option (Default [60,])')
flags.DEFINE_float('poisson_rate',      1.0,        'Poisson rate of new options in liability portfolio (Default 1.0)')
flags.DEFINE_float('moneyness_mean',    1.0,        'New options moneyness mean (Default 1.0)')
flags.DEFINE_float('moneyness_std',     0.0,        'New options moneyness std (Default 0.0)')

flags.DEFINE_integer('num_conts_to_add', -1,        'Number of contracts to add to the portfolio (Default -1)')
flags.DEFINE_integer('contract_size',   100,        'Number of shares per option contract (Default 100)')

flags.DEFINE_string('hed_type',         'European', 'Type of the hedging options: European or American (Default European)')
flags.DEFINE_integer('hed_ttm',         20,         'Hedging option maturity in days (Default 20)')

flags.DEFINE_float('init_vol',          0.2,        'Initial spot vol (Default 0.2)')

flags.DEFINE_float('kappa',             None,       'Rate at which variance reverts to its mean in the Heston model (Default None)')
flags.DEFINE_float('theta',             None,       'Long-term mean of the variance in the Heston model (Default None)')
flags.DEFINE_float('volvol',            None,       'Volatility of Volatility in the Heston model (Default None)')
flags.DEFINE_float('rho',               None,       'Correlation between the asset price and volatility in the Heston model (Default None)')

flags.DEFINE_string('stochastic_process',           'GBM', 'Default: GBM or select `Heston`')
flags.DEFINE_integer('time_to_simulate', 30,        'Number of days to simulate (Default 30)')
flags.DEFINE_integer('train_sim',       40_000,     'Train episodes (Default 40,000)')
flags.DEFINE_integer('frq',             1,          'Hedging frequency in steps per day (Default 1)')
flags.DEFINE_integer('TradingDaysPerYear', 252,     'Total trading days in a year (Default 252)')
flags.DEFINE_string('numerical_accuracy', 'low',    'Numerical accuracy level: high, low (Default low)')
flags.DEFINE_integer('n_jobs',          -1,         'Number of CPU cores for parallel processing (Default to all available cores: -1)')
flags.DEFINE_integer('train_seed',      1234,       'Training Seed (Default 1234)')  # New Flag for Training Seed

flags.DEFINE_list('action_space',       ['0', '1'], 'Hedging action space (Default [0,1])')

# Evaluation Flags
flags.DEFINE_integer('eval_sim',        1024*5,     'Evaluation episodes (Default 5,000)')
flags.DEFINE_integer('eval_seed',       1234,       'Evaluation Seed (Default 1234)')

# DRL Flags
flags.DEFINE_string('obj_func',         'var', 'Objective function: meanstd, var, or cvar (Default var)')
flags.DEFINE_integer('n_step',          5, 'DRL TD N-step (Default 5)')
flags.DEFINE_string('critic',           'qr-huber', 'Critic distribution type - c51, qr-huber, qr-gl, qr-gl_tl, qr-lapl, qr-lapl_tl, iqn-huber')
flags.DEFINE_float('std_coef',          1.645, 'Std coefficient when obj_func=meanstd. (Default 1.645)')
flags.DEFINE_float('threshold',         0.95, 'Objective function threshold. (Default 0.95)')
flags.DEFINE_float('lr',                1e-4, 'Learning rate for optimizer (Default 1e-4)')
flags.DEFINE_boolean('per',             False, 'Use PER for Replay sampling (Default False)')
flags.DEFINE_integer('batch_size',      256, 'Batch size to train the Network (Default 256)')
flags.DEFINE_float('priority_exponent', 0.6, 'Priority exponent for the Prioritized replay table (Default 0.6)')

# Other Flags
flags.DEFINE_boolean('vega_obs',        False, 'Include portfolio vega and hedging option vega in state variables (Default False)')
flags.DEFINE_string('logger_prefix',    '', 'Prefix folder for logger (Default None)')
flags.DEFINE_boolean('eval_only',       False, 'Ignore training (Default False)')
flags.DEFINE_string('agent_path',       '', 'Trained agent path, only used when eval_only=True')
flags.DEFINE_float('importance_sampling_exponent', 0.2, 'Importance sampling exponent for updating importance weight for PER (Default 0.2)')

# New Flag for Portfolio Folder
flags.DEFINE_string('portfolio_folder', None, 'Full path to the main portfolio folder to load or create')

def make_logger(work_folder, label, terminal=False):
    log_dir = str(Path('./logs') / work_folder)  # Convert Path to string
    print(f"Initializing CSVLogger with directory: {log_dir}")  # Debug statement
    loggers = [
        log_utils.CSVLogger(log_dir, label=label, add_uid=False)
    ]
    if terminal:
        loggers.append(log_utils.TerminalLogger(label=label))
    
    logger = log_utils.Dispatcher(loggers, log_utils.to_numpy)
    logger = log_utils.NoneFilter(logger)
    return logger

def make_loggers(work_folder):
    return dict(
        train_loop=make_logger(work_folder, 'train_loop', terminal=True),
        eval_loop=make_logger(work_folder, 'eval_loop', terminal=True),
        learner=make_logger(work_folder, 'learner')
    )

def make_environment(utils, logger=None, portfolio_folder=None) -> dm_env.Environment:
    # Determine if we need to load the portfolio from folder
    load_data_from_file = False
    full_folder_path = None

    if portfolio_folder is not None:
        full_folder_path = Path(portfolio_folder)
        if (full_folder_path / "MainPortfolio.pkl").exists():
            load_data_from_file = True
            print(f"Loading portfolio from '{full_folder_path}'")
        else:
            load_data_from_file = False
            print(f"No existing MainPortfolio found in '{full_folder_path}'. A new one will be created.")
    
    # Make sure the environment obeys the dm_env.Environment interface.
    environment = wrappers.GymWrapper(TradingEnv(
        utils=utils,
        logger=logger,
        load_data_from_file=load_data_from_file,
        portfolio_folder=full_folder_path,
    ))
    # Clip the action returned by the agent to the environment spec.
    environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
    environment = wrappers.SinglePrecisionWrapper(environment)

    return environment

def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    num_dimensions = np.prod(action_spec.shape, dtype=int)

    observation_network = tf2_utils.batch_concat

    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])

    critic_network = snt.Sequential([
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        ad.RiskDiscreteValuedHead(vmin, vmax, num_atoms),
    ])

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }

def make_quantile_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    quantile_interval: float = 0.01
) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    num_dimensions = np.prod(action_spec.shape, dtype=int)

    observation_network = tf2_utils.batch_concat

    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])
    quantiles = np.arange(quantile_interval, 1.0, quantile_interval)
    critic_network = snt.Sequential([
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        ad.QuantileDiscreteValuedHead(quantiles=quantiles, prob_type=ad.QuantileDistProbType.MID),
    ])
    
    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }

def make_iqn_networks(
    action_spec: specs.BoundedArray,
    cvar_th: float,
    n_cos=64, n_tau=8, n_k=32,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    quantile_interval: float = 0.01
) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    num_dimensions = np.prod(action_spec.shape, dtype=int)

    observation_network = tf2_utils.batch_concat

    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])
    quantiles = np.arange(quantile_interval, 1.0, quantile_interval)
    critic_network = ad.IQNCritic(cvar_th, n_cos, n_tau, n_k, critic_layer_sizes, quantiles, ad.QuantileDistProbType.MID)
    
    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }

def save_policy(policy_network, checkpoint_folder):
    snapshot = make_snapshot(policy_network)
    export_dir = str(Path(checkpoint_folder) / 'policy')  # Convert Path to string
    tf.saved_model.save(snapshot, export_dir)
    print(f"Policy saved to '{export_dir}'")

def load_policy(policy_network, checkpoint_folder):
    trainable_variables_snapshot = {}
    load_dir = str(Path(checkpoint_folder) / 'policy')  # Convert Path to string
    load_net = tf.saved_model.load(load_dir)
    for var in load_net.trainable_variables:
        var_name = '/'.join(var.name.split('/')[1:])
        trainable_variables_snapshot[var_name] = var.numpy()
    for var in policy_network.trainable_variables:
        var_name = '/'.join(var.name.split('/')[1:])
        if var_name in trainable_variables_snapshot:
            var.assign(trainable_variables_snapshot[var_name])

def initialize_portfolio(folder_path: Path, utils_obj: Utils, logger=None) -> dm_env.Environment:
    if (folder_path / "MainPortfolio.pkl").exists():
        print(f"Loading MainPortfolio from '{folder_path}'")
    else:
        print(f"No existing MainPortfolio found in '{folder_path}'. Creating a new one.")
    return make_environment(
        utils=utils_obj, 
        logger=logger,
        portfolio_folder=folder_path
    )

def main(argv):
    
    ##############################################################################################################################
    number_of_cores = cpu_count() if FLAGS.n_jobs == -1 else FLAGS.n_jobs
    # Set environment variables for TensorFlow threading
    os.environ['OMP_NUM_THREADS'] = str(number_of_cores)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(number_of_cores)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(number_of_cores)

    # Apply the threading configuration
    tf.config.threading.set_intra_op_parallelism_threads(number_of_cores)
    tf.config.threading.set_inter_op_parallelism_threads(number_of_cores)
    ##############################################################################################################################
    
    # Convert FLAGS to a dictionary
    flags_dict = FLAGS.flag_values_dict()
    
    # Pretty-print FLAGS to the console
    print('## FLAGS ####################################################################################')
    for key, value in flags_dict.items():
        print(f'    {key:50}: {value}')
    print('#############################################################################################')

    if FLAGS.per:
        from agent_per.agent_per import D4PG
    else:
        from agent.agent import D4PG

    # Handle Portfolio Folder #######################################################################################################

    if FLAGS.portfolio_folder:
        main_portfolio_folder = Path(FLAGS.portfolio_folder)
        if not main_portfolio_folder.exists():
            raise FileNotFoundError(f"Main portfolio folder '{main_portfolio_folder}' does not exist.")
    else:
        # Create a new main portfolio folder with timestamp up to minute
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")  
        main_portfolio_folder = Path("./portfolios") / f".Portfolio_{current_time}"
        main_portfolio_folder.mkdir(parents=True, exist_ok=False)
        print(f"Created new main portfolio folder at '{main_portfolio_folder}'")

    # Define Train and Eval subfolders
    train_folder = main_portfolio_folder / "Train"
    eval_folder = main_portfolio_folder / "Eval"

    ##############################################################################################################################
    # Create Train and Eval subfolders if they don't exist
    for folder in [train_folder, eval_folder]:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=False)
            print(f"Created subfolder '{folder}'")
        else:
            print(f"Subfolder '{folder}' already exists")

    # Define work_folder for logging
    work_folder = f'stochastic_process={FLAGS.stochastic_process}_spread={FLAGS.spread}_obj={FLAGS.obj_func}_threshold={FLAGS.threshold}_critic={FLAGS.critic}_hedttm={FLAGS.hed_ttm}'
    if FLAGS.logger_prefix:
        work_folder = Path(FLAGS.logger_prefix) / work_folder
    
    # Define the log path
    log_path = Path('./logs') / work_folder
    
    # Create the log directory if it doesn't exist
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Save FLAGS to a JSON file in the log directory
    flags_json_path = log_path / 'flags.json'
    with flags_json_path.open('w') as json_file:
        json.dump(flags_dict, json_file, indent=4)

    ##############################################################################################################################
    utils = Utils(
        # Liability Portfolio Parameters
        S0=FLAGS.S0, K=FLAGS.K, init_ttm=FLAGS.init_ttm, r=FLAGS.r, q=0.00, spread=FLAGS.spread,
        ttms=[int(ttm) for ttm in FLAGS.liab_ttms], poisson_rate=FLAGS.poisson_rate, moneyness_mean=FLAGS.moneyness_mean, moneyness_std=FLAGS.moneyness_std,  # ttms=[30 ,60, 90, 120]
        
        # Contract Parameters
        num_conts_to_add=FLAGS.num_conts_to_add, contract_size=FLAGS.contract_size,

        # Hedging Portfolio Parameters
        hed_ttm=FLAGS.hed_ttm, hed_type=FLAGS.hed_type,

        # init_vol is for both GBM and Heston
        init_vol=FLAGS.init_vol,
        # Heston, Model Parameters
        kappa=FLAGS.kappa, theta=FLAGS.theta, volvol=FLAGS.volvol, rho=FLAGS.rho,
        
        # Simulation Parameters
        stochastic_process=FLAGS.stochastic_process, time_to_simulate=FLAGS.time_to_simulate, num_sim=FLAGS.train_sim, frq=FLAGS.frq, TradingDaysPerYear=FLAGS.TradingDaysPerYear, # 2**14=16384
        numerical_accuracy=FLAGS.numerical_accuracy, n_jobs=FLAGS.n_jobs, np_seed=FLAGS.train_seed,

        # RL Environment Parameters
        action_low=float(FLAGS.action_space[0]), action_high=float(FLAGS.action_space[1]),
    )
    ##############################################################################################################################
    
    # Initialize Utils for Eval
    eval_utils = Utils(
        # Liability Portfolio Parameters
        S0=FLAGS.S0, K=FLAGS.K, init_ttm=FLAGS.init_ttm, r=FLAGS.r, q=0.00, spread=FLAGS.spread,
        ttms=[int(ttm) for ttm in FLAGS.liab_ttms], poisson_rate=FLAGS.poisson_rate, moneyness_mean=FLAGS.moneyness_mean, moneyness_std=FLAGS.moneyness_std,  # ttms=[30 ,60, 90, 120]
        
        # Contract Parameters
        num_conts_to_add=FLAGS.num_conts_to_add, contract_size=FLAGS.contract_size,

        # Hedging Portfolio Parameters
        hed_ttm=FLAGS.hed_ttm, hed_type=FLAGS.hed_type,

        # init_vol is for both GBM and Heston
        init_vol=FLAGS.init_vol,
        # Heston, Model Parameters
        kappa=FLAGS.kappa, theta=FLAGS.theta, volvol=FLAGS.volvol, rho=FLAGS.rho,
        
        # Simulation Parameters
        stochastic_process=FLAGS.stochastic_process, time_to_simulate=FLAGS.time_to_simulate, num_sim=FLAGS.eval_sim, frq=FLAGS.frq, TradingDaysPerYear=FLAGS.TradingDaysPerYear, # 2**14=16384
        numerical_accuracy=FLAGS.numerical_accuracy, n_jobs=FLAGS.n_jobs, np_seed=FLAGS.eval_seed,

        # RL Environment Parameters
        action_low=float(FLAGS.action_space[0]), action_high=float(FLAGS.action_space[1]),
    )
    ##############################################################################################################################
    
    # Initialize Training Environment
    print("Setting up Training Environment...")
    train_logger = make_logger(work_folder, 'train_env')  # Create a logger for train_env
    train_env = initialize_portfolio(train_folder, utils, logger=train_logger)
    train_environment_spec = specs.make_environment_spec(train_env)

    # Initialize Evaluation Environment
    print("Setting up Evaluation Environment...")
    eval_logger = make_logger(work_folder, 'eval_env')  # Create a logger for eval_env
    eval_env = initialize_portfolio(eval_folder, eval_utils, logger=eval_logger)
    # eval_environment_spec = specs.make_environment_spec(eval_env)

    # Create Networks based on Critic Type
    if FLAGS.critic == 'c51':
        agent_networks = make_networks(action_spec=train_environment_spec.actions)
    elif 'qr' in FLAGS.critic:
        agent_networks = make_quantile_networks(action_spec=train_environment_spec.actions)
    elif FLAGS.critic == 'iqn':
        assert FLAGS.obj_func == 'cvar', 'IQN only supports CVaR objective.'
        agent_networks = make_iqn_networks(action_spec=train_environment_spec.actions, cvar_th=FLAGS.threshold)
    
    # Setup Loggers
    loggers = make_loggers(work_folder=work_folder)

    # Initialize the Agent
    agent = D4PG(
        obj_func=FLAGS.obj_func,
        threshold=FLAGS.threshold,
        critic_loss_type=FLAGS.critic,
        environment_spec=train_environment_spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network=agent_networks['observation'],
        n_step=FLAGS.n_step,
        discount=1.0,
        sigma=0.3,
        checkpoint=False,
        logger=loggers['learner'],
        batch_size=FLAGS.batch_size,
        policy_optimizer=snt.optimizers.Adam(FLAGS.lr),
        critic_optimizer=snt.optimizers.Adam(FLAGS.lr),
    )

    if not FLAGS.eval_only:
        print("Starting Training Loop...")
        train_loop = acme.EnvironmentLoop(train_env, agent, label='train_loop', logger=loggers['train_loop'])
        train_loop.run(num_episodes=FLAGS.train_sim)
        save_policy(agent._learner._policy_network, Path('./logs') / work_folder)
    
    if FLAGS.eval_only:
        print("Starting Evaluation Loop...")
        policy_net = agent._learner._policy_network
        if FLAGS.agent_path:
            load_policy(policy_net, FLAGS.agent_path)
        else:
            load_policy(policy_net, Path('./logs') / work_folder)
        eval_policy = snt.Sequential([
            agent_networks['observation'],
            policy_net,
        ])
    else:
        eval_policy = snt.Sequential([
            agent_networks['observation'],
            agent_networks['policy'],
        ])

    eval_actor = actors.FeedForwardActor(policy_network=eval_policy)
    ##############################################################################################################################
    print("Starting Evaluation Loop...")
    eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=loggers['eval_loop'])
    eval_loop.run(num_episodes=FLAGS.eval_sim)
    
    Path(Path('./logs') / work_folder / 'ok').touch()

if __name__ == '__main__':
    # Suppress DeprecationWarnings if desired
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    app.run(main)