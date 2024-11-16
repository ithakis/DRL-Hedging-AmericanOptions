import os
from pathlib import Path
import json
import tensorflow as tf
from multiprocessing import cpu_count
from datetime import datetime

import acme
from acme import wrappers
import acme.utils.loggers as log_utils
import dm_env

from environment.EnvironmentAlexander import TradingEnv
from environment.utilsAlexander import Utils
from agent.agent import DeltaHedgeAgent, GammaHedgeAgent, VegaHedgeAgent

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# Utility Function Flags
flags.DEFINE_float('S0',                100.0, 'Initial spot price (Default 100.0)')
flags.DEFINE_float('K',                 100.0, 'Initial strike price (Default 100.0)')
flags.DEFINE_integer('init_ttm',        60, 'Number of days in one episode (Default 60)')
flags.DEFINE_float('r',                 0.0, 'Risk-Free-Rate (Default 0.0)')
flags.DEFINE_float('spread',            0.0, 'Hedging transaction cost (Default 0.0)')
flags.DEFINE_list('liab_ttms',          ['60'], 'List of maturities selected for new adding option (Default [60,])')
flags.DEFINE_float('poisson_rate',      1.0, 'Poisson rate of new options in liability portfolio (Default 1.0)')
flags.DEFINE_float('moneyness_mean',    1.0, 'New options moneyness mean (Default 1.0)')
flags.DEFINE_float('moneyness_std',     0.0, 'New options moneyness std (Default 0.0)')

flags.DEFINE_integer('num_conts_to_add', -1, 'Number of contracts to add to the portfolio (Default -1)')
flags.DEFINE_integer('contract_size',   100, 'Number of shares per option contract (Default 100)')

flags.DEFINE_string('hed_type',         'European', 'Type of the hedging options: European or American (Default European)')
flags.DEFINE_integer('hed_ttm',         20, 'Hedging option maturity in days (Default 20)')

flags.DEFINE_float('init_vol',          0.2, 'Initial spot vol (Default 0.2)')

flags.DEFINE_float('kappa',             None, 'Rate at which variance reverts to its mean in the Heston model (Default None)')
flags.DEFINE_float('theta',             None, 'Long-term mean of the variance in the Heston model (Default None)')
flags.DEFINE_float('volvol',            None, 'Volatility of Volatility in the Heston model (Default None)')
flags.DEFINE_float('rho',               None, 'Correlation between the asset price and volatility in the Heston model (Default None)')

flags.DEFINE_string('stochastic_process', 'GBM', 'Default: GBM or select `Heston`')
flags.DEFINE_integer('time_to_simulate', 30, 'Number of days to simulate (Default 30)')
flags.DEFINE_integer('frq',             1, 'Hedging frequency in steps per day (Default 1)')
flags.DEFINE_integer('TradingDaysPerYear', 252, 'Total trading days in a year (Default 252)')
flags.DEFINE_string('numerical_accuracy', 'low', 'Numerical accuracy level: high, low (Default low)')
flags.DEFINE_integer('n_jobs',          -1, 'Number of CPU cores for parallel processing (Default to all available cores: -1)')
flags.DEFINE_integer('eval_seed',       1234, 'Evaluation Seed (Default 1234)')
flags.DEFINE_string('portfolio_folder', None, 'Full path to the main portfolio folder to load or create')

flags.DEFINE_integer('eval_sim',        1024*5, 'Evaluation episodes (Default 5,000)')
flags.DEFINE_string('strategy',         'delta', 'Hedging strategy opt: delta / gamma / vega (Default delta)')
flags.DEFINE_string('logger_prefix',    '', 'Prefix folder for logger (Default None)')

flags.DEFINE_boolean('vega_obs', False, 'Include portfolio vega and hedging option vega in state variables (Default False)')

def make_logger(work_folder, label, terminal=False):
    log_dir = str(Path('./logs') / work_folder)
    print(f"Initializing CSVLogger with directory: {log_dir}")
    loggers = [
        log_utils.CSVLogger(log_dir, label=label, add_uid=False)
    ]
    if terminal:
        loggers.append(log_utils.TerminalLogger(label=label))
    
    logger = log_utils.Dispatcher(loggers, log_utils.to_numpy)
    logger = log_utils.NoneFilter(logger)
    return logger

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
    environment = wrappers.SinglePrecisionWrapper(environment)

    return environment

def main(argv):
    number_of_cores = cpu_count() if FLAGS.n_jobs == -1 else FLAGS.n_jobs
    # Set environment variables for TensorFlow threading
    os.environ['OMP_NUM_THREADS'] = str(number_of_cores)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(number_of_cores)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(number_of_cores)

    # Apply the threading configuration
    tf.config.threading.set_intra_op_parallelism_threads(number_of_cores)
    tf.config.threading.set_inter_op_parallelism_threads(number_of_cores)

    # Convert FLAGS to a dictionary
    flags_dict = FLAGS.flag_values_dict()
    
    # Pretty-print FLAGS to the console
    print('## FLAGS ####################################################################################')
    for key, value in flags_dict.items():
        print(f'    {key:50}: {value}')
    print('#############################################################################################')

    # Handle Portfolio Folder
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

    # Define Eval subfolder
    eval_folder = main_portfolio_folder / "Eval"
    # Create Eval subfolder if it doesn't exist
    if not eval_folder.exists():
        eval_folder.mkdir(parents=True, exist_ok=False)
        print(f"Created subfolder '{eval_folder}'")
    else:
        print(f"Subfolder '{eval_folder}' already exists")

    # Define work_folder for logging
    work_folder = f'greekhedge_stochastic_process={FLAGS.stochastic_process}_spread={FLAGS.spread}_hedttm={FLAGS.hed_ttm}'
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

    # Initialize Utils
    eval_utils = Utils(
        S0=FLAGS.S0, K=FLAGS.K, init_ttm=FLAGS.init_ttm, r=FLAGS.r, q=0.00, spread=FLAGS.spread,
        ttms=[int(ttm) for ttm in FLAGS.liab_ttms], poisson_rate=FLAGS.poisson_rate, moneyness_mean=FLAGS.moneyness_mean, moneyness_std=FLAGS.moneyness_std,
        num_conts_to_add=FLAGS.num_conts_to_add, contract_size=FLAGS.contract_size,
        hed_ttm=FLAGS.hed_ttm, hed_type=FLAGS.hed_type,
        init_vol=FLAGS.init_vol, kappa=FLAGS.kappa, theta=FLAGS.theta, volvol=FLAGS.volvol, rho=FLAGS.rho,
        stochastic_process=FLAGS.stochastic_process, time_to_simulate=FLAGS.time_to_simulate, num_sim=FLAGS.eval_sim, frq=FLAGS.frq, TradingDaysPerYear=FLAGS.TradingDaysPerYear,
        numerical_accuracy=FLAGS.numerical_accuracy, n_jobs=FLAGS.n_jobs, np_seed=FLAGS.eval_seed,
        action_high=1.0, action_low=0.0
    )

    # Initialize Evaluation Environment
    print("Setting up Evaluation Environment...")
    eval_logger = make_logger(work_folder, 'eval_env')
    eval_env = make_environment(utils=eval_utils, logger=eval_logger, portfolio_folder=eval_folder)

    # Create the evaluation actor and loop.
    if FLAGS.strategy == 'gamma':
        # gamma hedging
        eval_actor = GammaHedgeAgent(eval_env)
    elif FLAGS.strategy == 'delta':
        # delta hedging
        eval_actor = DeltaHedgeAgent(eval_env)
    elif FLAGS.strategy == 'vega':
        # vega hedging
        eval_actor = VegaHedgeAgent(eval_env)
    else:
        raise ValueError(f"Invalid strategy: {FLAGS.strategy}")

    eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, f'eval_{FLAGS.strategy}_loop', True))
    eval_loop.run(num_episodes=FLAGS.eval_sim)

    Path(log_path / 'ok').touch()

if __name__ == '__main__':
    app.run(main)
