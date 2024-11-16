"""A trading environment"""
from typing import Optional
import dataclasses
import numpy as np

import gym
from gym import spaces
from acme.utils import loggers


from environment.TradingAlexander import MainPortfolio

# Flags - If module is run from a Jupyter cell, the custom flags below should be used
from absl import flags
FLAGS = flags.FLAGS

#########################################################################################################
# Remove before use

# Replace Abseil Flags with a Simple Configuration Class
# class FLAGS:
#     vega_obs = True  # Set to True or False based on your requirement


# Remove before use
#########################################################################################################

@dataclasses.dataclass
class StepResult:
    """Logging step metrics for analysis."""
    episode: int = 0
    t: int = 0
    hed_action: float = 0.0
    hed_share: float = 0.0
    stock_price: float = 0.0
    vol: float = 0.0
    stock_position: float = 0.0
    stock_pnl: float = 0.0
    liab_port_gamma: float = 0.0
    liab_port_vega: float = 0.0
    liab_port_pnl: float = 0.0
    hed_cost: float = 0.0
    hed_port_gamma: float = 0.0
    hed_port_vega: float = 0.0
    hed_port_pnl: float = 0.0
    gamma_before_hedge: float = 0.0
    gamma_after_hedge: float = 0.0
    vega_before_hedge: float = 0.0
    vega_after_hedge: float = 0.0
    step_pnl: float = 0.0
    state_price: float = 0.0
    state_gamma: float = 0.0
    state_vega: float = 0.0
    state_hed_gamma: float = 0.0
    state_hed_vega: float = 0.0
    # New Fields
    action_low: float = 0.0
    action_high: float = 0.0
    liab_port_value: float = 0.0
    hed_port_value: float = 0.0
    underlying_value: float = 0.0
    total_port_value: float = 0.0
    # PnL values
    liab_port_pnl: float = 0.0
    hed_port_pnl: float = 0.0
    stock_pnl: float = 0.0
    # Delta values
    liab_port_delta: float = 0.0
    hed_port_delta: float = 0.0
    total_port_delta: float = 0.0
    # Gamma values
    total_port_gamma: float = 0.0
    # Vega values
    total_port_vega: float = 0.0
    # Total PnL
    total_port_pnl: float = 0.0

class TradingEnv(gym.Env):
    """
    This is the Gamma & Vega Trading Environment.
    """
    
    # trade_freq in unit of day, e.g 2: every 2 day; 0.5 twice a day;
    def __init__(self, utils, logger: Optional[loggers.Logger] = None, 
                 load_data_from_file=False, portfolio_folder=None):

        super(TradingEnv, self).__init__()
        self.logger = logger
        # Seed and start
        self.seed(utils.seed)

        # Create or load the portfolio object
        self.portfolio = MainPortfolio(utils, load_data_from_file=load_data_from_file, 
                                       portfolio_folder=portfolio_folder)
        self.utils = utils

        # Other attributes
        self.num_path = self.portfolio.a_price.shape[0]
        self.num_period = self.portfolio.a_price.shape[1]
        self.sim_episode = -1
        self.t = None

        # Time to maturity array
        self.ttm_array = np.arange(self.utils.init_ttm, -self.utils.frq, -self.utils.frq)

        # Action space
        self.action_space = spaces.Box(low=np.array([0.0]), 
                                       high=np.array([1.0]), dtype=np.float32)

        # Observation space
        max_gamma = self.portfolio.liab_port.max_gamma
        max_vega = self.portfolio.liab_port.max_vega
        obs_lowbound = np.array([self.portfolio.a_price.min(), 
                                 -1 * max_gamma * self.utils.contract_size, 
                                 -np.inf])
        obs_highbound = np.array([self.portfolio.a_price.max(), 
                                  max_gamma * self.utils.contract_size,
                                  np.inf])
        if FLAGS.vega_obs:
            obs_lowbound = np.concatenate([obs_lowbound, 
                                            [-1 * max_vega * self.utils.contract_size, -np.inf]])
            obs_highbound = np.concatenate([obs_highbound, 
                                            [max_vega * self.utils.contract_size, np.inf]])
        self.observation_space = spaces.Box(low=obs_lowbound, high=obs_highbound, dtype=np.float32)
            
        # Initializing the state values
        self.num_state = 5 if FLAGS.vega_obs else 3
        self.state = []

        # Reset the environment
        # self.reset()

    def seed(self, seed):
        # Set the np random seed
        np.random.seed(seed)

    def reset(self):
        """
        reset function which is used for each episode (spread is not considered at this moment)
        """

        # repeatedly go through available simulated paths (if needed)
        self.sim_episode = (self.sim_episode + 1) % self.num_path
        self.portfolio.reset(self.sim_episode)

        self.t = 0

        self.portfolio.liab_port.add(self.sim_episode, self.t, self.utils.num_conts_to_add)

        return self.portfolio.get_state(self.t)

    def step(self, action):
        """
        Profit and loss period reward
        """
        result = StepResult(
            episode=self.sim_episode,
            t=self.t,
            hed_action=action[0],
        )
        # Action constraints
        gamma_action_bound = -self.portfolio.get_gamma(self.t)/self.portfolio.hed_port.options[self.sim_episode, self.t].gamma_path[self.t]/self.utils.contract_size
        gamma_action_bound = 0 if np.isnan(gamma_action_bound) else gamma_action_bound
        action_low = [0, gamma_action_bound]
        action_high = [0, gamma_action_bound]
        
        if FLAGS.vega_obs:
            # Vega bounds
            vega_action_bound = -self.portfolio.get_vega(self.t)/self.portfolio.hed_port.options[self.sim_episode, self.t].vega_path[self.t]/self.utils.contract_size
            action_low.append(vega_action_bound)
            action_high.append(vega_action_bound)

        low_val = np.min(action_low)
        high_val = np.max(action_high)

        hed_share = low_val + action[0] * (high_val - low_val)
        result.hed_share = hed_share

        # Storing the action bounds in the StepResult
        result.action_low = low_val  # Store lower bound
        result.action_high = high_val # Store upper bound

        # Current prices at t
        result.gamma_before_hedge = self.portfolio.get_gamma(self.t)
        result.vega_before_hedge = self.portfolio.get_vega(self.t)
        _reward, PnL_List = self.portfolio.step(hed_share, self.t, result)
        result.step_pnl = reward = _reward
        result.gamma_after_hedge = self.portfolio.get_gamma(self.t)
        result.vega_after_hedge = self.portfolio.get_vega(self.t)
        
        # Storing PnL values of the portfolios
        result.liab_port_pnl = PnL_List[0]
        result.hed_port_pnl  = PnL_List[1]
        result.stock_pnl     = PnL_List[2]

        # Storing the values of the portfolios
        result.liab_port_value = self.portfolio.liab_port.get_value(self.t)
        result.hed_port_value = self.portfolio.hed_port.get_value(self.t)
        result.underlying_value = self.portfolio.underlying.get_value(self.t)
        result.total_port_value = self.portfolio.get_value(self.t)
        
        # Storing the delta values of the portfolios
        result.liab_port_delta = self.portfolio.liab_port.get_delta(self.t)
        result.hed_port_delta = self.portfolio.hed_port.get_delta(self.t)
        result.total_port_delta = self.portfolio.get_delta(self.t)

        # Storing the gamma and vega values of the total portfolio
        result.total_port_gamma = self.portfolio.get_gamma(self.t)
        result.total_port_vega = self.portfolio.get_vega(self.t)
        result.total_port_pnl = result.step_pnl

        self.t = self.t + 1

        state = self.portfolio.get_state(self.t)
        if self.t == self.num_period - 1:
            done = True
            state[1:] = 0
        else:
            done = False
        
        result.state_price, result.state_gamma, result.state_hed_gamma = state[:3]
        if FLAGS.vega_obs:
            result.state_vega, result.state_hed_vega = state[3:]
        
        # for other info later
        info = {"path_row": self.sim_episode}
        if self.logger:
            self.logger.write(dataclasses.asdict(result))
        
        return state, reward, done, info
