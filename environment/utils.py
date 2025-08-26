"""Utility Functions & Imports"""

import gc
import random

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

import inspect
from functools import wraps
from multiprocessing import cpu_count
from typing import List

import QuantLib as ql
from joblib import Parallel, delayed
from numba import njit
from numpy import exp, log
from scipy.stats import qmc
from tqdm import trange

from environment.Trading import Option, SyntheticOption

random.seed(1)
plt.style.use("dark_background")

def validate_input(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Bind the arguments to the function's signature
        sig = inspect.signature(func)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments
        params.pop("self")  # Remove 'self' from parameters

        # Define combined validation rules and error messages
        validations = {
            "S0": (
                lambda x: isinstance(x, (int, float)) and x > 0,
                "S0 must be a positive float representing the initial stock price.",
            ),
            "K": (
                lambda x: isinstance(x, (int, float)) and x > 0,
                "K must be a positive float representing the strike price.",
            ),
            "init_ttm": (
                lambda x: isinstance(x, (int, float)) and x >= 0,
                "init_ttm must be a positive integer or float representing the initial time-to-maturity.",
            ),
            "ttms": (
                lambda x: isinstance(x, list) and all(isinstance(t, int) and t > 0 for t in x),
                "ttms must be a list of positive integers representing time-to-maturity in days.",
            ),
            "r": (
                lambda x: isinstance(x, (int, float)),
                "r must be a float representing the risk-free interest rate.",
            ),
            "mu": (
                lambda x: isinstance(x, (int, float)),
                "mu must be a float representing the equity risk premium.",
            ),
            "q": (lambda x: isinstance(x, (int, float)), "q must be a float representing the dividend yield."),
            "spread": (
                lambda x: isinstance(x, (int, float)) and x >= 0,
                "spread must be a non-negative float representing the bid-ask spread.",
            ),
            "poisson_rate": (
                lambda x: isinstance(x, (int, float)) and x >= 0,
                "poisson_rate must be a positive float representing the Poisson process rate.",
            ),
            "TradingDaysPerYear": (
                lambda x: isinstance(x, int) and x > 0,
                "TradingDaysPerYear must be a positive integer representing total trading days in a year.",
            ),
            "moneyness_mean": (
                lambda x: isinstance(x, (int, float)) and x > 0,
                "moneyness_mean must be a positive float.",
            ),
            "moneyness_std": (
                lambda x: isinstance(x, (int, float)) and x >= 0,
                "moneyness_std must be a non-negative float.",
            ),
            "hed_ttm": (
                lambda x: isinstance(x, (int, float)) and x > 0,
                "hed_ttm must be a positive integer or float representing the time-to-maturity for hedging.",
            ),
            "hed_type": (
                lambda x: x in ["European", "American"],
                "hed_type must be either 'European' or 'American'.",
            ),
            "init_vol": (
                lambda x: x is None or (isinstance(x, (int, float)) and x > 0),
                "init_vol must be a positive float if provided.",
            ),
            "kappa": (
                lambda x: x is None or (isinstance(x, (int, float)) and x > 0),
                "kappa must be a positive float if provided.",
            ),
            "theta": (
                lambda x: x is None or (isinstance(x, (int, float)) and x > 0),
                "theta must be a positive float if provided.",
            ),
            "volvol": (
                lambda x: x is None or (isinstance(x, (int, float)) and x > 0),
                "volvol must be a positive float if provided.",
            ),
            "rho": (
                lambda x: x is None or (isinstance(x, (int, float)) and -1 <= x <= 1),
                "rho must be a float within the range [-1, 1].",
            ),
            "stochastic_process": (
                lambda x: x in [None, "GBM", "Heston"],
                "stochastic_process must be either 'GBM', 'Heston', or None.",
            ),
            "time_to_simulate": (
                lambda x: x is None or (isinstance(x, (int, float)) and x > 0),
                "time_to_simulate must be a positive integer or float if provided.",
            ),
            "num_sim": (
                lambda x: x is None or (isinstance(x, int) and x > 0),
                "num_sim must be a positive integer if provided.",
            ),
            "frq": (
                lambda x: isinstance(x, int) and x > 0,
                "frq must be a positive integer representing simulation steps per day.",
            ),
            "numerical_accuracy": (
                lambda x: x is None or x in ["high", "low", "medium"],
                "numerical_accuracy must be either 'high', 'low', 'medium', or None.",
            ),
            "n_jobs": (
                lambda x: x is None or (isinstance(x, int)),
                "n_jobs must be a positive integer if provided.",
            ),
            "num_conts_to_add": (lambda x: isinstance(x, int), "num_conts_to_add must be an integer."),
            "contract_size": (
                lambda x: isinstance(x, int) and x > 0,
                "contract_size must be a positive integer.",
            ),
            "action_low": (lambda x: isinstance(x, (int, float)), "action_low must be a numerical value."),
            "action_high": (lambda x: isinstance(x, (int, float)), "action_high must be a numerical value."),
        }

        # Perform general validations
        for param, (validate, error_msg) in validations.items():
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
            if not validate(params[param]):
                raise ValueError(error_msg)

        # Perform specific conditional validations
        if params["action_low"] >= params["action_high"]:
            raise ValueError("action_low must be less than action_high.")

        return func(self, *args, **kwargs)

    return wrapper


class Utils:
    @validate_input
    def __init__(
        self,
        # Liability Portfolio Parameters
        S0: float,
        K: float,
        init_ttm: float = 120,
        r: float = 0.0,
        mu: float = 0.0,
        q: float = 0.0,
        spread: float = 0.0,
        # Properties of the Simulation for the Liability Portfolio
        ttms: List[int] = [120],
        poisson_rate: float = 1.0,
        TradingDaysPerYear: int = 252,
        moneyness_mean: float = 1.0,
        moneyness_std: float = 0.0,
        # Hedging Portfolio Parameters
        hed_ttm: float = 60,
        hed_type: str = "European",
        # init_vol is for both GBM and Heston
        init_vol: float = None,
        # Heston, Model Parameters
        kappa: float = None,
        theta: float = None,
        volvol: float = None,
        rho: float = None,
        # Simulation Parameters
        stochastic_process: str = None,
        time_to_simulate: float = None,
        num_sim: int = None,
        frq: int = 1,
        numerical_accuracy: str = None,
        n_jobs: int = None,
        np_seed: int = 1234,
        # Contract Parameters
        num_conts_to_add: int = -1,
        contract_size: int = 100,
        # RL Environment Parameters
        action_low=0,
        action_high=3,
    ):
        """
        Initializes the parameters and configurations for the utility functions in the simulation environment.

        Parameters
        ----------
        S0 : float
            Initial stock price for the liability portfolio.
        K : float
            Strike price for the options in the liability portfolio.
        init_ttm : float, optional
            Initial time-to-maturity (in days) for the liability portfolio. Default is 120.
        r : float, optional
            Risk-free interest rate used for option pricing. Default is 0.0.
        mu : float, optional
            Equity risk premium. The total equity drift will be (r + mu - q). Default is 0.0.
        q : float, optional
            Dividend yield. Default is 0.0.
        spread : float, optional
            Bid-ask spread for options trading. Default is 0.0.
        ttms : list[int], optional
            List of time-to-maturities (in days) for the options in the liability portfolio. Default is [120].
        poisson_rate : float, optional
            Rate of Poisson process for the arrival of new options in the liability portfolio. Default is 1.0.
        TradingDaysPerYear : int, optional
            Total trading days in a year. Default is 252.
        moneyness_mean : float, optional
            Mean of the moneyness for the options in the liability portfolio. Default is 1.0.
        moneyness_std : float, optional
            Standard deviation of the moneyness for the options in the liability portfolio. Default is 0.0.
        hed_ttm : float, optional
            Time-to-maturity (in days) for the hedging options. Default is 60.
        hed_type : str, optional
            Type of the hedging options, either 'European' or 'American'. Default is 'European'.
        init_vol : float, optional
            Initial volatility used for both GBM and Heston models. Default is None.
        kappa : float, optional
            Rate at which variance reverts to its mean (theta) in the Heston model. Default is None.
        theta : float, optional
            Long-term mean of the variance in the Heston model. Default is None.
        volvol : float, optional
            Volatility of the variance process (volatility of volatility) in the Heston model. Default is None.
        rho : float, optional
            Correlation between the asset price and volatility in the Heston model. Default is None.
        stochastic_process : str, optional
            Type of stochastic process used for simulation, either 'GBM' or 'Heston'. Default is None.
        time_to_simulate : float, optional
            Total time (in days) for which the simulation is run. Default is None.
        num_sim : int, optional
            Number of simulation paths. Default is None.
        frq : int, optional
            Hedging Frequency in (every 'frq' days). Default is 1.
        numerical_accuracy : str, optional
            Numerical accuracy level, either 'high', 'medium', or 'low'. Default is None.
        n_jobs : int, optional
            Number of CPU cores used for parallel processing. Default is the number of available cores.
        np_seed : int, optional
            Random seed for NumPy's random number generator. Default is 1234.
        num_conts_to_add : int, optional
            Number of contracts to add to the portfolio. Default is -1.
        contract_size : int, optional
            Contract size representing the number of shares per option contract. Default is 100.
        action_low : float, optional
            Lower bound for the action space in the reinforcement learning environment. Default is 0.
        action_high : float, optional
            Upper bound for the action space in the reinforcement learning environment. Default is 3.

        Attributes
        ----------
        Attributes initialized within the class are used for various simulations, including liability portfolio generation, hedging portfolio simulations, and reinforcement learning environment setups.
        """
        # Liability Portfolio Parameters
        self.S0 = S0
        self.K = K
        self.init_ttm = init_ttm
        self.ttms = ttms  # in days
        self.r = r
        self.mu = mu
        self.q = q
        self.spread = spread
        self.poisson_rate = poisson_rate
        self.moneyness_mean = moneyness_mean
        self.moneyness_std = moneyness_std
        self.TradingDaysPerYear = TradingDaysPerYear  # trading days in a year

        # Hedging Portfolio Parameters
        self.hed_ttm = hed_ttm
        self.hed_type = hed_type

        # GBM and Heston initial vol
        self.init_vol = init_vol
        # Heston Model Parameters
        self.kappa = kappa if stochastic_process == "Heston" else None
        self.theta = theta if stochastic_process == "Heston" else None
        self.volvol = volvol if stochastic_process == "Heston" else None
        self.rho = rho if stochastic_process == "Heston" else None

        # Simulation Parameters
        self.stochastic_process = stochastic_process
        self.time_to_simulate = time_to_simulate  # in days
        self.num_sim = num_sim
        self.frq = frq  # simulation steps per day

        self.numerical_accuracy = numerical_accuracy
        self.dt = self.frq / self.TradingDaysPerYear if self.TradingDaysPerYear else None
        self.num_period = int(self.time_to_simulate * self.frq) if self.time_to_simulate else None

        # Contract Parameters
        self.num_conts_to_add = num_conts_to_add  # Number of contracts to add to the portfolio
        self.contract_size = contract_size  # Contract size is 100 shares per option contract

        # RL Environment Parameters
        self.action_low = action_low
        self.action_high = action_high

        self.n_jobs = n_jobs if n_jobs is not None else cpu_count()
        self._backend = "loky"
        # set the np random seed
        np.random.seed(np_seed)
        self.seed = np_seed
        self._number_of_cores = cpu_count()

        # min volatility in case the QE scheme generates negative non possitive values
        self._min_vol = 0.00001

    def get_params(self):
        """
        Returns the initialized parameters and configurations for the utility functions in the simulation environment.
        """
        return {
            "S0": self.S0,
            "K": self.K,
            "init_ttm": self.init_ttm,
            "ttms": self.ttms,
            "r": self.r,
            # "mu": self.mu,
            "q": self.q,
            "spread": self.spread,
            "poisson_rate": self.poisson_rate,
            "moneyness_mean": self.moneyness_mean,
            "moneyness_std": self.moneyness_std,
            "TradingDaysPerYear": self.TradingDaysPerYear,
            "hed_ttm": self.hed_ttm,
            "hed_type": self.hed_type,
            "init_vol": self.init_vol,
            "kappa": self.kappa,
            "theta": self.theta,
            "volvol": self.volvol,
            "rho": self.rho,
            "stochastic_process": self.stochastic_process,
            "time_to_simulate": self.time_to_simulate,
            "num_sim": self.num_sim,
            "numerical_accuracy": self.numerical_accuracy,
            "frq": self.frq,
            "n_jobs": self.n_jobs,
            "np_seed": self.seed,
            "num_conts_to_add": self.num_conts_to_add,
            "contract_size": self.contract_size,
            "action_low": self.action_low,
            "action_high": self.action_high,
        }

    def __eq__(self, other):
        if not isinstance(other, Utils):
            return NotImplemented

        # Retrieve parameters from both objects
        self_params = self.get_params().copy()
        other_params = other.get_params().copy()

        # Exclude specific keys from comparison
        excluded_keys = {"n_jobs", "np_seed", "obj_func"}
        for key in excluded_keys:
            self_params.pop(key, None)
            other_params.pop(key, None)

        # Check for equality
        if self_params == other_params:
            return True
        else:
            # Identify differences
            differences = {}
            all_keys = set(self_params.keys()).union(other_params.keys())
            for key in all_keys:
                self_value = self_params.get(key)
                other_value = other_params.get(key)
                if self_value != other_value:
                    differences[key] = {"self": self_value, "other": other_value}

            # Print differences
            print("Objects are not equal. Differences:")
            for key, vals in differences.items():
                print(f" - {key}: self = {vals['self']} vs other = {vals['other']}")

            return False

    def __repr__(self):
        params = self.get_params()
        df = pd.DataFrame(params.items(), columns=["Parameter", "Value"])
        return df.to_string(index=False)

    def get_sim_path_GBM(self):
        """
        Simulates the path of asset prices using the Geometric Brownian Motion (GBM) model.

        This function generates multiple simulation paths for the asset price using the GBM process,
        which is widely used in financial modeling to represent the stochastic behavior of asset prices.
        The GBM model assumes that the asset price follows a log-normal distribution, with continuous
        compounding and a constant drift and volatility.

        Parameters
        ----------
        None
            This method uses the parameters initialized in the Utils class.

        Returns
        -------
        a_price : np.ndarray
            A 2D numpy array of simulated asset prices with shape (num_sim, num_period + 1), where
            num_sim is the number of simulation paths and num_period is the number of time steps.
        vol : np.ndarray
            A 2D numpy array of constant volatility values, with the same shape as `a_price`.

        Methodology
        -----------
        The GBM model is defined as:
        dS_t = S_t * (r + mu - q) * dt + S_t * sigma * dW_t

        where:
        - S_t is the asset price at time t
        - r is the risk-free interest rate (used for option pricing)
        - mu is the equity risk premium 
        - q is the dividend yield
        - sigma is the volatility of the asset returns
        - dW_t is the increment of a Wiener process (normal distribution with mean 0 and variance dt)
        - dt is the time increment

        The asset price is simulated at each time step using the discrete version of the GBM equation:
        S_{t+1} = S_t * exp((r + mu - q - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z_t)

        where Z_t is a random variable drawn from a standard normal distribution.

        Example Usage
        -------------
        utils = Utils(S0=100., K=100., ...)
        a_price, vol = utils.get_sim_path_GBM()

        Notes
        -----
        - The initial stock price S0 is provided during the initialization of the Utils class.
        - This function assumes a constant volatility environment.
        - The output volatility array is filled with the constant volatility used in the simulation.
        """
        z = np.random.normal(size=(self.num_sim, self.num_period + 1))

        a_price = np.zeros((self.num_sim, self.num_period + 1))
        a_price[:, 0] = self.S0

        for t in range(self.num_period):
            a_price[:, t + 1] = a_price[:, t] * np.exp(
                (self.r + self.mu - self.q - (self.init_vol**2) / 2) * self.dt
                + self.init_vol * np.sqrt(self.dt) * z[:, t]
            )
        return a_price, np.full_like(a_price, self.init_vol)

    def get_sim_path_Heston(
        self, psiC=1.5, gamma1=0.5, gamma2=0.5, Martingale_Correction=True, _show_progress=True,
        measure='P',        #  <-- 'Q' (default) or 'P'
        lambda_v=0,      #  <-- variance-risk premium (only used if measure == 'P') default under P is -1.5
    ):
        """
        Simulates the Heston model using the Quadratic Exponential (QE) scheme, supporting both Q and P measure.


        Parameters
        ----------
        S0 : float
            Initial stock price.
        V0 : float
            Initial variance.
        rho : float
            Correlation between the two Brownian motions.
        theta : float
            Long-term mean of the variance.
        sigma : float
            Volatility of the variance process (vol of vol).
        kappa : float
            Rate at which variance reverts to theta.
        r : float
            Risk-free interest rate.
        q : float
            Dividend yield.
        dt : float
            Time increment.
        T_steps : int
            Number of time steps.
        N_paths : int
            Number of simulated paths.
        psiC : float, optional
            Threshold parameter for the QE scheme (default is 1.5).
        gamma1 : float, optional
            Gamma parameter for the QE scheme (default is 0.5).
        gamma2 : float, optional
            Gamma parameter for the QE scheme (default is 0.5).
        Martingale_Correction : bool, optional
            If True, applies Martingale correction (default is False).
        measure : str
            'Q' (risk-neutral, default) or 'P' (physical measure, applies variance risk premium).
        lambda_v : float
            Variance risk premium λᵥ (only used if measure == 'P').
        _show_progress : bool, optional
            If True, displays progress of the simulation (default is True).

        Returns
        -------
        S : ndarray
            Simulated stock prices, shape (N_paths, T_steps+1).
        V : ndarray
            Simulated volatility, shape (N_paths, T_steps+1). - Square Rooted Variances

        Notes
        -----
        The function uses Sobol sequences for generating random numbers and the inverse cumulative normal distribution to
        obtain normally distributed random variables. The QE scheme is used to handle the variance process.

        Examples
        --------
        S_T_array_QE, V_T_array_QE = QESim(S0=100, V0=0.04, rho=-0.7, theta=0.04, sigma=0.2, kappa=1.5, r=0.05, q=0.02, dt=1/252,
                                        T_steps=252, N_paths=10000, Martingale_Correction=1)

        References
        ----------
        Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance, 11(3), 1-42.
        """
        # -- 1. Unpack model inputs ----------------------------------------------
        S0   = self.S0
        V0   = self.init_vol**2
        r,mu,q  = self.r, self.mu, self.q
        kappa_Q, theta_Q = self.kappa, self.theta
        xi,rho = self.volvol, self.rho
        N_paths, T_steps, dt = self.num_sim, self.num_period, self.dt

        # -- 2. Apply variance-risk premium if required ---------------------------
        if measure.upper() == 'P':
            kappa = kappa_Q - lambda_v
            if kappa <= 0:
                raise ValueError('kappa_P must stay positive; pick λᵥ < κ_Q.')
            theta = (kappa_Q * theta_Q) / kappa        # θ_P
        else:              # stay under Q
            kappa, theta = kappa_Q, theta_Q

        # -- 3. Pre-compute constants with the *effective* (kappa,theta) ----------
        E  = np.exp(-kappa * dt)
        K0 = -(kappa * rho * theta) / xi * dt
        K1 = (kappa * rho / xi - 0.5) * gamma1 * dt - rho / xi
        K2 = (kappa * rho / xi - 0.5) * gamma2 * dt + rho / xi
        K3 = (1 - rho**2) * gamma1 * dt
        K4 = (1 - rho**2) * gamma2 * dt
        A  = K2 + 0.5 * K4
        if Martingale_Correction:
            K0_star = np.empty(N_paths)

        # Use the Generator as the seed parameter
        dim = 3 * (T_steps + 1)  # Z1,Z2,U per time step
        sampler = qmc.Sobol(d=dim, scramble=True, seed=self.seed)
        U_mat = sampler.random(n=N_paths)  # (N_paths, dim)

        split = T_steps + 1
        Z1 = norm.ppf(U_mat[:, :split]).T
        Z2 = norm.ppf(U_mat[:, split : 2 * split]).T
        U = U_mat[:, 2 * split :].T
        del dim, sampler, U_mat, split

        # Storage
        S = np.zeros((T_steps + 1, N_paths))
        V = np.zeros((T_steps + 1, N_paths))
        S[0] = log(S0)
        V[0] = V0

        for t in trange(
            1, T_steps + 1, desc="Generating Paths", position=0, leave=False, disable=not _show_progress
        ):
            m = theta + (V[t - 1] - theta) * E
            s2 = (V[t - 1] * xi**2 * E) / kappa * (1 - E) + (theta * xi**2) / (2 * kappa) * (1 - E) ** 2
            psi = s2 / m**2

            idx = psi <= psiC
            # When psi <= psiC
            b = np.sqrt(2 / psi[idx] - 1 + np.sqrt(2 / psi[idx] * (2 / psi[idx] - 1)))
            a = m[idx] / (1 + b**2)
            V[t, idx] = a * (b + Z1[t, idx]) ** 2

            # When psi > psiC
            p = (psi[~idx] - 1) / (psi[~idx] + 1)
            beta = (1 - p) / m[~idx]
            idx2 = U[t, ~idx] <= p
            V[t, ~idx] = np.where(idx2, 0, 1 / beta * np.log((1 - p) / (1 - U[t, ~idx])))

            if Martingale_Correction:
                K0_star[idx] = (
                    -(A * b**2 * a) / (1 - 2 * A * a)
                    + 0.5 * np.log(1 - 2 * A * a)
                    - (K1 + 0.5 * K3) * V[t - 1, idx]
                )  # type: ignore
                K0_star[~idx] = -np.log(p + (beta * (1 - p)) / (beta - A)) - (K1 + 0.5 * K3) * V[t - 1, ~idx]  # type: ignore

                S[t] = (
                    S[t - 1]
                    + (r + mu - q) * dt
                    + K0_star
                    + K1 * V[t - 1]
                    + K2 * V[t]
                    + np.sqrt(K3 * V[t - 1] + K4 * V[t]) * Z2[t]
                )
            else:
                S[t] = (
                    S[t - 1]
                    + (r + mu - q) * dt
                    + K0
                    + K1 * V[t - 1]
                    + K2 * V[t]
                    + np.sqrt(K3 * V[t - 1] + K4 * V[t]) * Z2[t]
                )

        S = exp(S).T
        V = V.T

        # Find if there are NaN values and replace those paths with S0
        idx = np.isnan(S).sum(axis=1) == 0
        if idx.sum() < N_paths:
            print(
                f"Warning: {N_paths - idx.sum()} paths removed due to NaN values and replaced with E[S_t] and E[V_t]"
            )
            S[~idx] = S0 * exp((r + mu - q) * np.arange(T_steps + 1) * dt)
            V[~idx] = V0 * np.exp(-kappa * np.arange(T_steps + 1) * dt) + theta * (
                1 - np.exp(-kappa * np.arange(T_steps + 1) * dt)
            )

        # print(f'Min_S_t: {S.min()}, Max_S_t: {S.max()}')
        return S, np.sqrt(V)

    def init_env(self):
        """Initialize environment
        Entrypoint to simulate market dynamics:
        1). stock prices
        2). implied volatilities

        If it is constant vol environment, then BSM model is used.
        If it is stochastic vol environment, then SABR model is used.

        Returns:
            np.ndarray: underlying asset price in shape (num_path, num_period)
            np.ndarray: implied volatility in shape (num_path, num_period)

        ** Note: Used in Trading.py
        """
        if self.stochastic_process == "Heston":
            return self.get_sim_path_Heston()
        elif self.stochastic_process == "GBM":
            return self.get_sim_path_GBM()

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _American_Option_GBM(
        S0: float, K: float, T: float, r: float, q: float, sigma: float, N=1000, option_type="put"
    ):
        """
        Calculate the price and Greeks (Delta, Gamma) of an American option with continuous dividend payments
        using the Binomial Tree model under the Geometric Brownian Motion (GBM) framework.

        Parameters:
        ----------
        S0 : float
            Initial stock price of the underlying asset.
        K : float
            Strike price of the option.
        T : float
            Time to maturity in years.
        r : float
            Risk-free interest rate.
        q : float
            Continuous dividend yield of the underlying asset.
        sigma : float
            Volatility of the underlying asset.
        N : int, optional
            Number of time steps in the binomial tree, default is 1000.
        option_type : str, optional
            Type of the option, either 'call' or 'put'. Default is 'put'.

        Returns:
        -------
        option_price : float
            The calculated price of the American option.
        delta : float
            The Delta of the option, representing the sensitivity of the option's price to changes in the underlying asset's price.
        gamma : float
            The Gamma of the option, representing the sensitivity of the Delta to changes in the underlying asset's price.

        Notes:
        -----
        - This method utilizes a binomial tree approach to evaluate American options, allowing for early exercise.
        - The option's value is calculated by backward induction, taking into account the optimal exercise strategy at each node.
        - The Greeks are derived from the final option prices obtained at the first few time steps of the binomial tree.
        - The method is optimized using Numba's `njit` decorator to enhance computational efficiency.

        Examples:
        --------
        option_price, delta, gamma = _American_Option_GBM(S0=100, K=95, T=1, r=0.05, q=0.02, sigma=0.2, N=1000, option_type='put')

        References:
        ----------
        - This method is a specialized implementation for pricing American options using a binomial tree model,
          often employed in quantitative finance for its ability to handle early exercise features typical of American options.
        """
        # print paramters
        # print(f"S0: {S0}, K: {K}, T: {T}, r: {r}, sigma: {sigma}, q: {q}, N: {N}, option_type: {option_type}")

        dt = T / N  # time step
        u = np.exp(sigma * np.sqrt(dt))  # up factor
        d = 1 / u  # down factor
        p = (np.exp((r - q) * dt) - d) / (u - d)  # risk-neutral probability

        # Initialize asset prices matrix
        ST = np.zeros((N + 1, N + 1), dtype=np.float64)
        ST[0, 0] = S0
        for i in range(1, N + 1):
            ST[i, 0] = ST[i - 1, 0] * d
            for j in range(1, i + 1):
                ST[i, j] = ST[i - 1, j - 1] * u

        # Initialize option values matrix
        values = np.zeros((N + 1, N + 1), dtype=np.float64)
        if option_type == "call":
            values[N, :] = np.maximum(0, ST[N, :] - K)
        else:
            values[N, :] = np.maximum(0, K - ST[N, :])

        # Backward induction
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                values[i, j] = (p * values[i + 1, j + 1] + (1 - p) * values[i + 1, j]) * np.exp(-r * dt)
                if option_type == "call":
                    values[i, j] = max(values[i, j], ST[i, j] - K)
                else:
                    values[i, j] = max(values[i, j], K - ST[i, j])

        # Option price
        option_price = values[0, 0]

        # Delta calculation
        delta = (values[1, 1] - values[1, 0]) / (ST[1, 1] - ST[1, 0])

        # Gamma calculation
        delta_up = (values[2, 2] - values[2, 1]) / (ST[2, 2] - ST[2, 1])
        delta_down = (values[2, 1] - values[2, 0]) / (ST[2, 1] - ST[2, 0])
        gamma = (delta_up - delta_down) / ((ST[2, 2] - ST[2, 0]) / 2)

        return option_price, delta, gamma

    @staticmethod
    def _American_Option_Heston(
        S0, K, T, r, q, v0, kappa, theta, volvol, rho, N_gridpoints=100, Type="put", exercise="American"
    ):
        """
        Calculate the price and Greeks (Delta, Gamma) of an American or European option under the Heston stochastic volatility model using a finite difference method.

        Parameters:
        -----------
        S0 : float
            Initial stock price.
        K : float
            Strike price of the option.
        T : float
            Time to maturity in years.
        r : float
            Risk-free interest rate.
        q : float
            Continuous dividend yield.
        v0 : float
            Initial variance (squared volatility) of the underlying asset.
        kappa : float
            Rate at which variance reverts to its long-term mean (theta).
        theta : float
            Long-term mean of the variance.
        volvol : float
            Volatility of the variance process (volatility of volatility).
        rho : float
            Correlation between the asset price and its variance.
        N_gridpoints : int, optional
            Number of grid points for the finite difference method, by default 100.
        Type : str, optional
            Type of option ('call' or 'put'), by default 'put'.
        exercise : str, optional
            Exercise style of the option ('American' or 'European'), by default 'American'.

        Returns:
        --------
        price : float
            The calculated price of the option.
        delta : float
            The Delta of the option, which measures the sensitivity of the option price to changes in the underlying asset price.
        gamma : float
            The Gamma of the option, which measures the sensitivity of Delta to changes in the underlying asset price.

        Notes:
        ------
        This function leverages QuantLib's `FdHestonVanillaEngine` to solve the Heston model PDE using a finite difference method. The function supports both American and European exercise styles and provides a detailed approximation of option prices and their corresponding Greeks under the Heston model framework.

        The method includes a grid configuration defined by `N_gridpoints` for time and space discretization, and it applies a Modified Craig-Sneyd finite difference scheme to ensure numerical stability and accuracy.

        Warning:
        --------
        If the Feller condition (2 * kappa * theta >= volvol^2) is violated, the variance process may become negative, which could lead to inaccuracies in the simulation.
        """
        if 2 * kappa * theta < volvol**2:
            print("Warrning: Feller condition violated: 2*kappa*theta < volvol**2")

        S0 = float(S0)
        K = float(K)
        T = float(T)
        r = float(r)
        q = float(q)
        v0 = float(v0)
        kappa = float(kappa)
        theta = float(theta)
        volvol = float(volvol)
        rho = float(rho)

        today = ql.Date().todaysDate()
        initialValue = ql.QuoteHandle(ql.SimpleQuote(S0))
        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, ql.Actual365Fixed()))

        hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, initialValue, v0, kappa, theta, volvol, rho)
        hestonModel = ql.HestonModel(hestonProcess)

        tGrid, xGrid, vGrid = N_gridpoints, N_gridpoints, int(N_gridpoints / 2)
        dampingSteps = 2
        fdScheme = ql.FdmSchemeDesc.ModifiedCraigSneyd()

        engine = ql.FdHestonVanillaEngine(hestonModel, tGrid, xGrid, vGrid, dampingSteps, fdScheme)
        exerciseDate = today + ql.Period(round(T * 365), ql.Days)
        option_type = ql.Option.Call if Type == "Call" else ql.Option.Put
        payoff = ql.PlainVanillaPayoff(option_type, K)

        exercise = (
            ql.AmericanExercise(today, exerciseDate)
            if exercise == "American"
            else ql.EuropeanExercise(exerciseDate)
        )

        americanOption = ql.VanillaOption(payoff, exercise)

        americanOption.setPricingEngine(engine)

        price = americanOption.NPV()
        delta = americanOption.delta()
        gamma = americanOption.gamma()

        return price, delta, gamma

    def American_put_option(
        self,
        iv: np.ndarray,
        ttms: np.ndarray,
        S0: np.ndarray,
        K: np.ndarray,
        T: int = 252,
        stochastic_process: str = None,
    ):
        """
        Computes the price and Greeks (Delta, Gamma, Vega) of an American put option using different stochastic processes.

        This function supports the following stochastic processes:
        - Geometric Brownian Motion (GBM) using a binomial tree model.
        - Heston model using the Finite Difference Method (FDM) with a Heston process.

        Parameters
        ----------
        iv : np.ndarray
            Implied volatility. For GBM, this is the volatility of the underlying asset. For Heston, this represents the initial volatility.
        ttms : np.ndarray
            Time-to-maturity array in days. Should have the same shape as `S0`.
        S0 : np.ndarray
            Current stock prices for the underlying asset. Should match the shape of `ttms`.
        K : np.ndarray
            Strike prices for the options. Should match the shape of `S0`.
        T : int, optional
            Total trading days in a year, by default 252.
        stochastic_process : str, optional
            The stochastic process used for the simulation ('GBM' or 'Heston'). This determines the model for pricing the option.

        Returns
        -------
        price : np.ndarray
            The calculated prices of the American put options.
        delta : np.ndarray
            The Delta of the options, indicating sensitivity to changes in the underlying asset price.
        gamma : np.ndarray
            The Gamma of the options, indicating the rate of change of Delta with respect to the underlying asset price.
        vega : np.ndarray
            The Vega of the options, indicating sensitivity to changes in the implied volatility.

        Raises
        ------
        ValueError
            If an unsupported stochastic process is specified.

        Notes
        -----
        - The GBM model uses a binomial tree to compute the option prices and Greeks. Vega is computed via finite differences.
        - The Heston model utilizes a grid-based finite difference method for pricing, considering stochastic volatility.
        - This function assumes that the time-to-maturity (`ttms`) and other inputs are provided as arrays with matching shapes, enabling parallelized processing across multiple scenarios.

        Examples
        --------
        >>> utils = Utils(...)
        >>> price, delta, gamma, vega = utils.American_put_option(iv, ttms, S0, K, stochastic_process='GBM')
        >>> price, delta, gamma, vega = utils.American_put_option(iv, ttms, S0, K, stochastic_process='Heston')

        """
        # Initialize result arrays with the same shape as S0
        dtype = np.float64  # if self.numerical_accuracy == "high" else np.float32
        price = np.zeros_like(S0, dtype=dtype)
        delta = np.zeros_like(S0, dtype=dtype)
        gamma = np.zeros_like(S0, dtype=dtype)
        vega = np.zeros_like(S0, dtype=dtype)

        # Compute matured option payoff
        matured_option = ttms == 0
        price = np.where(matured_option, np.maximum(K - S0, 0), price)

        active_indices = list(zip(*np.where(ttms > 0)))
        if stochastic_process == "GBM":
            N = 200 if self.numerical_accuracy == "low" else 2_000  # number of time steps in the binomial tree.
            eps_iv = 2e-2  # As a Percentage Bump: For numerical differentiation for calculating vega

            def calculate_gbm_option(idx):
                fT = max(ttms[idx], 1) / T
                C0, delta, gamma = self._American_Option_GBM(
                    S0=S0[idx], K=K[idx], T=fT, r=self.r, q=self.q, sigma=iv[idx], option_type="put", N=N
                )
                C0_up = self._American_Option_GBM(
                    S0=S0[idx],
                    K=K[idx],
                    T=fT,
                    r=self.r,
                    q=self.q,
                    sigma=iv[idx] * (1 + eps_iv),
                    option_type="put",
                    N=N,
                )[0]
                C0_down = self._American_Option_GBM(
                    S0=S0[idx],
                    K=K[idx],
                    T=fT,
                    r=self.r,
                    q=self.q,
                    sigma=iv[idx] * (1 - eps_iv),
                    option_type="put",
                    N=N,
                )[0]

                vega = (1 / 100) * (C0_up - C0_down) / (2 * iv[idx] * eps_iv)
                return idx, C0, delta, gamma, vega

            # Parallel processing for GBM
            results = Parallel(
                n_jobs=self.n_jobs,
                backend=self._backend,
                pre_dispatch="1*n_jobs",
                batch_size=100,
                timeout=120,
                mmap_mode="r",
                temp_folder="/home/atsoskouno/data/storageith2/gamma-vega-hedging-American-Heston/.temp_dir_loky",
                # max_nbytes='200000M'
                # inner_max_num_threads=1,
            )(delayed(calculate_gbm_option)(idx) for idx in active_indices)
            # Assign results to the respective arrays
            for result in results:
                idx, C0, delta_val, gamma_val, vega_val = result
                price[idx] = C0
                delta[idx] = delta_val
                gamma[idx] = gamma_val
                vega[idx] = vega_val
            del results
            gc.collect()

        elif stochastic_process == "Heston":
            N_gridpoints = 50 if self.numerical_accuracy == "low" else 110
            eps_iv = 2e-2  # As a Percentage Bump: For numerical differentiation for calculating vega

            def calculate_heston_option(idx):
                fT = max(ttms[idx], 1) / T
                v0 = iv[idx] ** 2
                vol_eps = (np.sqrt(v0) * eps_iv) ** 2
                vol_eps = np.where(vol_eps < 0.00002, 0.00002, vol_eps)  # To ensure better calculation of vega

                theta_up = (np.sqrt(self.theta) + vol_eps) ** 2
                v0_up = (np.sqrt(v0) + vol_eps) ** 2

                C0, delta, gamma = self._American_Option_Heston(
                    S0=S0[idx],
                    K=K[idx],
                    T=fT,
                    r=self.r,
                    q=self.q,
                    v0=v0,
                    theta=self.theta,
                    kappa=self.kappa,
                    volvol=self.volvol,
                    rho=self.rho,
                    N_gridpoints=N_gridpoints,
                    Type="put",
                    exercise="American",
                )

                C0_iv_up = self._American_Option_Heston(
                    S0=S0[idx],
                    K=K[idx],
                    T=fT,
                    r=self.r,
                    q=self.q,
                    v0=v0_up,
                    theta=theta_up,
                    kappa=self.kappa,
                    volvol=self.volvol,
                    rho=self.rho,
                    N_gridpoints=N_gridpoints,
                    Type="put",
                    exercise="American",
                )[0]

                vega = (1 / 100) * (C0_iv_up - C0) / vol_eps
                return idx, C0, delta, gamma, vega

            # Parallel processing for Heston
            results = Parallel(
                n_jobs=self.n_jobs,
                backend=self._backend,
                pre_dispatch="1*n_jobs",
                batch_size=100,
                timeout=120,
                mmap_mode="r",
                temp_folder="/home/atsoskouno/data/storageith2/gamma-vega-hedging-American-Heston/.temp_dir_loky",
                # inner_max_num_threads=1,
            )(delayed(calculate_heston_option)(idx) for idx in active_indices)

            # Assign results to the respective arrays
            for result in results:
                idx, C0, delta_val, gamma_val, vega_val = result
                price[idx] = C0
                delta[idx] = delta_val
                gamma[idx] = gamma_val
                vega[idx] = vega_val

        else:
            raise ValueError(
                f"Stochastic process must be either 'GBM' or 'Heston' but got {stochastic_process}"
            )

        return price, delta, gamma, vega

    @staticmethod
    def _European_Option_GBM(
        iv, ttm: np.ndarray, S: np.ndarray, K: np.ndarray, r: float, q: float, T: int = 252
    ):
        """
        Calculate the price and Greeks (delta, gamma, and vega) of a European put option using the Black-Scholes-Merton model.

        Parameters
        ----------
        iv : np.ndarray
            Implied volatility. This can be a scalar for constant volatility or an array matching the shape of S for stochastic volatility.
        ttm : np.ndarray
            Time to maturity, with the same dimensionality as the stock price S. If batch dimensions are not present, the time to maturity will be expanded to match S.
        S : np.ndarray
            Current stock price.
        K : np.ndarray
            Strike price of the option, matching the dimensionality of the stock price S.
        r : float
            Risk-free interest rate.
        q : float
            Dividend yield.
        T : int, optional
            Number of business days in a year. Defaults to 252.

        Returns
        -------
        price : np.ndarray
            The calculated price of the European put option, with the same shape as the stock price S.
        delta : np.ndarray
            The delta of the option, representing the rate of change of the option price with respect to the stock price.
        gamma : np.ndarray
            The gamma of the option, representing the rate of change of delta with respect to the stock price.
        vega : np.ndarray
            The vega of the option, representing the sensitivity of the option price to changes in volatility.

        Notes
        -----
        - This function assumes that the options are actively traded and that they mature according to the given time-to-maturity (ttm).
        - The function handles both active and matured options, where active options have ttm > 0, and matured options have ttm = 0.
        - For matured options, the payoff is directly computed as the intrinsic value (max(K - S, 0)).
        - The function integrates with stochastic models by adjusting the implied volatility input.
        - This method is based on the classical Black-Scholes-Merton framework, which assumes log-normal distribution of stock prices and constant volatility.
        """
        if (ttm.ndim + 1) == S.ndim:
            # expand batch dim
            ttm = np.tile(np.expand_dims(ttm, 0), (S.shape[0],) + tuple([1] * ttm.ndim))
        assert ttm.ndim == S.ndim, "Maturity dim does not match spot dim"
        for i in range(ttm.ndim):
            assert ttm.shape[i] == S.shape[i], f"Maturity dim {i} size does not match spot dim {i} size."
        # active option
        active_option = (ttm > 0).astype(np.uintp)
        matured_option = (ttm == 0).astype(np.uintp)

        # active option
        fT = np.maximum(ttm, 1) / T
        d1 = (np.log(S / K) + (r - q + iv * iv / 2) * np.abs(fT)) / (iv * np.sqrt(fT))
        d2 = d1 - iv * np.sqrt(fT)
        n_prime = np.exp(-1 * d1 * d1 / 2) / np.sqrt(2 * np.pi)

        active_bs_price = K * np.exp(-r * fT) * norm.cdf(-d2) - S * np.exp(-q * fT) * norm.cdf(-d1)
        active_bs_delta = np.exp(-q * fT) * (norm.cdf(d1) - 1)
        active_bs_gamma = (n_prime * np.exp(-q * fT)) / (S * iv * np.sqrt(fT))
        active_bs_vega = (1 / 100) * S * np.exp(-q * fT) * np.sqrt(fT) * n_prime

        # matured option
        payoff = np.maximum(K - S, 0)

        # consolidate
        # print(f'matured_option * put_payoff).mean(): {(matured_option * payoff)[np.where(matured_option == 1)].mean()}')
        # print(f'active_price.mean(): {active_option*active_bs_price.mean()}')
        price = active_option * active_bs_price + matured_option * payoff
        delta = active_option * active_bs_delta
        gamma = active_option * active_bs_gamma
        vega = active_option * active_bs_vega

        return price, delta, gamma, vega

    @staticmethod
    def _European_Option_Heston(
        S0, K, T, r, q, v0, kappa, theta, volvol, rho, Type="put", accuracy=1e-9, max_evaluations=1_000_000
    ):
        """
        Calculate the price of a European option under the Heston stochastic
        volatility model using QuantLib.

        Parameters:
        -----------
        S0 : float
            Initial stock price.
        K : float
            Strike price of the option.
        T : float
            Time to maturity in years.
        r : float
            Risk-free interest rate.
        q : float
            Continuous dividend yield.
        v0 : float
            Initial variance (squared volatility) of the underlying asset.
        kappa : float
            Rate at which variance reverts to its long-term mean (theta).
        theta : float
            Long-term mean of the variance.
        volvol : float
            Volatility of the variance process (volatility of volatility).
        rho : float
            Correlation between the asset price and its variance.
        Type : str, optional
            Type of option ('call' or 'put'), by default 'put'.
        accuracy : float, optional
            Relative tolerance for the pricing engine. Lower values increase accuracy,
            by default 1e-9.
        max_evaluations : int, optional
            Maximum number of evaluations for the pricing engine, by default 10_000.

        Returns:
        --------
        price : float
            The calculated price of the option.

        Notes:
        ------
        This function uses QuantLib's AnalyticHestonEngine to price European options under the Heston model.
        """
        # print the params
        # print(f"S0: {S0}, K: {K}, T: {T}, r: {r}, q: {q}, v0: {v0}, kappa: {kappa}, theta: {theta}, volvol: {volvol}, rho: {rho}, Type: {Type}, accuracy: {accuracy}, max_evaluations: {max_evaluations}")
        if 2 * kappa * theta < volvol**2:
            print("Warning: Feller condition violated: 2*kappa*theta < volvol**2")

        # Set evaluation date to today
        today = ql.Date().todaysDate()
        maturity = today + ql.Period(int(T * 365), ql.Days)

        # Market data
        spot = ql.QuoteHandle(ql.SimpleQuote(S0))
        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, ql.Actual365Fixed()))

        # Heston model parameters
        process = ql.HestonProcess(riskFreeTS, dividendTS, spot, v0, kappa, theta, volvol, rho)
        model = ql.HestonModel(process)

        # Decrease accuracy if max number of iterations reached
        while True:
            # Set up the AnalyticHestonEngine with specified accuracy
            engine = ql.AnalyticHestonEngine(model, accuracy, max_evaluations)

            # Define payoff and exercise
            payoff = ql.PlainVanillaPayoff(ql.Option.Call if Type.lower() == "call" else ql.Option.Put, K)
            exercise = ql.EuropeanExercise(maturity)
            option = ql.VanillaOption(payoff, exercise)

            # Set the pricing engine
            option.setPricingEngine(engine)

            # Calculate option price
            try:
                return option.NPV()
            except RuntimeError as e:
                if "max number of iterations reached" in str(e):
                    accuracy *= 10
                else:
                    raise

    def European_put_option(
        self,
        iv: np.ndarray,
        ttms: np.ndarray,
        S0: np.ndarray,
        K: np.ndarray,
        T: int = 252,
        stochastic_process: str = None,
    ):
        """
        Calculate the price and Greeks (Delta, Gamma, Vega) of a European put option under either the GBM or Heston stochastic process.

        Parameters
        ----------
        iv : np.ndarray
            Implied volatility for the option. Can be a scalar or an array matching the shape of stock prices `S0`.
        ttms : np.ndarray
            Time to maturity in days. Should match the shape of the stock prices `S0`.
        S0 : np.ndarray
            Current stock prices. Should be of the same shape as `iv` and `K`.
        K : np.ndarray
            Strike prices of the options. Should match the shape of `S0`.
        T : int, optional
            Total trading days in a year. Default is 252.
        stochastic_process : str, optional
            Type of stochastic process used, either 'GBM' or 'Heston'. Default is None.

        Returns
        -------
        price : np.ndarray
            The calculated price of the European put options.
        delta : np.ndarray
            The Delta of the European put options.
        gamma : np.ndarray
            The Gamma of the European put options.
        vega : np.ndarray
            The Vega of the European put options.

        Raises
        ------
        ValueError
            If an unsupported stochastic process is specified.

        Notes
        -----
        - For the GBM process, the Black-Scholes model is used to calculate the option price and Greeks.
        - For the Heston process, the option price and Greeks are computed using QuantLib's AnalyticHestonEngine.
        - The function returns the price and risk sensitivities (Greeks) for the specified European put options.
        - This method is integral to simulating option prices and risk profiles under different stochastic models, which is essential for portfolio management and hedging strategies.

        Examples
        --------
        >>> prices, deltas, gammas, vegas = European_put_option(iv=np.array([0.2]),
                                                                ttms=np.array([30]),
                                                                S0=np.array([100]),
                                                                K=np.array([100]),
                                                                stochastic_process='Heston')
        """
        # Validate that all input arrays have the same shape
        if not (iv.shape == ttms.shape == S0.shape == K.shape):
            raise ValueError("All input arrays (iv, ttms, S0, K) must have the same shape.")

        if stochastic_process == "GBM":
            # Return the European option price, delta, gamma, and vega using the Black-Scholes-Merton model
            return self._European_Option_GBM(iv=iv, ttm=ttms, S=S0, K=K, r=self.r, q=self.q, T=T)

        elif stochastic_process == "Heston":
            # Initialize result arrays with the same shape as S0
            dtype = np.float64
            price = np.zeros_like(S0, dtype=dtype)
            delta = np.zeros_like(S0, dtype=dtype)
            gamma = np.zeros_like(S0, dtype=dtype)
            vega = np.zeros_like(S0, dtype=dtype)

            # Compute matured option payoff
            matured_option = ttms == 0
            price = np.where(matured_option, np.maximum(K - S0, 0), price)

            # Active indices where ttm > 0
            active_indices = list(zip(*np.where(ttms > 0)))

            # Define perturbation parameters
            eps_S = 1e-5
            eps_iv = 2e-2

            # Define the function to calculate Heston option price and Greeks
            def calculate_heston_option(idx):
                opt_type = "put"
                fT = max(ttms[idx], 1) / T  # Avoid division by zero
                v0 = iv[idx] ** 2

                # Set accuracy based on self.accuracy
                accuracy = 1e-9
                if self.numerical_accuracy == "low":
                    max_evaluations = 10_000
                elif self.numerical_accuracy == "high":
                    max_evaluations = 100_000

                vol_eps = (np.sqrt(v0) * eps_iv) ** 2
                vol_eps = np.where(vol_eps < 0.00002, 0.00002, vol_eps)  # To ensure better calculation of vega

                theta_up = (np.sqrt(self.theta) + vol_eps) ** 2
                theta_down = (np.sqrt(self.theta) - vol_eps) ** 2

                v0_up = (np.sqrt(v0) + vol_eps) ** 2
                v0_down = (np.sqrt(v0) - vol_eps) ** 2

                # Base price
                C0 = self._European_Option_Heston(
                    S0=S0[idx],
                    K=K[idx],
                    T=fT,
                    r=self.r,
                    q=self.q,
                    v0=v0,
                    kappa=self.kappa,
                    theta=self.theta,
                    volvol=self.volvol,
                    rho=self.rho,
                    Type=opt_type,
                    accuracy=accuracy,
                    max_evaluations=max_evaluations,
                )

                # Perturb S0 for Delta and Gamma
                C0_S_up = self._European_Option_Heston(
                    S0=S0[idx] * (1 + eps_S),
                    K=K[idx],
                    T=fT,
                    r=self.r,
                    q=self.q,
                    v0=v0,
                    kappa=self.kappa,
                    theta=self.theta,
                    volvol=self.volvol,
                    rho=self.rho,
                    Type=opt_type,
                    accuracy=accuracy,
                    max_evaluations=max_evaluations,
                )
                C0_S_down = self._European_Option_Heston(
                    S0=S0[idx] * (1 - eps_S),
                    K=K[idx],
                    T=fT,
                    r=self.r,
                    q=self.q,
                    v0=v0,
                    kappa=self.kappa,
                    theta=self.theta,
                    volvol=self.volvol,
                    rho=self.rho,
                    Type=opt_type,
                    accuracy=accuracy,
                    max_evaluations=max_evaluations,
                )

                # Compute Delta and Gamma
                delta_val = (C0_S_up - C0_S_down) / (2 * S0[idx] * eps_S)
                gamma_val = (C0_S_up - 2 * C0 + C0_S_down) / (S0[idx] * eps_S) ** 2

                # Perturb v0 and theta for Vega
                C0_iv_up = self._European_Option_Heston(
                    S0=S0[idx],
                    K=K[idx],
                    T=fT,
                    r=self.r,
                    q=self.q,
                    v0=v0_up,
                    kappa=self.kappa,
                    theta=theta_up,
                    volvol=self.volvol,
                    rho=self.rho,
                    Type=opt_type,
                    accuracy=accuracy,
                    max_evaluations=max_evaluations,
                )
                C0_iv_down = self._European_Option_Heston(
                    S0=S0[idx],
                    K=K[idx],
                    T=fT,
                    r=self.r,
                    q=self.q,
                    v0=v0_down,
                    kappa=self.kappa,
                    theta=theta_down,
                    volvol=self.volvol,
                    rho=self.rho,
                    Type=opt_type,
                    accuracy=accuracy,
                    max_evaluations=max_evaluations,
                )

                # Compute Vega
                vega_val = (1 / 100) * (C0_iv_up - C0_iv_down) / (2 * vol_eps)
                # vega_val = (1 / 100) * (C0_iv_up - C0) / vol_eps
                return idx, C0, delta_val, gamma_val, vega_val

            # Parallel processing for Heston using only active indices
            results = Parallel(
                n_jobs=self.n_jobs,
                backend=self._backend,
                pre_dispatch="1*n_jobs",
                batch_size=100,
                timeout=120,
                mmap_mode="r",
                temp_folder="/home/atsoskouno/data/storageith2/gamma-vega-hedging-American-Heston/.temp_dir_loky",
                # inner_max_num_threads=1,
            )(delayed(calculate_heston_option)(idx) for idx in active_indices)

            # Assign results to the respective arrays
            for result in results:
                idx, C0, delta_val, gamma_val, vega_val = result
                price[idx] = C0
                delta[idx] = delta_val
                gamma[idx] = gamma_val
                vega[idx] = vega_val

            return price, delta, gamma, vega

        else:
            raise ValueError(f"Unsupported stochastic process: {stochastic_process}")

    def agg_poisson_dist(self, a_prices, vol):
        """
        Generate and aggregate Poisson-arrival options' prices and risk profiles to simulate a liability portfolio.

        This function simulates the generation of a liability portfolio where options arrive according to a Poisson process.
        It calculates and aggregates the prices and risk profiles (delta, gamma, vega) of these options over time, reflecting
        the stochastic nature of a dynamic portfolio. The aggregated portfolio is then used for further hedging and trading
        strategy analysis.

        Parameters
        ----------
        a_prices : np.ndarray
            Simulated underlying asset prices with shape (num_episodes, num_steps). This represents the asset price paths
            across different simulation episodes and time steps.
        vol : np.ndarray
            Simulated volatilities, either as a constant for the Black-Scholes model or as a time-varying array
            (num_episodes, num_steps) for more complex models like Heston.

        Returns
        -------
        np.ndarray
            An array of `SyntheticOption` objects, each representing the aggregated portfolio's price and risk profiles
            (delta, gamma, vega) for a given episode. The shape of the array is (num_episodes,).

        Notes
        -----
        - The Poisson process models the random arrival of new options into the portfolio. The intensity of the arrivals
        is controlled by `poisson_rate`.
        - For each time step, the function calculates the portfolio's price and risk profiles based on the current and
        previously accumulated options.
        - The strikes of the options are determined by the current asset prices and a randomized moneyness factor.
        - The function supports both GBM (Geometric Brownian Motion) and Heston stochastic processes, applying the
        appropriate pricing models for the generated options.
        - The methodology follows the approach outlined in "Gamma and vega hedging using deep distributional reinforcement
        learning" (Cao et al., 2023), where dynamic hedging strategies are applied to manage the risks of stochastically
        arriving options.

        Example
        -------
        utils = Utils(S0=100, K=100, ttms=[120], r=0.0, q=0.0, spread=0.0, poisson_rate=1.0,
                    hed_ttm=60, hed_type='European', init_vol=0.02, kappa=0.1, theta=0.02, volvol=0.2, rho=-0.5,
                    stochastic_process='Heston', time_to_simulate=252, num_sim=1000, frq=1, n_jobs=4, np_seed=1234)

        a_price, vol = utils.init_env()
        portfolio_options = utils.agg_poisson_dist(a_price, vol)

        References
        ----------
        Cao, J., Chen, J., Farghadani, S., Hull, J., Poulos, Z., Wang, Z., & Yuan, J. (2023). Gamma and vega hedging using
        deep distributional reinforcement learning. Frontiers in Artificial Intelligence, 6, 1129370.
        """
        # print(f'>>Inside agg_poisson_dist: {self.stochastic_process}')
        print("Genrate Poisson arrival portfolio option prices and risk profiles")
        options = []
        num_opts_per_day = np.random.poisson(self.poisson_rate, a_prices.shape)
        num_episode = a_prices.shape[0]
        num_step = a_prices.shape[1]
        max_num_opts = num_opts_per_day.max(axis=0)  # maximum number of options for each time step

        # generate portfolio price & risk profiles for each step
        # step dimension is the smallest size for a loop
        all_option_ttms = None  # all options ttms
        all_option_strikes = None  # all options strikes
        all_option_buysell = None  # all options buy/sell position
        port_price = next_port_price = port_delta = port_gamma = port_vega = None
        for step_i in tqdm(range(num_step)):
            if all_option_ttms is not None:
                # decrease ttm of accumulated options
                all_option_ttms -= 1
            # options shape for calculation is (num_episode, max_num_opts at step_i)
            if step_i == 0:
                ep_num_opts = np.ones(num_episode)
                step_num_opts = 1
            else:
                ep_num_opts = num_opts_per_day[:, step_i]
                step_num_opts = max_num_opts[step_i]
            step_a_prices = a_prices[:, step_i]
            # randomize option time to maturities by selecting from self.ttms
            option_ttms = np.random.choice(
                self.ttms, (num_episode, step_num_opts)
            )  # Default value for self.ttms is [120]
            # flush non-existing option's ttm to -1,
            # so option's price and risk profile will be calculated as 0 from bs_call
            option_ttms[ep_num_opts[:, None] <= np.arange(option_ttms.shape[1])] = -1
            # print(f'({step_i}) option_ttms: {option_ttms}')
            # randomize option strikes
            moneyness = np.random.normal(self.moneyness_mean, self.moneyness_std, (num_episode, step_num_opts))
            option_strikes = step_a_prices[:, None] * moneyness  # ATM
            # randomize buy or sell equal likely
            if step_i == 0:
                option_ttms = np.full((num_episode, 1), self.init_ttm, dtype=a_prices.dtype)
                option_strikes = np.full((num_episode, 1), self.K, dtype=a_prices.dtype)
                # step 0 - always underwrite one option
                option_buysell = np.ones((num_episode, step_num_opts), dtype=a_prices.dtype)

            else:
                option_buysell = np.random.choice([-1.0, 1.0], (num_episode, step_num_opts))
            # add this step's new options
            if all_option_ttms is not None:
                all_option_ttms = np.c_[all_option_ttms, option_ttms]
                all_option_strikes = np.c_[all_option_strikes, option_strikes]
                all_option_buysell = np.c_[all_option_buysell, option_buysell]
            else:
                all_option_ttms = option_ttms
                all_option_strikes = option_strikes
                all_option_buysell = option_buysell
            # vol
            if vol.ndim == 0:
                step_vol = vol
            else:
                step_vol = np.tile(np.expand_dims(vol[:, step_i], -1), (1, all_option_ttms.shape[1]))
            # expand stock price to (num_episode, step_num_opts)
            step_a_prices = np.tile(np.expand_dims(step_a_prices, -1), (1, all_option_ttms.shape[1]))
            # price and risk profiles
            step_port_price, step_port_delta, step_port_gamma, step_port_vega = self.American_put_option(
                iv=step_vol,
                ttms=all_option_ttms,
                S0=step_a_prices,
                K=all_option_strikes,
                T=self.TradingDaysPerYear,
                stochastic_process=self.stochastic_process,
            )

            step_port_price *= all_option_buysell
            step_port_delta *= all_option_buysell
            step_port_gamma *= all_option_buysell
            step_port_vega *= all_option_buysell
            if step_i > 0:
                if step_num_opts > 0:
                    step_next_port_price = step_port_price[:, :(-step_num_opts)].sum(
                        axis=1
                    )  # only consider the positions from last step
                else:
                    step_next_port_price = step_port_price.sum(axis=1)
            step_port_price = step_port_price.sum(axis=1)
            step_port_delta = step_port_delta.sum(axis=1)
            step_port_gamma = step_port_gamma.sum(axis=1)
            step_port_vega = step_port_vega.sum(axis=1)
            if step_i > 0:
                if step_i > 1:
                    # greater than step 2, concatenate
                    next_port_price = np.c_[next_port_price, step_next_port_price[:, None]]
                else:
                    # at step 2, initialize next port price
                    next_port_price = step_next_port_price[:, None]
                port_price = np.c_[port_price, step_port_price[:, None]]
                port_delta = np.c_[port_delta, step_port_delta[:, None]]
                port_gamma = np.c_[port_gamma, step_port_gamma[:, None]]
                port_vega = np.c_[port_vega, step_port_vega[:, None]]
            else:
                port_price = step_port_price[:, None]
                port_delta = step_port_delta[:, None]
                port_gamma = step_port_gamma[:, None]
                port_vega = step_port_vega[:, None]

        print("Initialize Poisson arrival liability portfolio options.")
        for ep_i in tqdm(range(num_episode)):
            options.append(
                SyntheticOption(
                    port_price[ep_i, :],
                    next_port_price[ep_i, :],
                    port_delta[ep_i, :],
                    port_gamma[ep_i, :],
                    port_vega[ep_i, :],
                    np.zeros((num_step,)).astype(np.bool_),
                    self.num_conts_to_add,
                    self.contract_size,
                )
            )

        return np.array(options)

    def atm_hedges(self, a_prices, vol):
        """
        Generate at-the-money (ATM) hedging options' prices and risk profiles for each simulation path.

        This function is responsible for creating a series of ATM options at every time step in a simulation, which are then used to hedge the portfolio's gamma and vega risks. The methodology follows a dynamic hedging approach where new options are initiated at each step, considering the current underlying asset price and volatility.

        Parameters
        ----------
        a_prices : np.ndarray
            Simulated underlying asset prices with shape (num_episodes, num_steps).
        vol : np.ndarray
            Simulated volatilities, which can either be a constant (for a constant volatility model) or an array of shape (num_episodes, num_steps) for stochastic volatility models.

        Returns
        -------
        np.ndarray
            A list of hedging `Option` objects in shape (num_episodes, num_hedges), where num_hedges equals num_steps, representing the series of ATM options created at each step.

        Notes
        -----
        - This function is used for dynamic gamma and vega hedging by generating ATM options at each time step.
        - The hedging options are priced based on the current underlying asset price and volatility, using either the Black-Scholes-Merton model for European options or a binomial tree model for American options, depending on the `hed_type`.
        - The function supports both European and American options for hedging, as specified by the `hed_type` parameter during initialization.
        - The implementation dynamically adjusts the hedge by recalculating the option's price, delta, gamma, and vega at each step, considering the potential expiration of options.
        - The hedging strategy is evaluated considering transaction costs, as discussed in the context of dynamic hedging strategies and their effectiveness in managing portfolio risks.

        **Usage:**
        This function is typically called within a trading simulation environment to ensure that the portfolio remains hedged against gamma and vega risks over the simulation horizon.
        """
        print("Generate hedging portfolio option prices and risk profiles")
        hedge_ttm = self.hed_ttm
        num_episode = a_prices.shape[0]
        num_hedge = num_step = a_prices.shape[1]
        price = np.empty((num_episode, num_step, num_hedge), dtype=float)
        delta = np.empty_like(price, dtype=float)
        gamma = np.empty_like(price, dtype=float)
        vega = np.empty_like(price, dtype=float)
        inactive_option = np.empty_like(price, dtype=np.bool_)

        # generate portfolio price & risk profiles for each step
        # step dimension is the smallest size for a loop
        all_option_ttms = None  # all options ttms
        all_option_strikes = None  # all options strikes

        pricing_counter = 0
        for step_i in tqdm(range(num_step)):
            if all_option_ttms is not None:
                # decrease ttm of accumulated options
                all_option_ttms -= 1
            # new options shape for calculation is (num_episode, )
            step_a_prices = a_prices[:, step_i]
            option_ttms = hedge_ttm * np.ones((num_episode,))
            option_strikes = step_a_prices
            # add this step's new options
            if all_option_ttms is not None:
                all_option_ttms = np.c_[all_option_ttms, option_ttms[:, None]]
                all_option_strikes = np.c_[all_option_strikes, option_strikes[:, None]]
            else:
                all_option_ttms = option_ttms[:, None]
                all_option_strikes = option_strikes[:, None]
            # vol
            if vol.ndim == 0:
                step_vol = vol
            else:
                step_vol = np.tile(np.expand_dims(vol[:, step_i], -1), (1, all_option_ttms.shape[1]))
            # expand stock price to (num_episode, step_num_opts)
            step_a_prices = np.tile(np.expand_dims(step_a_prices, -1), (1, all_option_ttms.shape[1]))

            # price and risk profiles of hedging options
            if self.hed_type == "American":
                step_option_price, step_option_delta, step_option_gamma, step_option_vega = (
                    self.American_put_option(
                        iv=step_vol,
                        ttms=all_option_ttms,
                        S0=step_a_prices,
                        K=all_option_strikes,
                        T=self.TradingDaysPerYear,
                        stochastic_process=self.stochastic_process,
                    )
                )
                # self.bs_call(step_vol, all_option_ttms, step_a_prices, all_option_strikes, self.r, self.q, self.T)
            if self.hed_type == "European":
                step_option_price, step_option_delta, step_option_gamma, step_option_vega = (
                    self.European_put_option(
                        iv=step_vol,
                        ttms=all_option_ttms,
                        S0=step_a_prices,
                        K=all_option_strikes,
                        T=self.TradingDaysPerYear,
                        stochastic_process=self.stochastic_process,
                    )
                )

            pricing_counter += step_vol.shape[0]
            for option_i in range(num_hedge):
                if (option_i > step_i) or (all_option_ttms[:, option_i].mean() < 0):
                    # option is not initiated or option expired
                    inactive_option[:, step_i, option_i] = True
                    price[:, step_i, option_i] = 0
                    delta[:, step_i, option_i] = 0
                    gamma[:, step_i, option_i] = 0
                    vega[:, step_i, option_i] = 0
                else:
                    # option expired
                    inactive_option[:, step_i, option_i] = False
                    price[:, step_i, option_i] = step_option_price[:, option_i]
                    delta[:, step_i, option_i] = step_option_delta[:, option_i]
                    gamma[:, step_i, option_i] = step_option_gamma[:, option_i]
                    vega[:, step_i, option_i] = step_option_vega[:, option_i]

        # construct options in shape (num_sim, num_hedge)
        print("Initialize hedging portfolio options.")
        options = []
        for ep_i in tqdm(range(num_episode)):
            options.append([])
            for option_i in range(num_hedge):
                options[-1].append(
                    Option(
                        price[ep_i, :, option_i],
                        delta[ep_i, :, option_i],
                        gamma[ep_i, :, option_i],
                        vega[ep_i, :, option_i],
                        inactive_option[ep_i, :, option_i],
                        0,
                        self.contract_size,
                    )
                )
            # Debugging
            if ep_i == -1:
                for _ in range(len(options[-1])):
                    print(f"({_}) options[-1].gamma_path: {options[-1][_].gamma_path}")
        print(f"priced options: {pricing_counter}")
        return np.array(options)
