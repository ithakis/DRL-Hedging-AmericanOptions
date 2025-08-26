#!/bin/bash
#  nohup "/home/atsoskouno/data/main_storage/gamma-vega-hedging-American-Heston/Robustness Testing.sh" > .terminal_outputR.txt 2>&1 & 
#  tail -f .terminal_outputR.txt

# nohup "/home/atsoskouno/data/main_storage/gamma-vega-hedging-American-Heston/Robustness Testing.sh" > .terminal_outputR.txt 2>&1 & tail -f .terminal_outputR.txt
# [1] 57863

# Path to trained agent (update this with actual path)
AGENT_PATH="./logs/(EXP 2.1) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_45/stochastic_process=Heston_spread=0.02_obj=cvar_threshold=0.95_critic=qr-huber_hedttm=30"

# Original Heston parameters
INIT_VOL=0.3
KAPPA=1.0
THETA=0.09
VOLVOL=0.3
RHO=-0.7

# Common parameters
NUM_EVAL_SIM=4096
EVALUATION_LOG_PREFIX="Robustness_Testing"

args=(
    # Utility Function Parameters
    -init_ttm=30 -r=0.0 -spread=0.02
    -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

    # Contract Parameters
    -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

    # Hedging Portfolio Parameters
    -hed_ttm=30 -hed_type='American'

    # init_vol is for both GBM and Heston
    -init_vol=$INIT_VOL

    # Heston, Model Parameters
    -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

    # Simulation Parameters
    -stochastic_process='Heston' -time_to_simulate=30 -train_sim=64 -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=55 -train_seed=1234

    # RL Environment Parameters
    -action_space="0,1"

    # RL Agent Parameters
    -n_step=5 -batch_size=1024 -vega_obs=True -critic='qr-huber'

    # -logger_prefix='(EXP 2) Heston Experiments/spread_05/Heston_Mean_Std_05'
    -obj_func='cvar' -std_coef=1.645
    -eval_sim=$NUM_EVAL_SIM -eval_seed=654
    -eval_only=True
    
    # IMPORTANT: Add this line to properly pass the agent path
    -agent_path="$AGENT_PATH"

    # Load Portfolio
    -portfolio_folder="$PORTFOLIO_FOLDER"
)

# # # Test 0: Volatility of volatility : 0.1
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_volvol_0.1"
# python runAlexander.py "${args[@]}" -volvol=0.10 -logger_prefix="$EVALUATION_LOG_PREFIX/volvol_0.1"

# # Test 1.1: Volatility of volatility : 0.15
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_volvol_0.15"
# python runAlexander.py "${args[@]}" -volvol=0.15 -logger_prefix="$EVALUATION_LOG_PREFIX/volvol_0.15"

# # Test 1.2: Volatility of volatility : 0.25
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_volvol_0.25"
# python runAlexander.py "${args[@]}" -volvol=0.25 -logger_prefix="$EVALUATION_LOG_PREFIX/volvol_0.25"

# # Test 1.3: Volatility of volatility : 0.35
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_volvol_0.35"
# python runAlexander.py "${args[@]}" -volvol=0.35 -logger_prefix="$EVALUATION_LOG_PREFIX/volvol_0.35"

# # Test 1.4: Volatility of volatility : 0.45
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_volvol_0.45"
# python runAlexander.py "${args[@]}" -volvol=0.45 -logger_prefix="$EVALUATION_LOG_PREFIX/volvol_0.45"


# # Test 2.1: Correlation : -0.85
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_rho_-0.85"
# python runAlexander.py "${args[@]}" -rho=-0.85 -logger_prefix="$EVALUATION_LOG_PREFIX/rho_-0.85"

# # Test 2.2: Correlation : -0.75
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_rho_-0.75"
# python runAlexander.py "${args[@]}" -rho=-0.75 -logger_prefix="$EVALUATION_LOG_PREFIX/rho_-0.75"

# # Test 2.3: Correlation : -0.65
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_rho_-0.65"
# python runAlexander.py "${args[@]}" -rho=-0.65 -logger_prefix="$EVALUATION_LOG_PREFIX/rho_-0.65"

# # Test 2.4: Correlation : -0.55
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_rho_-0.55"
# python runAlexander.py "${args[@]}" -rho=-0.55 -logger_prefix="$EVALUATION_LOG_PREFIX/rho_-0.55"


# # Test 3.1: Initial Volatility : 0.15
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_init_vol_0.15"
# python runAlexander.py "${args[@]}" -init_vol=0.15 -logger_prefix="$EVALUATION_LOG_PREFIX/init_vol_0.15"

# # Test 3.2: Initial Volatility : 0.25
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_init_vol_0.25"
# python runAlexander.py "${args[@]}" -init_vol=0.25 -logger_prefix="$EVALUATION_LOG_PREFIX/init_vol_0.25"

# # Test 3.3: Initial Volatility : 0.35
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_init_vol_0.35"
# python runAlexander.py "${args[@]}" -init_vol=0.35 -logger_prefix="$EVALUATION_LOG_PREFIX/init_vol_0.35"

# # Test 3.4: Initial Volatility : 0.45
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_init_vol_0.45"
# python runAlexander.py "${args[@]}" -init_vol=0.45 -logger_prefix="$EVALUATION_LOG_PREFIX/init_vol_0.45"


# # Test 4.1: Long-term Volatility (theta) : 0.045
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_theta_0.045"
# python runAlexander.py "${args[@]}" -theta=0.045 -logger_prefix="$EVALUATION_LOG_PREFIX/theta_0.045"

# # Test 4.2: Long-term Volatility (theta) : 0.075
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_theta_0.075"
# python runAlexander.py "${args[@]}" -theta=0.075 -logger_prefix="$EVALUATION_LOG_PREFIX/theta_0.075"

# # Test 4.3: Long-term Volatility (theta) : 0.105
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_theta_0.105"
# python runAlexander.py "${args[@]}" -theta=0.105 -logger_prefix="$EVALUATION_LOG_PREFIX/theta_0.105"

# # Test 4.4: Long-term Volatility (theta) : 0.135
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_theta_0.135"
# python runAlexander.py "${args[@]}" -theta=0.135 -logger_prefix="$EVALUATION_LOG_PREFIX/theta_0.135"


# # Test 5.1: Mean Reversion Speed (kappa) : 0.65
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_kappa_0.65"
# python runAlexander.py "${args[@]}" -kappa=0.65 -logger_prefix="$EVALUATION_LOG_PREFIX/kappa_0.65"

# Test 5.2: Mean Reversion Speed (kappa) : 0.9
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_kappa_0.9"
# python runAlexander.py "${args[@]}" -kappa=0.9 -logger_prefix="$EVALUATION_LOG_PREFIX/kappa_0.9"

# Test 5.3: Mean Reversion Speed (kappa) : 1.1
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_kappa_1.1"
# python runAlexander.py "${args[@]}" -kappa=1.1 -logger_prefix="$EVALUATION_LOG_PREFIX/kappa_1.1"

# Test 5.4: Mean Reversion Speed (kappa) : 1.35
# PORTFOLIO_FOLDER="portfolios/.Robustness_Testing_kappa_1.35"
# python runAlexander.py "${args[@]}" -kappa=1.35 -logger_prefix="$EVALUATION_LOG_PREFIX/kappa_1.35"