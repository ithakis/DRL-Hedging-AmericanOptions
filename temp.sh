#!/bin/bash
#  nohup "./temp.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt

# Set common parameters
NUM_TRAIN_SIM=32768    #32768 49152
NUM_EVAL_SIM=4096      #4096 or 8192

# Cao's Approximate Heston Parameters
INIT_VOL=0.3 # changed from 0.0
KAPPA=1.0
THETA=0.09
VOLVOL=0.3
RHO=-0.7

args_Greeks=(
    # Utility Function Parameters
    -init_ttm=30 -r=0.0 #-spread=0.005
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
    -stochastic_process='Heston' -time_to_simulate=30 -eval_sim=$NUM_EVAL_SIM -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=-1 -eval_seed=3456

    # RL Agent Parameters
    -vega_obs=True #-critic='qr-huber'

    # -logger_prefix='Cao_Heston_Delta_05pct'

    # Load Portfolio
    -portfolio_folder='portfolios/.Portfolio_01-11-2024_11-45'
)
####################################################################################################################################################
# Delta Run #########################################################################################################################################
python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='Heston_Delta_05pct' -spread=0.005 &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='Cao_Heston_Delta_1pct' -portfolio_folder='portfolios/.Portfolio_Heston_1'   -spread=0.01  &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='Cao_Heston_Delta_2pct' -portfolio_folder='portfolios/.Portfolio_Heston_2'   -spread=0.02  &

####################################################################################################################################################
# Delta-Gamma Run ###################################################################################################################################
python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='Heston_Gamma_05pct' -spread=0.005 &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='Cao_Heston_Gamma_1pct' -portfolio_folder='portfolios/.Portfolio_Heston_1'   -spread=0.01  &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='Cao_Heston_Gamma_2pct' -portfolio_folder='portfolios/.Portfolio_Heston_2'   -spread=0.02  

####################################################################################################################################################
# Delta-Gamma Run ###################################################################################################################################
python greek_runAlexander.py "${args_Greeks[@]}" -strategy='vega' -logger_prefix='Heston_Vega_05pct' -spread=0.005 
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='Cao_Heston_Gamma_1pct' -portfolio_folder='portfolios/.Portfolio_Heston_1'   -spread=0.01  &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='Cao_Heston_Gamma_2pct' -portfolio_folder='portfolios/.Portfolio_Heston_2'   -spread=0.02  