#!/bin/bash
#  nohup "./(EXP 1) Cao GBM Experiments.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt




# Set common parameters
NUM_TRAIN_SIM=32768    #32768 49152
NUM_EVAL_SIM=4096      #4096

# My Heston Parameters
# INIT_VOL=0.0228
# KAPPA=0.0807
# THETA=0.363
# VOLVOL=0.1760
# RHO=-0.3021

# Cao's Approximate Heston Parameters
INIT_VOL=0.3 # changed from 0.0
KAPPA=0.5
THETA=0.09 # must be the square of the init_vol
VOLVOL=0.3
RHO=-0.7

FOLDER='portfolios/.Portfolio_GBM_2_Am'

# # Mean-Std Experiment (Spread: 0.005) ###############################################################################################################
# args_Mean_Std=(
#     # Utility Function Parameters
#     -spread=0.01 -obj_func='meanstd' -stochastic_process='GBM'

#     -init_ttm=30 -r=0.0 -spread=0.02
#     -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

#     # Contract Parameters
#     -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

#     # Hedging Portfolio Parameters
#     -hed_ttm=30 -hed_type='American'

#     # init_vol is for both GBM and Heston
#     -init_vol=$INIT_VOL

#     # Heston, Model Parameters
#     -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

#     # Simulation Parameters
#     # -stochastic_process='GBM'
#     -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

#     -logger_prefix='(EXP 1) GBM Experiments/spread_2/GBM_MeanStd_2'
#     -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=2345
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )
# python runAlexander.py "${args_Mean_Std[@]}" &

# # # Get the most recently modified hidden folder inside ./portfolios ###################################################################################
# # [ -d "./portfolios" ] || { echo "Error: Directory ./portfolios does not exist."; exit 1; }
# # echo -e "\n>>Retrieving the most recently modified hidden folder inside ./portfolios..."
# # last_created_folder=$(ls -td ./portfolios/.[!.]*/ 2>/dev/null | head -1)
# # [ -n "$last_created_folder" ] && echo "Last modified hidden folder: $(basename "$last_created_folder")" || echo "No hidden subdirectories found in ./portfolios."
# # echo

# # VaR Experiment (Spread: 0.005) ###############################################################################################################
# args_VaR=(
#     # Utility Function Parameters
#     -init_ttm=30 -r=0.0 -spread=0.02
#     -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

#     # Contract Parameters
#     -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

#     # Hedging Portfolio Parameters
#     -hed_ttm=30 -hed_type='American'

#     # init_vol is for both GBM and Heston
#     -init_vol=$INIT_VOL

#     # Heston, Model Parameters
#     -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

#     # Simulation Parameters
#     -stochastic_process='GBM' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=3456

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

#     -logger_prefix='(EXP 1) GBM Experiments/spread_2/GBM_VaR_2'
#     -obj_func='var' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=4567
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# # python runAlexander.py "${args_VaR[@]}"

# # CVaR Experiment (Spread: 0.005) ###############################################################################################################
# args_CVaR=(
#     # Utility Function Parameters
#     -init_ttm=30 -r=0.0 -spread=0.02
#     -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

#     # Contract Parameters
#     -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

#     # Hedging Portfolio Parameters
#     -hed_ttm=30 -hed_type='American'

#     # init_vol is for both GBM and Heston
#     -init_vol=$INIT_VOL

#     # Heston, Model Parameters
#     -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

#     # Simulation Parameters
#     -stochastic_process='GBM' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=5678

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

#     -logger_prefix='(EXP 1) GBM Experiments/spread_2/GBM_CVaR_2'
#     -obj_func='cvar' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=6789
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# # python runAlexander.py "${args_CVaR[@]}"

# #####################################################################################################################################################
# # Run Both the last 2 at the same time as the data is generated from the first
# python runAlexander.py "${args_VaR[@]}" & python runAlexander.py "${args_CVaR[@]}" &


#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
# Greek Run ###########################################################################################################################################
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
    -stochastic_process='GBM' -time_to_simulate=30 -eval_sim=$NUM_EVAL_SIM -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=-1 -eval_seed=1234

    # RL Agent Parameters
    -vega_obs=False #-critic='qr-huber'

    # -logger_prefix='Cao_GBM_Delta_05pct'

    # Load Portfolio
    -portfolio_folder=$FOLDER
)
# #####################################################################################################################################################
# Delta Run #########################################################################################################################################
python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='(EXP 1) GBM Experiments/spread_2/GBM_Delta_2' -spread=0.02 &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='Cao_GBM_Delta_1pct' -portfolio_folder='portfolios/.Portfolio_GBM_1'   -spread=0.01  &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='Cao_GBM_Delta_2pct' -portfolio_folder='portfolios/.Portfolio_GBM_2'   -spread=0.02  &

#####################################################################################################################################################
# Delta-Gamma Run ###################################################################################################################################
python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='(EXP 1) GBM Experiments/spread_2/GBM_Gamma_2' -spread=0.02 
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='Cao_GBM_Gamma_1pct' -portfolio_folder='portfolios/.Portfolio_GBM_1'   -spread=0.01  &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='Cao_GBM_Gamma_2pct' -portfolio_folder='portfolios/.Portfolio_GBM_2'   -spread=0.02  