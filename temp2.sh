#!/bin/bash
#  nohup "./temp2.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt
# [2] 3133
# [2]-  Exit 2                  nohup "./temp.sh" > .terminal_output.txt 2>&1

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
THETA=0.09
VOLVOL=0.3
RHO=-0.7


# FOLDER='portfolios/.Portfolio_14-58_24-10-2024'

# # Mean-Std Experiment (Spread: 0.005) ###############################################################################################################
# args_Mean_Std=(
#     # Utility Function Parameters
#     -init_ttm=60 -r=0.0 -spread=0.005
#     -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

#     # Contract Parameters
#     -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

#     # Hedging Portfolio Parameters
#     -hed_ttm=30 -hed_type='European'

#     # init_vol is for both GBM and Heston
#     -init_vol=$INIT_VOL

#     # Heston, Model Parameters
#     -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

#     # Simulation Parameters
#     -stochastic_process='GBM' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

#     -logger_prefix='Cao_GBM_Mean_Std_05pct'
#     -obj_func='meanstd' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=2345
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# # python runAlexander.py "${args_Mean_Std[@]}"

# # VaR Experiment (Spread: 0.005) ###############################################################################################################
# args_VaR=(
#     # Utility Function Parameters
#     -init_ttm=60 -r=0.0 -spread=0.005
#     -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

#     # Contract Parameters
#     -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

#     # Hedging Portfolio Parameters
#     -hed_ttm=30 -hed_type='European'

#     # init_vol is for both GBM and Heston
#     -init_vol=$INIT_VOL

#     # Heston, Model Parameters
#     -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

#     # Simulation Parameters
#     -stochastic_process='GBM' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

#     -logger_prefix='Cao_GBM_VaR_1pct'
#     -obj_func='var' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=2345
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# # python runAlexander.py "${args_VaR[@]}"

# # CVaR Experiment (Spread: 0.005) ###############################################################################################################
# args_CVaR=(
#     # Utility Function Parameters
#     -init_ttm=60 -r=0.0 -spread=0.005
#     -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

#     # Contract Parameters
#     -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

#     # Hedging Portfolio Parameters
#     -hed_ttm=30 -hed_type='European'

#     # init_vol is for both GBM and Heston
#     -init_vol=$INIT_VOL

#     # Heston, Model Parameters
#     -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

#     # Simulation Parameters
#     -stochastic_process='GBM' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

#     -logger_prefix='Cao_GBM_CVaR_1pct'
#     -obj_func='cvar' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=2345
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# # python runAlexander.py "${args_CVaR[@]}"

# #####################################################################################################################################################
# # Run Both the last 2 at the same time as the data is generated from the first
# python runAlexander.py "${args_Mean_Std[@]}" & python runAlexander.py "${args_VaR[@]}" & python runAlexander.py "${args_CVaR[@]}" 


# #####################################################################################################################################################
# #####################################################################################################################################################
# #####################################################################################################################################################

# FOLDER='portfolios/.Portfolio_16-17_24-10-2024'

# # Mean-Std Experiment (Spread: 0.01) ###############################################################################################################
# args_Mean_Std=(
#     # Utility Function Parameters
#     -init_ttm=60 -r=0.0 -spread=0.01
#     -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

#     # Contract Parameters
#     -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

#     # Hedging Portfolio Parameters
#     -hed_ttm=30 -hed_type='European'

#     # init_vol is for both GBM and Heston
#     -init_vol=$INIT_VOL

#     # Heston, Model Parameters
#     -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

#     # Simulation Parameters
#     -stochastic_process='GBM' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

#     -logger_prefix='Cao_GBM_Mean_Std_1pct'
#     -obj_func='meanstd' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=2345
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# # VaR Experiment (Spread: 0.01) ###############################################################################################################
# args_VaR=(
#     # Utility Function Parameters
#     -init_ttm=60 -r=0.0 -spread=0.01
#     -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

#     # Contract Parameters
#     -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

#     # Hedging Portfolio Parameters
#     -hed_ttm=30 -hed_type='European'

#     # init_vol is for both GBM and Heston
#     -init_vol=$INIT_VOL

#     # Heston, Model Parameters
#     -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

#     # Simulation Parameters
#     -stochastic_process='GBM' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

#     -logger_prefix='Cao_GBM_VaR_1pct'
#     -obj_func='var' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=2345
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# # python runAlexander.py "${args_VaR[@]}"

# # CVaR Experiment (Spread: 0.01) ###############################################################################################################
# args_CVaR=(
#     # Utility Function Parameters
#     -init_ttm=60 -r=0.0 -spread=0.01
#     -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

#     # Contract Parameters
#     -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

#     # Hedging Portfolio Parameters
#     -hed_ttm=30 -hed_type='European'

#     # init_vol is for both GBM and Heston
#     -init_vol=$INIT_VOL

#     # Heston, Model Parameters
#     -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

#     # Simulation Parameters
#     -stochastic_process='GBM' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

#     -logger_prefix='Cao_GBM_CVaR_1pct'
#     -obj_func='cvar' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=2345
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# # python runAlexander.py "${args_CVaR[@]}"

# #####################################################################################################################################################
# # Run Both the last 2 at the same time as the data is generated from the first
# python runAlexander.py "${args_Mean_Std[@]}" & python runAlexander.py "${args_VaR[@]}" & python runAlexander.py "${args_CVaR[@]}" 


#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################

FOLDER='portfolios/.Portfolio_18-16_24-10-2024'
# Mean-Std Experiment (Spread: 0.02) ###############################################################################################################
args_Mean_Std=(
    # Utility Function Parameters
    -init_ttm=60 -r=0.0 -spread=0.02
    -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

    # Contract Parameters
    -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

    # Hedging Portfolio Parameters
    -hed_ttm=30 -hed_type='European'

    # init_vol is for both GBM and Heston
    -init_vol=$INIT_VOL

    # Heston, Model Parameters
    -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

    # Simulation Parameters
    -stochastic_process='GBM' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

    # RL Environment Parameters
    -action_space="0,1"

    # RL Agent Parameters
    -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

    -logger_prefix='Cao_GBM_Mean_Std_2pct'
    -obj_func='meanstd' -std_coef=1.645
    -eval_sim=$NUM_EVAL_SIM -eval_seed=2345
    -eval_only=False

    # Load Portfolio
    -portfolio_folder=$FOLDER
)

# python runAlexander.py "${args_Mean_Std[@]}"

# VaR Experiment (Spread: 0.02) ###############################################################################################################
args_VaR=(
    # Utility Function Parameters
    -init_ttm=60 -r=0.0 -spread=0.02
    -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

    # Contract Parameters
    -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

    # Hedging Portfolio Parameters
    -hed_ttm=30 -hed_type='European'

    # init_vol is for both GBM and Heston
    -init_vol=$INIT_VOL

    # Heston, Model Parameters
    -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

    # Simulation Parameters
    -stochastic_process='GBM' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

    # RL Environment Parameters
    -action_space="0,1"

    # RL Agent Parameters
    -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

    -logger_prefix='Cao_GBM_VaR_2pct'
    -obj_func='var' -std_coef=1.645
    -eval_sim=$NUM_EVAL_SIM -eval_seed=2345
    -eval_only=False

    # Load Portfolio
    -portfolio_folder=$FOLDER
)

# python runAlexander.py "${args_VaR[@]}"

# CVaR Experiment (Spread: 0.02) ###############################################################################################################
args_CVaR=(
    # Utility Function Parameters
    -init_ttm=60 -r=0.0 -spread=0.02
    -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

    # Contract Parameters
    -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

    # Hedging Portfolio Parameters
    -hed_ttm=30 -hed_type='European'

    # init_vol is for both GBM and Heston
    -init_vol=$INIT_VOL

    # Heston, Model Parameters
    -kappa=$KAPPA -theta=$THETA -volvol=$VOLVOL -rho=$RHO

    # Simulation Parameters
    -stochastic_process='GBM' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

    # RL Environment Parameters
    -action_space="0,1"

    # RL Agent Parameters
    -n_step=5 -batch_size=512 -vega_obs=False -critic='qr-huber'

    -logger_prefix='Cao_GBM_CVaR_2pct'
    -obj_func='cvar' -std_coef=1.645
    -eval_sim=$NUM_EVAL_SIM -eval_seed=2345
    -eval_only=False

    # Load Portfolio
    -portfolio_folder=$FOLDER
)

# python runAlexander.py "${args_CVaR[@]}"

#####################################################################################################################################################
# Run Both the last 2 at the same time as the data is generated from the first
python runAlexander.py "${args_Mean_Std[@]}" & python runAlexander.py "${args_VaR[@]}" & python runAlexander.py "${args_CVaR[@]}" 