#!/bin/bash
#  nohup "./(EXP 4) Heston Test.sh" > .terminal_output4.txt 2>&1 &
#  tail -f .terminal_output4.txt
#
# or:  bash "./(EXP 4) Heston Test.sh"

# Set common parameters
NUM_TRAIN_SIM=65536    #32768 49152
NUM_EVAL_SIM=8192      #4096 or 8192

# Cao's Approximate Heston Parameters
INIT_VOL=0.3 # changed from 0.0
KAPPA=1.0
THETA=0.09
VOLVOL=0.3
RHO=-0.7


# FOLDER=''
# FOLDER='portfolios/Portfolio_Heston_1'
# ##################################################################################################################
# args=(
#     # Utility Function Parameters
#     -init_ttm=30 -r=0.0 -spread=0.01
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
#     -stochastic_process='Heston' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

#     # RL Environment Parametersj
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=1024 -vega_obs=True -critic='qr-huber'

#     # -logger_prefix='(EXP 2) Heston Experiments/spread_05/Heston_Mean_Std_05'
#     -obj_func='cvar' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=654
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# curl -s \
#   --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
#   --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
#   --form-string "message=Your script has started successfully!" \
#   https://api.pushover.net/1/messages.json

# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_1' -train_seed=1 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Mean_Std_Batch/Heston_Mean_Std_1_2' -train_seed=2 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Mean_Std_Batch/Heston_Mean_Std_1_3' -train_seed=3 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Mean_Std_Batch/Heston_Mean_Std_1_4' -train_seed=4 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Mean_Std_Batch/Heston_Mean_Std_1_5' -train_seed=5 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Mean_Std_Batch/Heston_Mean_Std_1_6' -train_seed=6 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Mean_Std_Batch/Heston_Mean_Std_1_7' -train_seed=7 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Mean_Std_Batch/Heston_Mean_Std_1_8' -train_seed=8 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Mean_Std_Batch/Heston_Mean_Std_1_9' -train_seed=9 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Mean_Std_Batch/Heston_Mean_Std_1_10' -train_seed=10 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Mean_Std_Batch/Heston_Mean_Std_1_11' -train_seed=11

# curl -s \
#   --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
#   --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
#   --form-string "message=11/21 Completed" \
#   https://api.pushover.net/1/messages.json

# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_CVaR_Batch/Heston_CVaR_1_12' -train_seed=12 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_CVaR_Batch/Heston_CVaR_1_13' -train_seed=13 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_CVaR_Batch/Heston_CVaR_1_14' -train_seed=14 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_CVaR_Batch/Heston_CVaR_1_15' -train_seed=15 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_CVaR_Batch/Heston_CVaR_1_16' -train_seed=16 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_CVaR_Batch/Heston_CVaR_1_17' -train_seed=17 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_CVaR_Batch/Heston_CVaR_1_18' -train_seed=18 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_CVaR_Batch/Heston_CVaR_1_19' -train_seed=19 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_CVaR_Batch/Heston_CVaR_1_20' -train_seed=20 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_CVaR_Batch/Heston_CVaR_1_21' -train_seed=21

# curl -s \
#   --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
#   --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
#   --form-string "message=21/21 Completed" \
#   https://api.pushover.net/1/messages.json

# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_22' -train_seed=22 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_23' -train_seed=23 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_24' -train_seed=24 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_25' -train_seed=25 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_26' -train_seed=26 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_27' -train_seed=27 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_28' -train_seed=28 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_29' -train_seed=29 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_30' -train_seed=30 &
# python run.py "${args[@]}" -logger_prefix='(EXP 4) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_31' -train_seed=31

# curl -s \
#   --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
#   --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
#   --form-string "message=31/31 VaR Completed" \
#   https://api.pushover.net/1/messages.json

######################################################################################################################################################





# Greek Run ###########################################################################################################################################
FOLDER='portfolios/Portfolio_Heston_1'
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
    -numerical_accuracy='low' -n_jobs=-1 -eval_seed=1234

    # RL Agent Parameters
    -vega_obs=True #-critic='qr-huber'

    # -logger_prefix='Cao_GBM_Delta_05pct'

    # Load Portfolio
    -portfolio_folder=$FOLDER
)
# #####################################################################################################################################################
# Delta Run #########################################################################################################################################
python greek_run.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Delta_1' -spread=0.01 &

#####################################################################################################################################################
# Delta-Gamma Run ###################################################################################################################################
python greek_run.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Gamma_1' -spread=0.01 &

#####################################################################################################################################################
# Delta-Vega Run ###################################################################################################################################
python greek_run.py "${args_Greeks[@]}" -strategy='vega' -logger_prefix='(EXP 4) Heston Experiments/spread_1/Heston_Vega_1' -spread=0.01