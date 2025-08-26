#!/bin/bash
#  nohup "./(EXP 2.1) Cao Heston Experiments.sh" > .terminal_output2.txt 2>&1 &
#  tail -f .terminal_output2.txt

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
FOLDER='portfolios/.Portfolio_Heston_2_Am'
### Mean-Std Experiment (Spread: 0.005) ###############################################################################################################
args_Mean_Std=(
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
    -stochastic_process='Heston' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

    # RL Environment Parameters
    -action_space="0,1"

    # RL Agent Parameters
    -n_step=5 -batch_size=1024 -vega_obs=True -critic='qr-huber'

    # -logger_prefix='(EXP 2) Heston Experiments/spread_05/Heston_Mean_Std_05'
    -obj_func='meanstd' -std_coef=1.645
    -eval_sim=$NUM_EVAL_SIM -eval_seed=654
    -eval_only=False

    # Load Portfolio
    -portfolio_folder=$FOLDER
)
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_1' -train_seed=1 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_2' -train_seed=2 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_3' -train_seed=3 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_4' -train_seed=4 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_5' -train_seed=5 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_6' -train_seed=6 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_7' -train_seed=7 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_8' -train_seed=8 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_9' -train_seed=9 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_10' -train_seed=10

# curl -s \
#   --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
#   --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
#   --form-string "message=Mean Std 2pct 10/30 Finished (VM_60C_1)" \
#   https://api.pushover.net/1/messages.json

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_11' -train_seed=11 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_12' -train_seed=12 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_13' -train_seed=13 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_14' -train_seed=14 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_15' -train_seed=15 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_16' -train_seed=16 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_17' -train_seed=17 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_18' -train_seed=18 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_19' -train_seed=19 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_20' -train_seed=20 

curl -s \
  --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
  --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
  --form-string "message=Mean Std 2pct 20/30 Finished (VM_60C_1)" \
  https://api.pushover.net/1/messages.json

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_21' -train_seed=21 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_22' -train_seed=22 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_23' -train_seed=23 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_24' -train_seed=24 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_25' -train_seed=25 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_26' -train_seed=26 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_27' -train_seed=27 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_28' -train_seed=28 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_29' -train_seed=29 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Mean_Std_Batch/Heston_Mean_Std_2_30' -train_seed=30 

# Send notification that the script finished
curl -s \
  --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
  --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
  --form-string "message=Mean Std 2pct 30/30 finished (VM_60C_1)" \
  https://api.pushover.net/1/messages.json

#Shutdown VM
# sudo poweroff


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
    -stochastic_process='Heston' -time_to_simulate=30 -eval_sim=$NUM_EVAL_SIM -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=-1 -eval_seed=3456

    # RL Agent Parameters
    -vega_obs=True #-critic='qr-huber'

    # -logger_prefix='Cao_Heston_Delta_05pct'

    # Load Portfolio
    -portfolio_folder=$FOLDER
)
###################################################################################################################################################
# Delta Run #########################################################################################################################################
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='(EXP 2.1) Heston Experiments/spread_05/Heston_Delta_05' -spread=0.005 &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='(EXP 2.1) Heston Experiments/spread_1/Heston_Delta_1' -spread=0.01 &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Delta_2' -spread=0.02 &

# ###################################################################################################################################################
# # Delta-Gamma Run ###################################################################################################################################
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='(EXP 2.1) Heston Experiments/spread_05/Heston_Gamma_05' -spread=0.005 &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='(EXP 2.1) Heston Experiments/spread_1/Heston_Gamma_1' -spread=0.01 &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Gamma_2' -spread=0.02 &

# ###################################################################################################################################################
# # Delta-Gamma Run ###################################################################################################################################
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='vega' -logger_prefix='(EXP 2.1) Heston Experiments/spread_05/Heston_Vega_05' -spread=0.005 &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='vega' -logger_prefix='(EXP 2.1) Heston Experiments/spread_1/Heston_Vega_1' -spread=0.01 &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='vega' -logger_prefix='(EXP 2.1) Heston Experiments/spread_2/Heston_Vega_2' -spread=0.02 


# curl -s \
#   --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
#   --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
#   --form-string "message=Greeks 2 finished" \
#   https://api.pushover.net/1/messages.json