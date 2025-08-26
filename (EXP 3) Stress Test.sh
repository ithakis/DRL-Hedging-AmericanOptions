#!/bin/bash
#  nohup "./(EXP 3) Stress Test.sh" > .terminal_output3.txt 2>&1 &
#  tail -f .terminal_output3.txt

# Set common parameters
NUM_TRAIN_SIM=65536    #32768 49152
NUM_EVAL_SIM=8192      #4096 or 8192

# Cao's Approximate Heston Parameters
INIT_VOL=0.3 # changed from 0.0
KAPPA=1.0
THETA=0.09
VOLVOL=0.3
RHO=-0.7


FOLDER=''
# FOLDER='portfolios/.Portfolio_Heston_1_Am_-0.2rho'
### Mean-Std Experiment (Spread: 0.005) ###############################################################################################################
args_Mean_Std=(
    # Utility Function Parameters
    -init_ttm=30 -r=0.0 -spread=0.01
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

    # RL Environment Parametersj
    -action_space="0,1"

    # RL Agent Parameters
    -n_step=5 -batch_size=1024 -vega_obs=True -critic='qr-huber'

    # -logger_prefix='(EXP 2) Heston Experiments/spread_05/Heston_Mean_Std_05'
    -obj_func='var' -std_coef=1.645
    -eval_sim=$NUM_EVAL_SIM -eval_seed=654
    -eval_only=False

    # Load Portfolio
    -portfolio_folder=$FOLDER
)
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 4) ST Heston Experiments/spread_05/Heston_VaR_Batch/Heston_VaR_1_1' -train_seed=1 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_2_2' -train_seed=2 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_2_3' -train_seed=3 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_2_4' -train_seed=4 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_2_5' -train_seed=5 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_2_6' -train_seed=6 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_2_7' -train_seed=7 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_2_8' -train_seed=8 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_2_9' -train_seed=9 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_10' -train_seed=10 
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_11' -train_seed=11 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_12' -train_seed=12 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_13' -train_seed=13 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_14' -train_seed=14 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_15' -train_seed=15 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_16' -train_seed=16 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_17' -train_seed=17 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_18' -train_seed=18 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_19' -train_seed=19 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_20' -train_seed=20 
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_21' -train_seed=21 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_22' -train_seed=22 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_23' -train_seed=23 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_24' -train_seed=24 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_25' -train_seed=25 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_26' -train_seed=26 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_27' -train_seed=27 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_28' -train_seed=28 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_29' -train_seed=29 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_30' -train_seed=30 

# Send notification that the script finished
curl -s \
  --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
  --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
  --form-string "message=Your script has completed successfully!" \
  https://api.pushover.net/1/messages.json

#Shutdown VM
API_TOKEN="f541174f-9e4c-48c2-bf57-813f23075dfe"
WORKSPACE_UUID="f541174f-9e4c-48c2-bf57-813f23075dfe"
API_URL="https://workspace.server.api.endpoint/workspaces/$WORKSPACE_UUID/pause"

# Send the API request to pause (shutdown) the workspace
curl -X POST "$API_URL" \
     -H "Authorization: Bearer $API_TOKEN" \
     -H "Content-Type: application/json"





# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_31' -train_seed=31 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_32' -train_seed=32 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_33' -train_seed=33 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_34' -train_seed=34 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_35' -train_seed=35 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_36' -train_seed=36 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_37' -train_seed=37 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_38' -train_seed=38 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_39' -train_seed=39 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_40' -train_seed=40 
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_41' -train_seed=41 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_42' -train_seed=42 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_43' -train_seed=43 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_44' -train_seed=44 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_45' -train_seed=45 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_46' -train_seed=46 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_47' -train_seed=47 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_48' -train_seed=48 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_49' -train_seed=49 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_50' -train_seed=50 
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_51' -train_seed=51 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_52' -train_seed=52 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_53' -train_seed=53 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_54' -train_seed=54 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_55' -train_seed=55 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_56' -train_seed=56 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_57' -train_seed=57 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_58' -train_seed=58 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_59' -train_seed=59 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 3.1) ST Heston Experiments (rho-0.2)/spread_1/Heston_VaR_Batch/Heston_VaR_1_60' -train_seed=60 &
