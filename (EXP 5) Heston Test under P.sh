#!/bin/bash
#  nohup "./(EXP 5) Heston Test under P.sh" > .terminal_output5.txt 2>&1 &
#  tail -f .terminal_output5.txt
# 
# EXP 5: Heston Test under P-Measure with r=2%, mu=3% (5% total equity drift)
# This experiment tests Heston model hedging with physical measure equity dynamics

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
# FOLDER='portfolios/Portfolio_Heston_1_P'
# ##################################################################################################################
# args=(
#     # Utility Function Parameters
#     -init_ttm=30 -r=0.02 -mu=0.03 -spread=0.02
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

#     -obj_func='var' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=654
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# curl -s \
#   --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
#   --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
#   --form-string "message=EXP 5: Heston P-Measure (r=2%, mu=3%) started successfully!" \
#   https://api.pushover.net/1/messages.json

# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_1' -train_seed=1


# ## Send notification after the first run
# curl -s \
#   --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
#   --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
#   --form-string "message=EXP 5: Heston P-Measure (r=2%, mu=3%): First VaR Run Completed" \
#   https://api.pushover.net/1/messages.json

# After runing the first run, get the last created folder inside portfolios. -> Rename it to "Portfolio_Heston_05_P" -> then set the FOLDER variable above to this path.
# Get the last created folder in portfolios directory and rename it
# LAST_FOLDER=$(ls -t portfolios/ | head -1)
# mv "portfolios/$LAST_FOLDER" "portfolios/Portfolio_Heston_2_P"

# Set the FOLDER variable to the new portfolio path
# FOLDER='portfolios/Portfolio_Heston_2_P'

# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_2' -portfolio_folder=$FOLDER -train_seed=2 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_3' -portfolio_folder=$FOLDER -train_seed=3 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_4' -portfolio_folder=$FOLDER -train_seed=4 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_5' -portfolio_folder=$FOLDER -train_seed=5 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_6' -portfolio_folder=$FOLDER -train_seed=6 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_7' -portfolio_folder=$FOLDER -train_seed=7 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_8' -portfolio_folder=$FOLDER -train_seed=8 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_9' -portfolio_folder=$FOLDER -train_seed=9 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_10' -portfolio_folder=$FOLDER -train_seed=10 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_11' -portfolio_folder=$FOLDER -train_seed=11

# curl -s \
#   --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
#   --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
#   --form-string "message=11/21 Completed" \
#   https://api.pushover.net/1/messages.json

# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_12' -portfolio_folder=$FOLDER -train_seed=12 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_13' -portfolio_folder=$FOLDER -train_seed=13 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_14' -portfolio_folder=$FOLDER -train_seed=14 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_15' -portfolio_folder=$FOLDER -train_seed=15 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_16' -portfolio_folder=$FOLDER -train_seed=16 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_17' -portfolio_folder=$FOLDER -train_seed=17 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_18' -portfolio_folder=$FOLDER -train_seed=18 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_19' -portfolio_folder=$FOLDER -train_seed=19 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_20' -portfolio_folder=$FOLDER -train_seed=20 &
# python run.py "${args[@]}" -logger_prefix='(EXP 5) Heston P-Measure/spread_2/Heston_VaR_P_2_21' -portfolio_folder=$FOLDER -train_seed=21

# curl -s \
#   --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
#   --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
#   --form-string "message=EXP 5: P-Measure CVaR 21/21 Completed (r=2%, mu=3%)" \
#   https://api.pushover.net/1/messages.json



# Greek Run ###########################################################################################################################################
# Use the same portfolio folder that was created and renamed in the main experiment section
FOLDER='portfolios/Portfolio_Heston_1_P'
args_Greeks=(
    # Utility Function Parameters
    -init_ttm=30 -r=0.02 -mu=0.03 #-spread=0.005
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

    # Load Portfolio - Use the same folder variable that was set in the main section
    -portfolio_folder=$FOLDER
)
# #####################################################################################################################################################
# Delta Run #########################################################################################################################################
python greek_run.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='(EXP 5) Heston P-Measure/spread_1/Heston_Delta_P' -spread=0.01 &

#####################################################################################################################################################
# Delta-Gamma Run ###################################################################################################################################
python greek_run.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='(EXP 5) Heston P-Measure/spread_1/Heston_Gamma_P' -spread=0.01 &

#####################################################################################################################################################
# Delta-Vega Run ###################################################################################################################################
python greek_run.py "${args_Greeks[@]}" -strategy='vega' -logger_prefix='(EXP 5) Heston P-Measure/spread_1/Heston_Vega_P' -spread=0.01


curl -s \
  --form-string "token=akuyyk1ewemroxn61w5tmkp13hvoge" \
  --form-string "user=uiz86t468s8d936bu652atbawzfm6a" \
  --form-string "message=EXP 5: P-Measure Greeks Completed (r=2%, mu=3%)" \
  https://api.pushover.net/1/messages.json