#!/bin/bash
#  nohup "./(EXP 2.2) Heston Experiments Single Option - European Hedging.sh" > .terminal_output2_2.txt 2>&1 & 
#  tail -f .terminal_output2_2.txt


# Set common parameters
NUM_TRAIN_SIM=65536    #32768 49152
NUM_EVAL_SIM=8192      #4096 or 8192

# My Heston Parameters
# INIT_VOL=0.0228
# KAPPA=0.0807
# THETA=0.363
# VOLVOL=0.1760
# RHO=-0.3021

# Cao's Approximate Heston Parameters
INIT_VOL=0.3 
KAPPA=1.0
THETA=0.09
VOLVOL=0.3
RHO=-0.7


FOLDER='portfolios/.Portfolio_Heston_1_Eu_Single'
# Mean-Std Experiment (Spread: 0.005) ###############################################################################################################
args_Mean_Std=(
    # Utility Function Parameters
    -init_ttm=30 -r=0.0 -spread=0.005
    -liab_ttms=60 -poisson_rate=0.0 -moneyness_mean=1.0 -moneyness_std=0.0

    # Contract Parameters
    -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

    # Hedging Portfolio Parameters
    -hed_ttm=30 -hed_type='European'

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
    -n_step=5 -batch_size=512 -vega_obs=True -critic='qr-huber'

    # -logger_prefix='(EXP 2) Heston Experiments/spread_05/Heston_Mean_Std_05'
    -obj_func='meanstd' -std_coef=1.645
    -eval_sim=$NUM_EVAL_SIM -eval_seed=654
    -eval_only=False

    # Load Portfolio
    -portfolio_folder=$FOLDER
)

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_1' -train_seed=1 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_2' -train_seed=2 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_3' -train_seed=3 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_4' -train_seed=4

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_5' -train_seed=5 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_6' -train_seed=6 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_7' -train_seed=7 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_8' -train_seed=8

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_9' -train_seed=9 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_10' -train_seed=10 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_11' -train_seed=11 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_12' -train_seed=12

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_13' -train_seed=13 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_14' -train_seed=14 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_15' -train_seed=15 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_16' -train_seed=16

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_17' -train_seed=17 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_18' -train_seed=18 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_19' -train_seed=19 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_20' -train_seed=20

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_21' -train_seed=21 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_22' -train_seed=22 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_23' -train_seed=23 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_24' -train_seed=24

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_25' -train_seed=25 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_26' -train_seed=26 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_27' -train_seed=27 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_28' -train_seed=28

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_29' -train_seed=29 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_30' -train_seed=30 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_31' -train_seed=31 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_32' -train_seed=32

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_33' -train_seed=33 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_34' -train_seed=34 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_35' -train_seed=35 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_36' -train_seed=36

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_37' -train_seed=37 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_38' -train_seed=38 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_39' -train_seed=39 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_40' -train_seed=40

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_41' -train_seed=41 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_42' -train_seed=42 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_43' -train_seed=43 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_44' -train_seed=44

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_45' -train_seed=45 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_46' -train_seed=46 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_47' -train_seed=47 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_48' -train_seed=48

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_49' -train_seed=49 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_50' -train_seed=50 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_51' -train_seed=51 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_52' -train_seed=52

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_53' -train_seed=53 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_54' -train_seed=54 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_55' -train_seed=55 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_56' -train_seed=56

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_57' -train_seed=57 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_58' -train_seed=58 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_59' -train_seed=59 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_60' -train_seed=60

python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_61' -train_seed=61 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_62' -train_seed=62 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_63' -train_seed=63 &
python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Mean_Std_Batch/Heston_Mean_Std_05_64' -train_seed=64




# # Mean-Std Experiment (Spread: 0.005) ###############################################################################################################
# args_Mean_Std=(
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
#     -stochastic_process='Heston' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=True -critic='qr-huber'

#     # -logger_prefix='(EXP 2) Heston Experiments/spread_05/Heston_Mean_Std_05'
#     -obj_func='var' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=654
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_1' -train_seed=1234 &
# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_2' -train_seed=2234 &
# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_3' -train_seed=3234 &
# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_4' -train_seed=4234 

# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_5' -train_seed=5234 &
# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_6' -train_seed=6234 &
# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_7' -train_seed=7234 &
# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_8' -train_seed=8234 

# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_9' -train_seed=9234 &
# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_10' -train_seed=1334 &
# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_11' -train_seed=1434 &
# # python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_12' -train_seed=1534

# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_13' -train_seed=1634 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_14' -train_seed=1734 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_15' -train_seed=1834 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_VaR_Batch/Heston_VaR_2_16' -train_seed=1934 


# # Mean-Std Experiment (Spread: 0.005) ###############################################################################################################
# args_Mean_Std=(
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
#     -stochastic_process='Heston' -time_to_simulate=30 -train_sim=$NUM_TRAIN_SIM -frq=1 -TradingDaysPerYear=252
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=True -critic='qr-huber'

#     # -logger_prefix='(EXP 2) Heston Experiments/spread_05/Heston_Mean_Std_05'
#     -obj_func='cvar' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=654
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )

# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_1' -train_seed=1234 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_2' -train_seed=2234 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_3' -train_seed=3234 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_4' -train_seed=4234 

# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_5' -train_seed=5234 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_6' -train_seed=6234 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_7' -train_seed=7234 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_8' -train_seed=8234 

# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_9' -train_seed=9234 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_10' -train_seed=1334 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_11' -train_seed=1434 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_12' -train_seed=1534 

# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_13' -train_seed=1634 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_14' -train_seed=1734 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_15' -train_seed=1834 &
# python runAlexander.py "${args_Mean_Std[@]}" -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_CVaR_Batch/Heston_CVaR_2_16' -train_seed=1934 


#####################################################################################################################################################
# Get the most recently modified hidden folder inside ./portfolios ###################################################################################
# [ -d "./portfolios" ] || { echo "Error: Directory ./portfolios does not exist."; exit 1; }
# echo -e "\n>>Retrieving the most recently modified hidden folder inside ./portfolios..."
# last_created_folder=$(ls -td ./portfolios/.[!.]*/ 2>/dev/null | head -1)
# [ -n "$last_created_folder" ] && echo "Last modified hidden folder: $(basename "$last_created_folder")" || echo "No hidden subdirectories found in ./portfolios."
# echo
# #####################################################################################################################################################

# VaR Experiment (Spread: 0.005) ###############################################################################################################
# args_VaR=(
#     # Utility Function Parameters
#     -init_ttm=30 -r=0.0 -spread=0.005
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

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=True -critic='qr-huber'

#     -logger_prefix='(EXP 2) Heston Experiments/spread_05/Heston_VaR_05'
#     -obj_func='var' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=4
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )
# python runAlexander.py "${args_VaR[@]}"

# CVaR Experiment (Spread: 0.005) ###############################################################################################################
# args_CVaR=(
#     # Utility Function Parameters
#     -init_ttm=30 -r=0.0 -spread=0.005
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
#     -numerical_accuracy='low' -n_jobs=-1 -train_seed=8

#     # RL Environment Parameters
#     -action_space="0,1"

#     # RL Agent Parameters
#     -n_step=5 -batch_size=512 -vega_obs=True -critic='qr-huber'

#     -logger_prefix='(EXP 2) Heston Experiments/spread_05/Heston_CVaR_05'
#     -obj_func='cvar' -std_coef=1.645
#     -eval_sim=$NUM_EVAL_SIM -eval_seed=5678
#     -eval_only=False

#     # Load Portfolio
#     -portfolio_folder=$FOLDER
# )
# # python runAlexander.py "${args_CVaR[@]}"

# # #####################################################################################################################################################
# # # Run Both the last 2 at the same time as the data is generated from the first
# python runAlexander.py "${args_VaR[@]}" & python runAlexander.py "${args_CVaR[@]}" 05

#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
# Greek Run ###########################################################################################################################################
args_Greeks=(
    # Utility Function Parameters
    -init_ttm=30 -r=0.0 #-spread=0.005
    -liab_ttms=60 -poisson_rate=0.0 -moneyness_mean=1.0 -moneyness_std=0.0

    # Contract Parameters
    -S0=10 -K=10 -num_conts_to_add=-1 -contract_size=100

    # Hedging Portfolio Parameters
    -hed_ttm=30 -hed_type='European'

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
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Delta_05' -spread=0.005 &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='Cao_Heston_Delta_1pct' -portfolio_folder='portfolios/.Portfolio_Heston_1'   -spread=0.01  &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='delta' -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_Delta_2' -portfolio_folder='portfolios/.Portfolio_Heston_2_Am'   -spread=0.02  &

###################################################################################################################################################
# Delta-Gamma Run ###################################################################################################################################
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Gamma_05' -spread=0.005 &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='Cao_Heston_Gamma_1pct' -portfolio_folder='portfolios/.Portfolio_Heston_1'   -spread=0.01  &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='gamma' -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_Gamma_2' -portfolio_folder='portfolios/.Portfolio_Heston_2_Am'   -spread=0.02 &

###################################################################################################################################################
# Delta-Gamma Run ###################################################################################################################################
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='vega' -logger_prefix='(EXP 2.2) Heston Experiments/spread_05/Heston_Vega_05' -spread=0.005
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='vega' -logger_prefix='Cao_Heston_Gamma_1pct' -portfolio_folder='portfolios/.Portfolio_Heston_1'   -spread=0.01  &
# python greek_runAlexander.py "${args_Greeks[@]}" -strategy='vega' -logger_prefix='(EXP 2) Heston Experiments/spread_2/Heston_Vega_2' -portfolio_folder='portfolios/.Portfolio_Heston_2_Am'   -spread=0.02  