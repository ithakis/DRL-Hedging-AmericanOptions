#!/bin/bash

# Mean-Std Experiment ##################################################################################################################################

args=(
    # Utility Function Parameters
    -init_ttm=60 -r=0.0 -spread=0.005
    -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

    # Contract Parameters
    -num_conts_to_add=-1 -contract_size=100

    # Hedging Portfolio Parameters
    -hed_ttm=30 -hed_type='European'

    # init_vol is for both GBM and Heston
    -init_vol=0.0228

    # Heston, Model Parameters
    -kappa=0.0807 -theta=0.363 -volvol=0.1760 -rho=-0.3021

    # Simulation Parameters
    -stochastic_process='GBM' -time_to_simulate=60 -train_sim=1024 -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

    # RL Environment Parameters
    -action_space="0,1"


    # RL Agent Parameters
    -n_step=5 -batch_size=256 -vega_obs=False -critic='qr-huber'

    -logger_prefix='results_meanstd_poisson'
    -obj_func='meanstd' -std_coef=1.645
    -eval_sim=1024
    -eval_only=False
)

python runAlexander.py "${args[@]}"

## VaR_95 Experiment ##################################################################################################################################
args=(
    # Utility Function Parameters
    -init_ttm=60 -r=0.0 -spread=0.005
    -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

    # Contract Parameters
    -num_conts_to_add=-1 -contract_size=100

    # Hedging Portfolio Parameters
    -hed_ttm=30 -hed_type='European'

    # init_vol is for both GBM and Heston
    -init_vol=0.0228

    # Heston, Model Parameters
    -kappa=0.0807 -theta=0.363 -volvol=0.1760 -rho=-0.3021

    # Simulation Parameters
    -stochastic_process='GBM' -time_to_simulate=60 -train_sim=1024 -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=-2 -train_seed=1234

    # RL Environment Parameters
    -action_space="0,1"


    # RL Agent Parameters
    -n_step=5 -batch_size=256 -vega_obs=False -critic='qr-huber'

    -logger_prefix='results_var_poisson'
    -obj_func='var' -threshold=0.95
    -eval_sim=1024
    -eval_only=False
)

python runAlexander.py "${args[@]}"

# Mean-Std Experiment ##################################################################################################################################

args=(
    # Utility Function Parameters
    -init_ttm=60 -r=0.0 -spread=0.005
    -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

    # Contract Parameters
    -num_conts_to_add=-1 -contract_size=100

    # Hedging Portfolio Parameters
    -hed_ttm=30 -hed_type='European'

    # init_vol is for both GBM and Heston
    -init_vol=0.0228

    # Heston, Model Parameters
    -kappa=0.0807 -theta=0.363 -volvol=0.1760 -rho=-0.3021

    # Simulation Parameters
    -stochastic_process='GBM' -time_to_simulate=30 -train_sim=1024 -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=-1 -train_seed=1234

    # RL Environment Parameters
    -action_space="0,1"


    # RL Agent Parameters
    -n_step=5 -batch_size=256 -vega_obs=False -critic='qr-huber'

    -logger_prefix='results_meanstd_poisson30'
    -obj_func='meanstd' -std_coef=1.645
    -eval_sim=1024
    -eval_only=False
)

python runAlexander.py "${args[@]}"

## VaR_95 Experiment ##################################################################################################################################
args=(
    # Utility Function Parameters
    -init_ttm=60 -r=0.0 -spread=0.005
    -liab_ttms=60 -poisson_rate=1.0 -moneyness_mean=1.0 -moneyness_std=0.0

    # Contract Parameters
    -num_conts_to_add=-1 -contract_size=100

    # Hedging Portfolio Parameters
    -hed_ttm=30 -hed_type='European'

    # init_vol is for both GBM and Heston
    -init_vol=0.0228

    # Heston, Model Parameters
    -kappa=0.0807 -theta=0.363 -volvol=0.1760 -rho=-0.3021

    # Simulation Parameters
    -stochastic_process='GBM' -time_to_simulate=30 -train_sim=1024 -frq=1 -TradingDaysPerYear=252
    -numerical_accuracy='low' -n_jobs=-2 -train_seed=1234

    # RL Environment Parameters
    -action_space="0,1"


    # RL Agent Parameters
    -n_step=5 -batch_size=256 -vega_obs=False -critic='qr-huber'

    -logger_prefix='results_var_poisson30'
    -obj_func='var' -threshold=0.95
    -eval_sim=1024
    -eval_only=False
)

python runAlexander.py "${args[@]}"