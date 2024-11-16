#!/bin/bash
#  nohup "./(Cao) SABR Experiments.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt


args_Mean_Std=(
-spread=0.02 -obj_func=meanstd -train_sim=40000 -eval_sim=5000 -critic='qr-huber' -std_coef=1.645
-init_vol=0.3 -mu=0.0 -vov=0.3 -vega_obs=True -gbm=False -sabr=True -hed_ttm=30 
-liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -action_space=0,1 -logger_prefix='(Cao) Experiments/spread_2/Cao_Mean_Std_2'
-n_step=5
)
python run.py "${args_Mean_Std[@]}" &

args_VaR=(
-spread=0.02 -obj_func=var -train_sim=40000 -eval_sim=5000 -critic='qr-huber' -std_coef=1.645
-init_vol=0.3 -mu=0.0 -vov=0.3 -vega_obs=True -gbm=False -sabr=True -hed_ttm=30 
-liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -action_space=0,1 -logger_prefix='(Cao) Experiments/spread_2/Cao_VaR_2'
-n_step=5
)
python run.py "${args_VaR[@]}" &

args_CVaR=(
-spread=0.02 -obj_func=cvar -train_sim=40000 -eval_sim=5000 -critic='qr-huber' -std_coef=1.645
-init_vol=0.3 -mu=0.0 -vov=0.3 -vega_obs=True -gbm=False -sabr=True -hed_ttm=30 
-liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -action_space=0,1 -logger_prefix='(Cao) Experiments/spread_2/Cao_CVaR_2'
-n_step=5
)
python run.py "${args_CVaR[@]}" &

Make the greek runs
args_Greeks=(
-spread=0.02 -eval_sim=5000
-init_vol=0.3 -mu=0.0 -vov=0.3 -vega_obs=True -gbm=False -sabr=True -hed_ttm=30 
-liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 
)
python greek_run.py "${args_Greeks[@]}" -logger_prefix='(Cao) Experiments/spread_2/Cao_Delta_2' -strategy='delta' &
python greek_run.py "${args_Greeks[@]}" -logger_prefix='(Cao) Experiments/spread_2/Cao_Gamma_2' -strategy='gamma' &
python greek_run.py "${args_Greeks[@]}" -logger_prefix='(Cao) Experiments/spread_2/Cao_Vega_2' -strategy='vega' 
