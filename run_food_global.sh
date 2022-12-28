
# propensity score
python run_recbole.py --config_files=debias_config_global_food/get_ps.yaml  --model=PMF --dataset=food_global #get psv
python run_recbole.py --config_files=debias_config_global_food/get_ps.yaml  --model=DiscretePS --dataset=food_global #get pst

python run_recbole.py --config_files=debias_config_global_food/get_dps_tmf.yaml  --model=TMF --dataset=food_global #get dancer_pst
python run_recbole.py --config_files=debias_config_global_food/get_dps_tmtf.yaml  --model=TMTF --dataset=food_global #get dancer_pst


#dips
python run_recbole.py --config_files=debias_config_global_food/dips_tmf.yaml  --model=TMF --dataset=food_global #x loss nan get dancer_ips
python run_recbole.py --config_files=debias_config_global_food/dips_tmtf.yaml  --model=TMF --dataset=food_global #get dancer_ips



#ips
python run_recbole.py --config_files=debias_config_global_food/ips.yaml  --model=TMF --dataset=food_global #get ips
python run_recbole.py --config_files=debias_config_global_food/rd_ips.yaml  --model=TMF --dataset=food_global #N lr -> Y  get rd_ips


#dr
#python run_recbole.py --config_files=debias_config_global_food/dr.yaml  --model=TMF --dataset=food_global #get dr
#python run_recbole.py --config_files=debias_config_global_food/rd_dr.yaml  --model=TMF --dataset=food_global #get rd_dr


python run_recbole.py --config_files=debias_config_global_food/dips_tmf.yaml  --model=BPTF --dataset=food_global #x loss nan get dancer_ips
python run_recbole.py --config_files=debias_config_global_food/dips_tmtf.yaml  --model=BPTF --dataset=food_global #get dancer_ips



#ips
python run_recbole.py --config_files=debias_config_global_food/ips.yaml  --model=BPTF --dataset=food_global #get ips
python run_recbole.py --config_files=debias_config_global_food/rd_ips.yaml  --model=BPTF --dataset=food_global #N lr -> Y get rd_ips


#dr
#python run_recbole.py --config_files=debias_config_global_food/dr.yaml  --model=BPTF --dataset=food_global #get dr
#python run_recbole.py --config_files=debias_config_global_food/rd_dr.yaml  --model=BPTF --dataset=food_global #get rd_dr

#base
python run_recbole.py --config_files=debias_config_global_food/base.yaml  --model=TMF --dataset=food_global #get base
python run_recbole.py --config_files=debias_config_global_food/base.yaml  --model=BPTF --dataset=food_global #get base