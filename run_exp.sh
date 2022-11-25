
# propensity score
python run_recbole.py --config_files=debias_config/get_ps.yaml  --model=PMF --dataset=ml_100k_week #get psv
python run_recbole.py --config_files=debias_config/get_ps.yaml  --model=DiscretePS --dataset=ml_100k_week #get pst

python run_recbole.py --config_files=debias_config/get_dps.yaml  --model=TMF --dataset=ml_100k_week #get dancer_pst
python run_recbole.py --config_files=debias_config/get_dps.yaml  --model=TMTF --dataset=ml_100k_week #get dancer_pst


#dips
python run_recbole.py --config_files=debias_config/dips.yaml  --model=TimeSVD --dataset=ml_100k_week #get dancer_ips
python run_recbole.py --config_files=debias_config/dips.yaml  --model=TimeSVD --dataset=ml_100k_week #get dancer_ips



#ips
python run_recbole.py --config_files=debias_config/ips.yaml  --model=TimeSVD --dataset=ml_100k_week #get ips
python run_recbole.py --config_files=debias_config/rd_ips.yaml  --model=TimeSVD --dataset=ml_100k_week #get rd_ips


#dr
python run_recbole.py --config_files=debias_config/dr.yaml  --model=TimeSVD --dataset=ml_100k_week #get dr
python run_recbole.py --config_files=debias_config/rd_dr.yaml  --model=TimeSVD --dataset=ml_100k_week #get rd_dr


python run_recbole.py --config_files=debias_config/dips.yaml  --model=BPTF --dataset=ml_100k_week #get dancer_ips
python run_recbole.py --config_files=debias_config/dips.yaml  --model=BPTF --dataset=ml_100k_week #get dancer_ips



#ips
python run_recbole.py --config_files=debias_config/ips.yaml  --model=BPTF --dataset=ml_100k_week #get ips
python run_recbole.py --config_files=debias_config/rd_ips.yaml  --model=BPTF --dataset=ml_100k_week #get rd_ips


#dr
python run_recbole.py --config_files=debias_config/dr.yaml  --model=BPTF --dataset=ml_100k_week #get dr
python run_recbole.py --config_files=debias_config/rd_dr.yaml  --model=BPTF --dataset=ml_100k_week #get rd_dr

