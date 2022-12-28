

import argparse

from recbole.quick_start.quick_start import run_recbole,objective_function


import os
import time

import logging
from logging import getLogger

from recbole.config.configurator import Config
from recbole.utils.utils import   init_seed
from recbole.utils.logger import init_logger
from recbole.data.utils import create_dataset
def writer_csv(result_dict,dataset,model,task,robust,gamma_v=0,gamma_t=0):
    curPath = os.path.abspath(os.path.dirname(__file__))
    #rootPath = curPath[:curPath.find("recbole") + len("recbole")]
    rootPath=curPath
    rootPath = os.path.join(rootPath, 'excel')
    if not os.path.exists(rootPath):
        os.makedirs(rootPath)
    rootPath = os.path.join(rootPath, 'robust')
    if not os.path.exists(rootPath):
        os.makedirs(rootPath)
    filename=model+"_"+dataset+"_rd_"+task+"_3"
    # if robust:
    #     filename=filename+"_rd_dr"
    # else:
    #     filename=filename+"_dr"
    rootPath = os.path.join(rootPath, filename+'.csv')
    FF = True
    while FF:
        try:
            writer = open(rootPath, 'a+')
        except IOError:
            print('Waiting for writing the file')
            time.sleep(1)
        else:
            writer.write(model+','+str(gamma_v)+','+str(gamma_t))

            for method in ['mse','rmse','mae']:
                #metric = '{}@{}'.format(method, 5)
                writer.write(',')
                writer.write(str(result_dict[method]))
            writer.write('\n')
            writer.close()
            FF = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='recbole', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='patio', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    config_dict={
            # "train_batch_size": [1024,512,256,128,64,32],
            "gamma_v": [1,2,3,4,5,6,7,8,9,10],
            "gamma_t": [1,2,3,4,5,6,7,8,9,10],
            # "embedding_size": [32, 64, 128],
            
    }
    if args.config_files.find('ips')!=-1:
        task='ips'
    else:
        task='dr'
    # configurations initialization
    config = Config(model=args.model, dataset=args.dataset,config_file_list=config_file_list, config_dict={})
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    # config['eval_batch_size'] = 128
    # config['learning_rate'] = 0.01
    logger.info(config)
    # # dataset filtering
    dataset = create_dataset(config)
    #
    logger.info(dataset)
    result_dict = None
    best_config={}
    config={}
    best_rmse = 0.0
    best_item_v = None
    best_item_t = None
    for item_v in config_dict['gamma_v']:

        config['gamma_v']=item_v
        for item_t in config_dict['gamma_t']:
            config['gamma_t']=item_t
            result = objective_function(config_dict=config, config_file_list=config_file_list,saved=True)
            result=result["test_result"]
            writer_csv(result, args.dataset, args.model,task,True,item_v,item_t)
            rmse = result['rmse']
            if rmse < best_rmse:
                best_item_v = item_v
                best_item_t = item_t
            if result_dict == None or rmse < result_dict['rmse']:
                result_dict = result
        # config[key] = best_item
    best_config['gamma_v'] = best_item_v
    best_config['gamma_t'] = best_item_t
    writer_csv(result_dict, args.dataset, args.model,'ips',True)
    print('The best parameters of '+args.model+' are: ',best_config)
    print('The best result of '+args.model+' are: ',result_dict)