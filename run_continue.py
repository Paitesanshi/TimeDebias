# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn


import argparse

from recbole.quick_start import run_recbole, continue_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', '-f', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    # config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    continue_train(args.file_path)
