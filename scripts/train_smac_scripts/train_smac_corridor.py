import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument("--env_name", type=str, default="StarCraft2", help="环境名称")
    parser.add_argument("--map_name", type=str, default="corridor", help="地图名称")
    parser.add_argument("--algorithm_name", type=str, default="rmappo", help="算法名称")
    parser.add_argument("--experiment_name", type=str, default="check", help="实验名称")
    parser.add_argument("--seed_max", type=int, default=1, help="最大种子数")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA设备ID")
    
    # 训练参数
    parser.add_argument("--n_training_threads", type=int, default=1)
    parser.add_argument("--n_rollout_threads", type=int, default=8)
    parser.add_argument("--num_mini_batch", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=400)
    parser.add_argument("--num_env_steps", type=int, default=10000000)
    parser.add_argument("--ppo_epoch", type=int, default=5)
    parser.add_argument("--use_value_active_masks", action='store_true')
    parser.add_argument("--use_eval", action='store_true')
    parser.add_argument("--eval_episodes", type=int, default=32)
    
    # 超网络参数
    parser.add_argument("--use_smooth_regularizer", type=bool, default=True)
    parser.add_argument("--use_align_regularizer", type=bool, default=True)
    parser.add_argument("--smooth_weight", type=float, default=0.1)
    parser.add_argument("--align_weight", type=float, default=0.05)
    
    # RNN相关参数
    parser.add_argument("--use_recurrent_policy", action="store_true", default=True)
    parser.add_argument("--recurrent_N", type=int, default=1)
    parser.add_argument("--data_chunk_length", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--layer_N", type=int, default=1)
    parser.add_argument("--use_orthogonal", action="store_true", default=True)
    parser.add_argument("--gain", type=float, default=0.01)
    
    # 优化器参数
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--critic_lr", type=float, default=5e-4)
    parser.add_argument("--opti_eps", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    
    args = parser.parse_args()
    return args, parser

def main():
    args, parser = parse_args()
    
    # 设置 GPU 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    print(f"env is {args.env_name}, map is {args.map_name}, algo is {args.algorithm_name}, "
          f"exp is {args.experiment_name}, max seed is {args.seed_max}")

    # 导入训练模块
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train.train_smac import main as train_main

    # 对每个种子运行训练
    for seed in range(1, args.seed_max + 1):
        print(f"seed is {seed}:")
        
        # 将参数转换为列表形式
        arg_list = []
        for arg_name, arg_value in vars(args).items():
            if arg_value is True:
                arg_list.append(f"--{arg_name}")
            elif arg_value is not False and arg_value is not None:  # 跳过False和None值
                arg_list.append(f"--{arg_name}")
                arg_list.append(str(arg_value))
        
        # 更新种子
        arg_list.extend(["--seed", str(seed)])
        
        # 调用训练函数
        train_main(arg_list)

if __name__ == "__main__":
    main() 