import pickle
from collections import Counter
import numpy as np

def load_pickle_file(filepath):
    """加载 pickle 文件"""
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def hash_trajectory(traj):
    """
    将 trajectory 转换为一个可哈希的字符串，用于检测重复。
    这里只考虑 context_states, context_actions, context_rewards。
    """
    context_states = traj['context_states']
    context_actions = traj['context_actions']
    context_rewards = traj['context_rewards']
    # 将数组转换为字符串以便哈希
    return (
        np.array2string(context_states, separator=','),
        np.array2string(context_actions, separator=','),
        np.array2string(context_rewards, separator=',')
    )

def sample_sequences(data, num_samples=1):
    """随机抽取一些序列"""
    sampled_data = data[:num_samples] # 取前 num_samples 个序列
    return sampled_data

def main():
    # 替换为你的 pickle 文件路径
    pickle_file = './datasets/trajs_bandit_envs100000_hists1_samples1_H500_d5_var0.3_cov0.0_train.pkl'

    # 加载数据
    data = load_pickle_file(pickle_file)

    # 随机抽取一些序列
    sampled_data = sample_sequences(data)

        # 打印结果
    print(f"随机抽取了 {len(sampled_data)} 个序列：")
    for i, traj in enumerate(sampled_data):
        print(f"\n序列 {i + 1}:")
        print(f"Context States Shape: {traj['context_states'].shape}")
        print(f"Context Actions Shape: {traj['context_actions'].shape}")
        print(f"Context Rewards Shape: {traj['context_rewards'].shape}")
        # 打印查询状态
        print(f"Query States: {traj['query_state']}")
        # 打印最优动作
        print(f"Optimal Action: {traj['optimal_action']}")

if __name__ == '__main__':
    main()
