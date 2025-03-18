import torch
import torch.nn as nn
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.utils.util import get_shape_from_obs_space
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np

class HyperNetwork(nn.Module):
    def __init__(self, args, obs_space, action_space, device):
        super(HyperNetwork, self).__init__()
        self.args = args
        self.device = device
        self.obs_space = obs_space
        self.action_space = action_space

        # 输入编码器：对每个智能体的 obs 进行编码
        self.obs_encoder = MLPBase(args, obs_space).to(device)

        # 注意力机制：聚合全局信息
        self.attention = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=4).to(device)

        # 历史信息处理：LSTM
        self.history_lstm = RNNLayer(
            inputs_dim=args.hidden_size,
            outputs_dim=args.hidden_size,
            recurrent_N=args.recurrent_N,
            use_orthogonal=args.use_orthogonal
        ).to(device)

        # Actor 网络参数维度
        self.actor_input_size = get_shape_from_obs_space(obs_space)[0]
        self.actor_hidden_size = 64  # Actor 隐藏层大小
        self.actor_output_size = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]

        # 输出层：生成 Actor 权重的增量
        self.delta_w1 = nn.Linear(args.hidden_size, self.actor_input_size * self.actor_hidden_size).to(device)
        self.delta_b1 = nn.Linear(args.hidden_size, self.actor_hidden_size).to(device)
        self.delta_w2 = nn.Linear(args.hidden_size, self.actor_hidden_size * self.actor_output_size).to(device)
        self.delta_b2 = nn.Linear(args.hidden_size, self.actor_output_size).to(device)

    def forward(self, obs, rnn_states, history_obs=None, history_actions=None, history_rewards=None):
        """
        输入：
            obs: (batch_size, obs_dim) 或 (batch_size, num_agents, obs_dim)，可以是numpy数组或torch张量
            rnn_states: (batch_size, hidden_size) 或 (batch_size, num_agents, hidden_size)
            history_obs: (batch_size, seq_len, num_agents, obs_dim) [可选]
        输出：
            delta_w1, delta_b1, delta_w2, delta_b2: Actor 权重增量
            rnn_states: 更新后的 LSTM 状态
        """
        print("HyperNetwork.forward called")

        # 将输入转换为torch张量
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        if isinstance(rnn_states, np.ndarray):
            rnn_states = torch.from_numpy(rnn_states).to(self.device)

        # 处理输入维度
        if len(obs.shape) == 2:
            batch_size, obs_dim = obs.shape
            num_agents = 1
            obs = obs.unsqueeze(1)  # 添加 num_agents 维度
        else:
            batch_size, num_agents, obs_dim = obs.shape

        # 1. 编码当前 obs
        obs_encoded = self.obs_encoder(obs.view(-1, obs_dim)).view(batch_size, num_agents, -1)

        # 2. 注意力机制聚合全局信息
        obs_encoded = obs_encoded.permute(1, 0, 2)  # (num_agents, batch_size, hidden_size)
        attn_output, _ = self.attention(obs_encoded, obs_encoded, obs_encoded)
        global_info = attn_output.permute(1, 0, 2).mean(dim=1)  # (batch_size, hidden_size)

        # 3. 处理历史信息（可选）
        if history_obs is not None:
            if isinstance(history_obs, np.ndarray):
                history_obs = torch.from_numpy(history_obs).to(self.device)
            history_input = history_obs.view(batch_size, -1, num_agents * obs_dim)
            history_encoded, rnn_states = self.history_lstm(history_input, rnn_states)
            hyper_input = torch.cat([global_info, history_encoded[:, -1, :]], dim=1)
        else:
            hyper_input = global_info

        # 4. 生成 Actor 权重增量
        delta_w1 = self.delta_w1(hyper_input).view(batch_size, self.actor_input_size, self.actor_hidden_size)
        delta_b1 = self.delta_b1(hyper_input)
        delta_w2 = self.delta_w2(hyper_input).view(batch_size, self.actor_hidden_size, self.actor_output_size)
        delta_b2 = self.delta_b2(hyper_input)

        # 检查编码后的观测是否非零
        print(f"[HyperNet] obs_encoded mean: {obs_encoded.mean().item()}")
        print(f"[HyperNet] attn_output mean: {attn_output.mean().item()}")
        print(f"[HyperNet] delta_w1 mean: {delta_w1.mean().item()}")
        print(f"[HyperNet] delta_b1 mean: {delta_b1.mean().item()}")

        return delta_w1, delta_b1, delta_w2, delta_b2, rnn_states

#这个类的权重将由超网络生成，而不是直接作为固定的参数存储在模型中
class DynamicActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(DynamicActor, self).__init__()
        self.hidden_size = args.hidden_size
        self.device = device

        # 获取观测和动作空间的维度
        obs_shape = get_shape_from_obs_space(obs_space)
        self.input_size = obs_shape[0] if isinstance(obs_shape, list) else obs_shape
        self.output_size = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]

        # 初始化基础权重（作为 nn.Parameter）
        self.w1 = nn.Parameter(torch.zeros(self.input_size, self.hidden_size))
        self.b1 = nn.Parameter(torch.zeros(self.hidden_size))
        self.w2 = nn.Parameter(torch.zeros(self.hidden_size, self.output_size))
        self.b2 = nn.Parameter(torch.zeros(self.output_size))

        # 如果是连续动作空间，添加 log_std
        if not hasattr(action_space, 'n'):
            self.log_std = nn.Parameter(torch.zeros(self.output_size))

        self.to(device)

    def forward(self, obs, delta_w1, delta_b1, delta_w2, delta_b2, available_actions=None):
        """
        前向传播，动态更新权重并生成动作分布。
        :param obs: (batch_size, obs_dim) 观测输入，可以是numpy数组或torch张量
        :param delta_w1: (batch_size, input_size, hidden_size) 权重增量
        :param delta_b1: (batch_size, hidden_size) 偏置增量
        :param delta_w2: (batch_size, hidden_size, output_size) 权重增量
        :param delta_b2: (batch_size, output_size) 偏置增量
        :return: 动作分布
        """
        # 将输入转换为torch张量
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        
        # 确保obs是浮点类型
        obs = obs.float()

        # 动态更新权重
        w1 = self.w1 + delta_w1
        b1 = self.b1 + delta_b1
        w2 = self.w2 + delta_w2
        b2 = self.b2 + delta_b2

        # 前向传播
        hidden = torch.relu(torch.bmm(obs.unsqueeze(1), w1).squeeze(1) + b1)
        logits = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        

        # 如果传入 available_actions，则对 logits 进行掩码处理
        if available_actions is not None:
            # 如果 available_actions 是 numpy 数组，则转换成 torch.Tensor，并将其移动到 logits 的设备上
            if isinstance(available_actions, np.ndarray):
                available_actions = torch.from_numpy(available_actions).to(logits.device)
            logits = logits.masked_fill(available_actions == 0, -1e10)


        # 根据动作空间返回分布
        if hasattr(self, 'log_std'):  # 连续动作空间
            std = torch.exp(self.log_std)
            return Normal(logits, std)
        else:  # 离散动作空间
            # 使用softmax确保概率分布合法
            probs = torch.softmax(logits, dim=-1)
            return Categorical(probs=probs)


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self._use_smooth_regularizer = args.use_smooth_regularizer
        self._use_align_regularizer = args.use_align_regularizer

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # 首先初始化网络
        self.hyper_net = HyperNetwork(args, obs_space, act_space, device)
        self.actor = DynamicActor(args, obs_space, act_space, device).to(device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        # 然后初始化优化器
        self.actor_optimizer = torch.optim.Adam(
            [
                {'params': self.hyper_net.parameters()},
                {'params': self.actor.parameters()}
            ],
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay
        )
        
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay
        )

        print(f"Actor Optimizer Parameters: {len(self.actor_optimizer.param_groups)} groups")

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    # def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
    #     # 1. 使用超网络生成权重增量
    #     delta_w1, delta_b1, delta_w2, delta_b2, rnn_states_actor = self.hyper_net(
    #         obs, 
    #         rnn_states_actor
    #     )
        
    #     # 2. 使用动态Actor生成动作
    #     dist = self.actor(obs, delta_w1, delta_b1, delta_w2, delta_b2)
        
    #     if deterministic:
    #         actions = dist.mode()
    #     else:
    #         actions = dist.sample()
        
    #     action_log_probs = dist.log_prob(actions)
        
    #     # 3. 获取critic的值估计
    #     values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        
    #     return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
    
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        # 如果 obs 是 numpy 数组，则转换成 torch 张量
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device).float()
        # 如果 available_actions 也是 numpy 数组，也转换一下（如果你后续用到的话）
        if available_actions is not None and isinstance(available_actions, np.ndarray):
            available_actions = torch.from_numpy(available_actions).to(self.device)

        # 如果 obs 是二维 (batch, obs_dim)，扩展成 (batch, 1, obs_dim)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)
        
        batch_size, num_agents, obs_dim = obs.shape
        # 1. 使用超网络生成权重增量（超网络的输出为每个环境一组参数）
        delta_w1, delta_b1, delta_w2, delta_b2, rnn_states_actor = self.hyper_net(obs, rnn_states_actor)
        # 超网络输出的动态参数形状均为 (batch, input_size, hidden_size) 等，
        # 但对于每个环境中所有智能体，我们采用同一组动态参数。
        # 为了方便后续按 agent 处理，将其扩展至 (batch, num_agents, ...)
        delta_w1 = delta_w1.unsqueeze(1).expand(batch_size, num_agents, -1, -1)
        delta_b1 = delta_b1.unsqueeze(1).expand(batch_size, num_agents, -1)
        delta_w2 = delta_w2.unsqueeze(1).expand(batch_size, num_agents, -1, -1)
        delta_b2 = delta_b2.unsqueeze(1).expand(batch_size, num_agents, -1)

        actions = []
        action_log_probs = []
        
        # 2. 对每个智能体分别调用动态 actor，同时传入对应的 available_actions
        for i in range(num_agents):
            agent_obs = obs[:, i, :]  # (batch, obs_dim)
            if available_actions is not None:
            # 如果 available_actions 为二维，则直接使用；如果为三维，则取对应 agent 的掩码
                if available_actions.dim() == 2:
                    agent_available_actions = available_actions
                else:
                    agent_available_actions = available_actions[:, i, :]
            else:
                agent_available_actions = None
            # 调用 actor.forward，注意提取当前智能体对应的动态参数
            dist = self.actor(agent_obs,
                            delta_w1[:, i, :, :],
                            delta_b1[:, i, :],
                            delta_w2[:, i, :, :],
                            delta_b2[:, i, :],
                            available_actions=agent_available_actions)

            # 根据是否确定性策略选择动作
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            actions.append(action)
            action_log_probs.append(dist.log_prob(action))
        
        # 3. 将各 agent 的输出堆叠，形成 (batch, num_agents)
        actions = torch.stack(actions, dim=1)
        action_log_probs = torch.stack(action_log_probs, dim=1)

        # 4. 获取 Critic 的值估计
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic


    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                        available_actions=None, active_masks=None):
        # 1. 使用超网络生成权重增量
        delta_w1, delta_b1, delta_w2, delta_b2, rnn_states_actor = self.hyper_net(
            obs, 
            rnn_states_actor
        )

        # 2. 使用动态Actor评估动作
        dist = self.actor(obs, delta_w1, delta_b1, delta_w2, delta_b2, available_actions=available_actions)

        # 处理 action 确保是 torch.Tensor
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        
        # 确保 action 是整数类型
        action = action.long()

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()
        
        # 3. 获取critic的值估计
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)  # 确保 Critic 也返回 rnn_states_critic
        # ✅ 确保返回 8 个值
        return values, action_log_probs, dist_entropy, rnn_states_actor, rnn_states_critic, delta_w1, delta_b1, delta_w2, delta_b2

    
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        输入：
            obs: (batch_size, num_agents, obs_dim)
            rnn_states_actor: (batch_size, num_agents, hidden_size)
            masks: (batch_size, num_agents)
            available_actions: (batch_size, num_agents, num_actions) 可用动作的掩码
            deterministic: 是否使用确定性策略
        输出：
            actions: (batch_size, num_agents) 整数张量
            action_log_probs: (batch_size, num_agents) 浮点张量
            rnn_states_actor: 更新后的RNN状态
        """
        # 生成权重增量
        delta_w1, delta_b1, delta_w2, delta_b2, rnn_states_actor = self.hyper_net(obs, rnn_states_actor)

        actions = []
        action_log_probs = []
        for i in range(obs.shape[1]):  # num_agents
            agent_obs = obs[:, i, :]

            # 提取当前 agent 的 available_actions，形状为 [batch_size, num_actions]
            agent_available_actions = available_actions[:, i, :] if available_actions is not None else None

            # 直接将 agent_available_actions 传入 actor
            # dist = self.actor(agent_obs, delta_w1, delta_b1, delta_w2, delta_b2, available_actions=agent_available_actions)
            # 调用 actor 时传入 available_actions（确保 DynamicActor.forward 中正确处理）
            dist = self.actor(agent_obs,
                              delta_w1[:, i, :, :],
                              delta_b1[:, i, :],
                              delta_w2[:, i, :, :],
                              delta_b2[:, i, :],
                              available_actions=agent_available_actions)

            # # 处理可用动作掩码
            # if available_actions is not None:
            #     agent_available_actions = available_actions[:, i]
            #     masked_probs = dist.probs * agent_available_actions
            #     masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-10)
            #     dist = Categorical(probs=masked_probs)

            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            # 确保动作是标量
            action = action.squeeze(-1)

            # # 检查生成的动作是否在可用范围内
            # if available_actions is not None:
            #     action = action * agent_available_actions.argmax(dim=-1)  # 选择可用动作

            # 打印调试信息
            print(f"Agent {i} - Generated Action: {action}, Available Actions: {agent_available_actions}")

            action_log_prob = dist.log_prob(action)
            actions.append(action)
            action_log_probs.append(action_log_prob)

        actions = torch.stack(actions, dim=1)
        action_log_probs = torch.stack(action_log_probs, dim=1)

        # 确保动作是整数类型
        actions = actions.long()

        return actions, action_log_probs, rnn_states_actor