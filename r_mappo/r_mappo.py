import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):
        
        self.args = args  # ✅ 这里存储 args，防止 AttributeError

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        # 添加正则化相关的属性
        self._use_smooth_regularizer = args.use_smooth_regularizer
        self._use_align_regularizer = args.use_align_regularizer

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # # 调用 evaluate_actions 并接收所有返回值
        # values, action_log_probs, dist_entropy, delta_w1, delta_b1, delta_w2, delta_b2, new_dists = \
        #     self.policy.evaluate_actions(share_obs_batch, obs_batch, rnn_states_batch, 
        #                                 rnn_states_critic_batch, actions_batch, masks_batch, 
        #                                 available_actions_batch, active_masks_batch)
        
        values, action_log_probs, dist_entropy, rnn_states_actor, rnn_states_critic, delta_w1, delta_b1, delta_w2, delta_b2 = self.policy.evaluate_actions(
            sample[0],  # cent_obs
            sample[1],  # obs
            sample[2],  # rnn_states_actor
            sample[3],  # rnn_states_critic
            sample[4],  # actions
            sample[5],  # masks
            sample[6],  # available_actions
            sample[7]   # active_masks
        )


        # 计算 PPO 策略损失
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss
    
        # 强制打印 delta_w1 的均值（无论是否启用正则化）
        print(f"[Debug] delta_w1 mean: {delta_w1.mean().item()}")
        
        # # 添加正则化
        # if self._use_smooth_regularizer:
        #     delta_theta = torch.cat([delta_w1.flatten(), delta_b1.flatten(), delta_w2.flatten(), delta_b2.flatten()])
        #     if not hasattr(self, 'delta_theta_prev'):
        #         self.delta_theta_prev = torch.zeros_like(delta_theta, device=self.device)
        #     l_smooth = torch.norm(delta_theta - self.delta_theta_prev, p=2)

        #     # 强制打印 l_smooth（即使未启用正则化）
        # delta_theta = torch.cat([delta_w1.flatten(), delta_b1.flatten(), delta_w2.flatten(), delta_b2.flatten()])
        # l_smooth = torch.norm(delta_theta - self.delta_theta_prev, p=2)
        # print(f"[Debug] l_smooth = {l_smooth.item()}")
        # print(f"[Debug] Smooth Loss: {l_smooth.item()}")  # 调试输出

        # self.delta_theta_prev = delta_theta.detach()
        # policy_loss += 0.1 * l_smooth  # 平滑正则化权重 0.1

                # 添加平滑正则化（归一化处理）
        if self._use_smooth_regularizer:
            # 拼接所有权重增量为一个向量
            delta_theta = torch.cat([delta_w1.flatten(), delta_b1.flatten(), delta_w2.flatten(), delta_b2.flatten()])
            # 如果 delta_theta_prev 未初始化，则初始化为全零向量
            if not hasattr(self, 'delta_theta_prev'):
                self.delta_theta_prev = torch.zeros_like(delta_theta, device=self.device)
            # 计算 L2 范数
            l_smooth = torch.norm(delta_theta - self.delta_theta_prev, p=2)
            # 对平滑损失进行归一化：除以参数总数的平方根
            num_params = delta_theta.numel()
            l_smooth_normalized = l_smooth / (num_params ** 0.5)
            print(f"[Debug] l_smooth (normalized) = {l_smooth_normalized.item()}")
        else:
            l_smooth_normalized = torch.tensor(0.0, device=self.device)

        # 更新 delta_theta_prev
        self.delta_theta_prev = delta_theta.detach()
        # 将归一化后的平滑正则化项加到策略损失中，使用 args.smooth_weight
        policy_loss += self.args.smooth_weight * l_smooth_normalized


        # if self._use_align_regularizer:
        #     # 这里需要获取 old_dists，假设从 buffer 中提供（需要额外修改 buffer）
        #     # 临时解决方案：跳过对齐正则化，或从 old_action_log_probs_batch 重建
        #     kl_div = torch.tensor(0.0, device=self.device)
        #     # 假设 old_dists 已从 sample 中获取
        #     for old_dist, new_dist in zip(old_dists, new_dists):
        #         kl_div += torch.distributions.kl_divergence(old_dist, new_dist).mean()
        #     l_align = kl_div / len(new_dists) if len(new_dists) > 0 else torch.tensor(0.0, device=self.device)
        #     policy_loss += 0.05 * l_align  # 对齐正则化权重 0.05

        if self._use_align_regularizer:
            # 暂时注释掉KL计算，仅测试平滑正则化
            # kl_div = torch.tensor(0.0, device=self.device)
            # for old_dist, new_dist in zip(old_dists, new_dists):
            #     kl_div += torch.distributions.kl_divergence(old_dist, new_dist).mean()
            # l_align = kl_div / len(new_dists)
            l_align = torch.tensor(0.0, device=self.device)
            policy_loss += self.args.align_weight * l_align

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()
        
        # 打印调试信息
        print(f"policy_loss: {policy_loss}, l_smooth: {l_smooth}, l_align: {l_align}")

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        # 计算优势
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
