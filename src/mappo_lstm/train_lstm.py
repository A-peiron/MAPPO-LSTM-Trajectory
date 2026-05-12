import os
import shutil
from time import sleep
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from ma_gym.envs.combat.combat import Combat
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 清空日志文件和绘图文件夹
log_file = "training_log_metrics_weight.txt"
plot_dir = "plots_metrics_weight"
if os.path.exists(log_file):
    open(log_file, "w").close()
if os.path.exists(plot_dir):
    shutil.rmtree(plot_dir)
os.makedirs(plot_dir)

# 日志记录函数
def log_message(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

def plot_all_metrics(metrics_dict, episode):
    """
    将所有指标绘制到一个包含多个子图的图表中
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics (Up to Episode {episode})', fontsize=16)
    axes = axes.flatten()
    any_metric = list(metrics_dict.values())[0]
    x_values = [50 * (i + 1) for i in range(len(any_metric))]
    window_size = min(5, len(x_values)) if len(x_values) > 0 else 1

    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        if i >= 6:
            break
        ax = axes[i]
        values_array = np.array(values)
        if len(values) > window_size:
            smoothed = np.convolve(values_array, np.ones(window_size)/window_size, mode='valid')
            std_values = np.array([np.std(values_array[j:j+window_size]) for j in range(len(values) - window_size + 1)])
            smoothed_x = x_values[window_size-1:]
            ax.plot(smoothed_x, smoothed, '-', linewidth=2, label='Smoothed')
            ax.scatter(x_values, values, alpha=0.3, label='Original')
            ax.fill_between(smoothed_x, smoothed-std_values, smoothed+std_values, alpha=0.2, label='±1 StdDev')
        else:
            ax.plot(x_values, values, 'o-', label='Data')
        ax.set_title(metric_name.replace('_', ' '))
        ax.set_xlabel('Episodes')
        ax.set_ylabel(metric_name.replace('_', ' '))
        ax.grid(True, alpha=0.3)
        ax.legend()
    if len(metrics_dict) < 6:
        for j in range(len(metrics_dict), 6):
            fig.delaxes(axes[j])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plot_dir, f'training_metrics.png'))
    plt.close(fig)

def compute_entropy(probs):
    dist = torch.distributions.Categorical(probs)
    return dist.entropy().mean().item()

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class PolicyNet(torch.nn.Module):
    """
    Actor 网络，包含带注意力的 LSTM 轨迹预测头
    输入: 单个智能体的当前状态
    输出: 对应动作分布和未来 pred_steps 步内所有智能体状态的预测
    """
    def __init__(self, state_dim, hidden_dim, action_dim, team_size, pred_steps, attn_heads=4):
        super(PolicyNet, self).__init__()
        self.state_dim = state_dim
        self.team_size = team_size
        self.pred_steps = pred_steps
        self.hidden_dim = hidden_dim

        # 策略网络部分
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)  # 策略输出头

        # LSTM 轨迹预测模块
        self.h0 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.c0 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.pred_lstm = torch.nn.LSTMCell(hidden_dim, hidden_dim)
        self.pred_fc = torch.nn.Linear(hidden_dim, team_size * state_dim)  # 每步预测输出

        # 智能体间注意力机制，用于对预测状态进行交互建模
        self.agent_embed = torch.nn.Linear(state_dim, hidden_dim)
        self.agent_attn = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attn_heads, batch_first=True)
        self.agent_unembed = torch.nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        # x: [batch, state_dim]
        # 策略网络前向
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        action_probs = F.softmax(self.fc3(x), dim=1)

        # 轨迹预测：使用 LSTMCell 生成未来 pred_steps 步
        h = self.h0(x)
        c = self.c0(x)
        batch_size = x.size(0)
        predictions = []
        zero_input = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        for _ in range(self.pred_steps):
            h, c = self.pred_lstm(zero_input, (h, c))
            pred_step = self.pred_fc(h)  # [batch, team_size*state_dim]
            pred_step = pred_step.view(batch_size, self.team_size, self.state_dim)  # [batch, team_size, state_dim]

            # 应用智能体间自注意力
            agent_embeds = self.agent_embed(pred_step)  # [batch, team_size, hidden_dim]
            attn_output, _ = self.agent_attn(agent_embeds, agent_embeds, agent_embeds)  # 自注意力
            attn_output = self.agent_unembed(attn_output)  # [batch, team_size, state_dim]

            predictions.append(attn_output)

        # 拼接成 [batch, pred_steps, team_size, state_dim]
        pred = torch.stack(predictions, dim=1)
        return action_probs, pred

class CentralValueNet(torch.nn.Module):
    """
    全局价值网络 (Critic)
    输入: 所有智能体状态拼接 [batch, team_size*state_dim]
    输出: 每个智能体的价值估计 [batch, team_size]
    """
    def __init__(self, total_state_dim, hidden_dim, team_size):
        super(CentralValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(total_state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, team_size)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)

class MAPPO:
    def __init__(self, team_size, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, lmbda, eps, gamma, device,
                 pred_steps=5, aux_weight=0.1, info_weight=0.1, tau=0.07): 
        self.team_size = team_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.device = device
        self.aux_weight = aux_weight    # 轨迹预测损失权重
        self.info_weight = info_weight  # InfoNCE 损失权重
        self.pred_steps = pred_steps
        self.tau = tau                # 对比损失温度系数

        # 为每个智能体创建独立的 Actor，包括预测部分
        self.actors = [PolicyNet(state_dim, hidden_dim, action_dim, team_size, pred_steps).to(device)
                       for _ in range(team_size)]
        # 全局 Critic
        self.critic = CentralValueNet(team_size * state_dim, hidden_dim, team_size).to(device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=actor_lr)
                                 for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def save_model(self, path="mappo_weights"):
        if not os.path.exists(path):
            os.makedirs(path)
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(path, f"actor_{i}.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def load_model(self, path="mappo_weights"):
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(path, f"actor_{i}.pth")
            if os.path.exists(actor_path):
                actor.load_state_dict(torch.load(actor_path))
        critic_path = os.path.join(path, "critic.pth")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))

    def take_action(self, state_per_agent):
        actions = []
        action_probs = []
        for i, actor in enumerate(self.actors):
            s = torch.tensor([state_per_agent[i]], dtype=torch.float).to(self.device)
            probs, _ = actor(s)  # 只需要策略输出
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            actions.append(action.item())
            action_probs.append(probs.detach().cpu().numpy()[0])
        return actions, action_probs

    def update(self, transition_dicts, state_dim):
        T = len(transition_dicts[0]['states'])
        # 构建全局状态序列，用于 Critic 计算
        states_all = []
        next_states_all = []
        for t in range(T):
            concat_state = []
            concat_next = []
            for i in range(self.team_size):
                concat_state.append(transition_dicts[i]['states'][t])
                concat_next.append(transition_dicts[i]['next_states'][t])
            states_all.append(np.concatenate(concat_state))
            next_states_all.append(np.concatenate(concat_next))
        states_all = torch.tensor(np.array(states_all), dtype=torch.float).to(self.device)      # [T, team_size*state_dim]
        next_states_all = torch.tensor(np.array(next_states_all), dtype=torch.float).to(self.device)  # [T, team_size*state_dim]
        rewards_all = torch.tensor([[transition_dicts[i]['rewards'][t] for i in range(self.team_size)] for t in range(T)],
                                    dtype=torch.float).to(self.device)    # [T, team_size]
        dones_all = torch.tensor([[transition_dicts[i]['dones'][t] for i in range(self.team_size)] for t in range(T)],
                                    dtype=torch.float).to(self.device)    # [T, team_size]

        # Critic 估值
        values = self.critic(states_all)        # [T, team_size]
        next_values = self.critic(next_states_all)  # [T, team_size]
        td_target = rewards_all + self.gamma * next_values * (1 - dones_all)
        td_delta = td_target - values

        # 计算优势函数
        advantages = []
        for i in range(self.team_size):
            adv = compute_advantage(self.gamma, self.lmbda, td_delta[:, i]).to(self.device)
            advantages.append(adv)

        # 更新 Critic
        critic_loss = F.mse_loss(values, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor（带预测损失）
        action_losses = []
        entropies = []
        pred_losses = []
        for i in range(self.team_size):
            states = torch.tensor(np.array(transition_dicts[i]['states']), dtype=torch.float).to(self.device)  # [T, state_dim]
            actions = torch.tensor(transition_dicts[i]['actions']).view(-1, 1).to(self.device)               # [T, 1]
            old_probs = torch.tensor(np.array(transition_dicts[i]['action_probs']), dtype=torch.float).to(self.device)  # [T, action_dim]

            current_probs, pred_out = self.actors[i](states)  # 策略概率和预测输出
            log_probs = torch.log(current_probs.gather(1, actions))
            old_log_probs = torch.log(old_probs.gather(1, actions)).detach()
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages[i].unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages[i].unsqueeze(-1)
            action_loss = torch.mean(-torch.min(surr1, surr2))
            entropy_val = compute_entropy(current_probs)

            # 计算轨迹预测损失 (MSE + InfoNCE)
            # pred_out: [T, pred_steps, team_size, state_dim]
            pred_out_flat = pred_out.view(pred_out.size(0), -1)  # [T, pred_steps*team_size*state_dim]
            if T >= self.pred_steps:
                target_list = []
                for t in range(T - self.pred_steps + 1):
                    future_states = []
                    for k in range(self.pred_steps):
                        future_states.append(next_states_all[t + k])  # [team_size*state_dim]
                    traj = torch.cat(future_states, dim=-1)  # [team_size*state_dim*pred_steps]
                    target_list.append(traj)
                if target_list:
                    target_tensor = torch.stack(target_list)  # [T - pred_steps + 1, team_size*state_dim*pred_steps]
                    pred_subset = pred_out_flat[:target_tensor.size(0)]
                    mse_loss = F.mse_loss(pred_subset, target_tensor)
                    # InfoNCE 损失：最大化预测与真实序列的相似度
                    pred_vecs = pred_subset
                    actual_vecs = target_tensor
                    sim_matrix = torch.matmul(pred_vecs, actual_vecs.T) / self.tau  # [N, N]
                    labels = torch.arange(sim_matrix.size(0)).to(self.device)
                    info_loss = F.cross_entropy(sim_matrix, labels)
                    pred_loss = mse_loss + self.info_weight * info_loss
                else:
                    pred_loss = torch.tensor(0.0, device=self.device)
            else:
                pred_loss = torch.tensor(0.0, device=self.device)

            total_loss = action_loss + self.aux_weight * pred_loss
            self.actor_optimizers[i].zero_grad()
            total_loss.backward()
            self.actor_optimizers[i].step()

            action_losses.append(action_loss.item())
            entropies.append(entropy_val)
            pred_losses.append(pred_loss.item())

        return np.mean(action_losses), critic_loss.item(), np.mean(entropies), np.mean(pred_losses)

# 参数设置
actor_lr = 3e-4
critic_lr = 1e-3
total_episodes = 150000
hidden_dim = 64
gamma = 0.99
lmbda = 0.97
eps = 0.3
team_size = 5
grid_size = (20, 20)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# 创建环境
env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n

# 创建带有 LSTM 预测模块的 MAPPO 智能体
pred_steps = 5  # 预测步数超参数
mappo = MAPPO(team_size, state_dim, hidden_dim, action_dim,
              actor_lr, critic_lr, lmbda, eps, gamma, device, pred_steps=pred_steps)

# 训练指标记录
total_rewards_per_episode = []
episode_lengths = []
policy_losses = []
value_losses = []
entropies = []
pred_losses = []

avg_total_rewards_per_50 = []
avg_episode_length_per_50 = []
avg_policy_loss_per_50 = []
avg_value_loss_per_50 = []
avg_entropy_per_50 = []
avg_pred_loss_per_50 = []

with tqdm(total=total_episodes, desc="Training") as pbar:
    for episode in range(1, total_episodes + 1):
        # 初始化每个智能体的轨迹缓冲区
        buffers = [{'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_probs': []}
                   for _ in range(team_size)]
        s = env.reset()
        terminal = False
        episode_reward = 0.0
        steps = 0

        while not terminal:
            steps += 1
            actions, prob_dists = mappo.take_action(s)
            next_s, r, done, info = env.step(actions)
            step_reward = sum(r)
            episode_reward += step_reward
            # 存储转换
            for i in range(team_size):
                buffers[i]['states'].append(np.array(s[i]))
                buffers[i]['actions'].append(actions[i])
                buffers[i]['next_states'].append(np.array(next_s[i]))
                buffers[i]['rewards'].append(r[i])
                buffers[i]['dones'].append(float(done[i]))
                buffers[i]['action_probs'].append(prob_dists[i])
            s = next_s
            terminal = all(done)

        a_loss, c_loss, ent, pred_loss = mappo.update(buffers, state_dim)
        total_rewards_per_episode.append(episode_reward)
        episode_lengths.append(steps)
        policy_losses.append(a_loss)
        value_losses.append(c_loss)
        entropies.append(ent)
        pred_losses.append(pred_loss)

        # 每500集保存模型
        if episode % 500 == 0:
            mappo.save_model()
            log_message(f"Model saved at episode {episode}")

        # 每50集统计并保存指标
        if episode % 50 == 0:
            avg_reward_50 = np.mean(total_rewards_per_episode[-50:])
            avg_length_50 = np.mean(episode_lengths[-50:])
            avg_policy_loss_50 = np.mean(policy_losses[-50:])
            avg_value_loss_50 = np.mean(value_losses[-50:])
            avg_entropy_50 = np.mean(entropies[-50:])
            avg_pred_loss_50 = np.mean(pred_losses[-50:])

            avg_total_rewards_per_50.append(avg_reward_50)
            avg_episode_length_per_50.append(avg_length_50)
            avg_policy_loss_per_50.append(avg_policy_loss_50)
            avg_value_loss_per_50.append(avg_value_loss_50)
            avg_entropy_per_50.append(avg_entropy_50)
            avg_pred_loss_per_50.append(avg_pred_loss_50)

            log_message(f"Episode {episode}: "
                        f"AvgTotalReward(last50)={avg_reward_50:.3f}, "
                        f"AvgEpisodeLength(last50)={avg_length_50:.3f}, "
                        f"AvgPolicyLoss(last50)={avg_policy_loss_50:.3f}, "
                        f"AvgValueLoss(last50)={avg_value_loss_50:.3f}, "
                        f"AvgEntropy(last50)={avg_entropy_50:.3f}, "
                        f"AvgPredLoss(last50)={avg_pred_loss_50:.3f}")

            metrics_dict = {
                "Average_Total_Reward": avg_total_rewards_per_50,
                "Average_Episode_Length": avg_episode_length_per_50,
                "Average_Policy_Loss": avg_policy_loss_per_50,
                "Average_Value_Loss": avg_value_loss_per_50,
                "Average_Entropy": avg_entropy_per_50,
                "Average_Prediction_Loss": avg_pred_loss_per_50
            }
            plot_all_metrics(metrics_dict, episode)
        pbar.update(1)
