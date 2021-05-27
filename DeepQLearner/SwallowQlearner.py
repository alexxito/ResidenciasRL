import random
import torch
import numpy as np
from utils.decay_schedule import LinearDecaySchedule
from utils.expericence_memory import ExperienceMemory, Experience
from libs.perceptron import SLP
import gym

STEPS_PER_EPISODE = 300
MAX_NUM_EPISODES = 100000


class SwallowQlearner(object):
    def __init__(self, environment, learning_rate=0.005, gamma=0.98):
        self.obs_shape = environment.observation_space.shape  # Espacio de estados
        self.action_shape = environment.action_space.n  # espacio de acciones
        self.Q = SLP(self.obs_shape, self.action_shape)
        # Optimizador para la red neuronal
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon_max = 1.0  # error maximo
        self.epsilon_min = 0.05  # error minimo
        self.epsilon_decay = LinearDecaySchedule(
            initial_value=self.epsilon_max, final_value=self.epsilon_min, max_steps=0.5*MAX_NUM_EPISODES*STEPS_PER_EPISODE)

        self.step_num = 0
        self.policy = self.epsilon_greddy_Q
        self.memory = ExperienceMemory(capacity=int(1e5))
        self.device = torch.device("cpu")

    def get_action(self, obs):
        return self.policy(obs)

    def epsilon_greddy_Q(self, obs):
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(self.device).numpy())
        self.step_num += 1
        return action

    def learn(self, obs, action, reward, next_obs):
        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        self.Q_optim.zero_grad()
        td_error.backward()
        self.Q_optim.step()

    def replay_experience(self, batch_size):
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)

    def learn_from_batch_experience(self, experiences):
        batch_exp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_exp.obs)
        action_batch = np.array(batch_exp.action)
        reward_batch = np.array(batch_exp.reward)
        next_obs_batch = np.array(batch_exp.next_obs)
        done_batch = np.array(batch_exp.done)

        td_target = reward_batch + ~done_batch * \
            np.tile(self.gamma, len(next_obs_batch)) * \
            torch.max(self.Q(next_obs_batch).detach(), 1)[0].data.tolist()
        td_target = torch.from_numpy(td_target)
        td_target = td_target.to(self.device)
        action_idx = torch.from_numpy(action_batch).to(self.device)
        td_error = torch.nn.functional.mse_loss(self.Q(obs_batch).gather(
            1, action_idx.view(-1, 1).long()), td_target.float().unsqueeze(1))
        self.Q_optim.zero_grad()
        td_error.mean().backward()
        self.Q_optim.step()


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = SwallowQlearner(env)
    first_episode = True
    episode_rewards = list()
    for episode in range(MAX_NUM_EPISODES):
        obs = env.reset()
        total_reward = 0.0
        for step in range(STEPS_PER_EPISODE):
            # env.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.memory.store(Experience(obs, action, reward, next_obs, done))
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward

            if done is True:
                if first_episode is True:
                    max_reward = total_reward
                    first_episode = False
                episode_rewards.append(total_reward)
                if total_reward > max_reward:
                    max_reward = total_reward
                print("\nEpisodio #{} finalizado con {} iteraciones. Recompensa = {}, Recompensa media = {}, Mejor recompensa = {}".format(
                    episode, step+1, total_reward, np.mean(episode_rewards), max_reward))
                if agent.memory.get_size() > 100:
                    # muestras aleatorias de la memoria
                    agent.replay_experience(32)
                break
    env.close()
