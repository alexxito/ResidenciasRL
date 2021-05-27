# numero maxismo de episodios
# epsilon_min: vamos aprendiendo mientras el incremento de aprendizaje sea mayor a este valor
# STEPS_PER_EPISODE: numero maximo de pasos a realizar en cada episodio
# ALPHA: ratio de aprendizaje del agente
# GAMMA: factor de descuento del agente
# NUM_DISCRETE_BINS: numero de diviisiones en el caso de discretizar el espacio continuo

import numpy as np
import gym

STEPS_PER_EPISODE = 200
MAX_NUM_EPISODES = 50000
EPSILON_MIN = 0.005
MAX_NUM_STEPS = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / MAX_NUM_STEPS
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30


class Qlearner(object):
    def __init__(self, environment):
        self.obs_shape = environment.observation_space.shape #Espacio de estados
        self.obs_high = environment.observation_space.high #Valor máximo de estados
        self.obs_low = environment.observation_space.low #valor minimo de estados
        self.obs_bins = NUM_DISCRETE_BINS
        self.obs_width = (self.obs_high-self.obs_low)/self.obs_bins #extensión de observación

        self.action_shape = environment.action_space.n #espacion de acciones
        self.Q = np.zeros((self.obs_bins+1, self.obs_bins+1, self.action_shape))  # matriz 31 x 31 x 3
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0 #error inicial

    def get_action(self, obs):
        discrete_obs = self.discretize(obs)
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discrete_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def discretize(self, obs):
        return tuple(((obs-self.obs_low) / self.obs_width).astype(int))

    def learn(self, obs, action, reward, next_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.max(self.Q[discrete_next_obs])
        td_error = td_target - self.Q[discrete_obs][action]
        self.Q[discrete_obs][action] += self.alpha*td_error


def train(agent, environment):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = environment.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, info = environment.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episodio {} con recompensa: {}, mejor recompensa: {}, epsilon: {}".format(episode, total_reward, best_reward, agent.epsilon))
    return np.argmax(agent.Q, axis=2)


def test(agent, environment, policy):
    done = False
    obs = environment.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)] #accion que dictamina la politica que hemos entrenado
        next_obs, reward, done, info = environment.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    environment = gym.make("MountainCar-v0")
    agent = Qlearner(environment)
    learned_policy = train(agent, environment)
    monitor_path = './monitor_output2' 
    environment = gym.wrappers.Monitor(environment, monitor_path, force=True)
    environment.render()
    for _ in range(1000):
        test(agent, environment, learned_policy)
    environment.close()