import datetime
import random
from argparse import ArgumentParser

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import environments.atari as Atari
import environments.utils as env_utils
from libs.cnn import CNN
from libs.perceptron import SLP
from utils.decay_schedule import LinearDecaySchedule
from utils.expericence_memory import ExperienceMemory, Experience
from utils.params_manager import ParamsManager

# Parseador de argumentos
args = ArgumentParser('DeepQlearning')
args.add_argument('--params-file', help='Path del fichero JSON', default='parameters.json', metavar='PFILE')
args.add_argument('--env', help='Entorno de Atari disponible, por defecto SeaquestNoFrameskip-v4',
                  default='SeaquestNoFrameskip-v4', metavar='ENV')
args.add_argument('--test', help='Modo de testing para jugar sin realizar aprendizaje', action='store_true',
                  default=False)
args.add_argument('--render', help='Renderiza el entorno en pantalla, Desactivado por defecto', action='store_true',
                  default=False)
args.add_argument('--record', help='almacena videos y estados de la performance del agente', action='store_true',
                  default=False)
args.add_argument('--output-dir', help='Directorio para almacenar los outputs, defecto=./trained_models/results',
                  default='./trained_models/results')
args = args.parse_args()

# Parametros globales
manager = ParamsManager(args.params_file)
seed = manager.get_agent_params()['seed']
summary_file_prefix = manager.get_agent_params()['summary_filename_prefix']
summary_filename = summary_file_prefix + args.env + datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
writer = SummaryWriter(summary_filename)
manager.export_agent_params(summary_filename + '/' + 'agent_params.json')
manager.export_env_params(summary_filename + '/' + 'environment_params.json')

# contador global de entrenamiento
global_step_num = 0
# habilitar el uso de la grafica
use_cuda = manager.get_agent_params()["use_cuda"]
device = torch.device('cuda:0')
torch.manual_seed(seed)
# np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)


class DeepQlearner(object): 
    def __init__(self, obs_shape, action_shape, params):
        self.params = params
        self.gamma = self.params['gamma']
        self.learning_rate = self.params['learning_rate']
        self.best_mean_reward = -float('inf')
        self.best_reward = -float('inf')
        self.training_steps_completed = 0
        self.action_shape = action_shape
        if len(obs_shape) == 1:  # Una dimension en el espacio de dimensiones
            self.DQN = SLP
        elif len(obs_shape) == 3:  # 3 dimensiones en el espacio de dimensiones
            self.DQN = CNN

        self.Q = self.DQN(obs_shape, action_shape, device=device).to(device)
        # Optimizador para la red neuronal
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        if self.params['use_target_network']:
            self.Q_target = self.DQN(obs_shape, action_shape, device=device).to(device)

        self.policy = self.epsilon_greddy_Q
        self.epsilon_max = self.params['epsilon_max']  # error maximo
        self.epsilon_min = self.params['epsilon_min']  # error minimo
        self.epsilon_decay = LinearDecaySchedule(
            initial_value=self.epsilon_max, final_value=self.epsilon_min,
            max_steps=self.params['epsilon_decay_final_step'])

        self.step_num = 0

        self.memory = ExperienceMemory(capacity=int(self.params['experience_memory_size']))

    def get_action(self, obs):
        # obs = np.array(obs)
        obser = torch.Tensor(obs)
        # obs = obs / 255.0
        obser = obser / torch.tensor(255.0)
        if len(obser.shape) == 3:  # es una imagen
            if obser.shape[2] < obser.shape[0]:  # WxHxC -> C x H x W
                # obs = obs.reshape(obs.shape[2], obs.shape[1], obs.shape[0])
                obser = torch.reshape(obser, (obser.shape[2], obser.shape[1], obser.shape[0]))
            # obser = np.expand_dims(obser, 0)
            obser = torch.unsqueeze(obser, 0)
        return self.policy(obser)

    def epsilon_greddy_Q(self, obs):
        writer.add_scalar('DQL/epsilon', self.epsilon_decay(self.step_num), self.step_num)
        self.step_num += 1
        if random.random() < self.epsilon_decay(self.step_num) and not self.params['test']:
            action = random.choice([a for a in range(self.action_shape)])
        else:
            # action = np.argmax(self.Q(obs).data.to(device).numpy())
            action = torch.argmax(self.Q(obs).data.to(device))
        return action

    def learn(self, obs, action, reward, next_obs, done):
        if done:
            td_target = reward + 0.0
        else:
            td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        self.Q_optim.zero_grad()
        td_error.backward()
        writer.add_scalar('DQL/td_error', td_error.mean(), self.step_num)
        self.Q_optim.step()

    def replay_experience(self, batch_size=None):
        """
        Vuelve a jugar usando la experiencia aleatoria almacenada
        :param batch_size:
        :return:
        """
        batch_size = batch_size if batch_size is not None else self.params['replay_batch_size']
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)
        self.training_steps_completed += 1

    def learn_from_batch_experience(self, experiences):
        """
        Actualiza la red neuronal en base al conjunto de experiencias anteriores
        :param experiences: recuerdos anteriores
        :return:
        """
        batch_exp = Experience(*zip(*experiences))
        obs_batch = torch.tensor(batch_exp.obs) / torch.tensor(255.0)
        # obs_batch = np.array(batch_exp.obs) / 255.0
        action_batch = torch.tensor(batch_exp.action)
        # action_batch = np.array(batch_exp.action)
        reward_batch = torch.tensor(batch_exp.reward)
        # reward_batch = np.array(batch_exp.reward)
        if self.params['clip_reward']:
            # reward_batch = np.sign(reward_batch)
            reward_batch = torch.sign(reward_batch)

        # next_obs_batch = np.array(batch_exp.next_obs) / 255.0
        next_obs_batch = torch.tensor(batch_exp.next_obs) / torch.tensor(255.0)
        # done_batch = np.array(batch_exp.done)
        done_batch = torch.tensor(batch_exp.done)
        if self.params['use_target_network']:
            if self.step_num % self.params['target_network_update_frecuency'] == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
            # td_target = reward_batch + ~done_batch * \
            #            np.tile(self.gamma, len(next_obs_batch)) * \
            #            torch.max(self.Q_target(next_obs_batch), 1)[0].data.tolist()
            td_target = reward_batch + ~done_batch * torch.tile(self.gamma, (len(next_obs_batch),)) * \
                        torch.max(self.Q_target(next_obs_batch), 1)[0].data
            # td_target = torch.from_numpy(td_target)
        else:
            # td_target = reward_batch + ~done_batch * \
            #            np.tile(self.gamma, len(next_obs_batch)) * \
            #            torch.max(self.Q(next_obs_batch).detach(), 1)[0].data.tolist()
            td_target = reward_batch + ~done_batch * torch.tile(self.gamma, (len(next_obs_batch),)) * \
                        torch.max(self.Q(next_obs_batch).detach(), 1)[0].data
            # td_target = torch.from_numpy(td_target)
        td_target = td_target.to(device)
        # action_idx = torch.from_numpy(action_batch).to(device)
        action_idx = action_batch
        td_error = torch.nn.functional.mse_loss(self.Q(obs_batch).gather(
            1, action_idx.view(-1, 1)), td_target.float().unsqueeze(1))
        self.Q_optim.zero_grad()
        td_error.mean().backward()
        self.Q_optim.step()

    def save(self, env_name):
        file_name = self.params['save_dir'] + 'DQL_' + env_name + ".ptm"
        agent_state = {
            'Q': self.Q.state_dict(),
            'best_mean_reward': self.best_mean_reward,
            'best_reward': self.best_reward
        }
        torch.save(agent_state, file_name)
        print("Estado del agente guardado en:", file_name)

    def load(self, env_name):
        file_name = self.params['load_dir'] + 'DQL_' + env_name + '.ptm'
        agent_state = torch.load(file_name, map_location=lambda storage, loc: storage)
        self.Q.load_state_dict(agent_state['Q'])
        self.Q.to(device)
        self.best_mean_reward = agent_state['best_mean_reward']
        self.best_reward = agent_state['best_reward']
        print("cargando el modelo Q desde:", file_name, "con recompensa media de:", self.best_mean_reward,
              "y recompensa maxima de:", self.best_reward)


if __name__ == "__main__":
    env_conf = manager.get_environment_params()
    env_conf['env_name'] = args.env

    if args.test:
        env_conf['episodic_life'] = False
    reward_type = 'LIFE' if env_conf['episodic_life'] else 'GAME'

    custom_region_available = False
    for key, value in env_conf['useful_region'].items():
        if key in args.env:
            env_conf['useful_region'] = value
            custom_region_available = True
            break
    if custom_region_available is not True:
        env_conf['useful_region'] = env_conf['useful_region']['Default']
    print("Configuracion a utilizar:", env_conf)
    atari_env = False
    for game in Atari.get_games_list():
        if game.replace('_', '') in args.env.lower():
            atari_env = True
    if atari_env:
        env = Atari.make_env(args.env, env_conf)
    else:
        env = env_utils.ResizeReshapeFrames(gym.make(args.env))
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.n
    agent_params = manager.get_agent_params()
    agent_params['test'] = args.test
    agent_params['clip_reward'] = env_conf['clip_reward']
    agent = DeepQlearner(obs_shape, action_shape, agent_params)

    episode_rewards = list()
    previous_checkpoint_mean_ep_rew = agent.best_mean_reward
    num_improved_episodes_before_checkpoint = 0
    if agent_params['load_trained_model']:
        try:
            agent.load(env_conf['env_name'])
            previous_checkpoint_mean_ep_rew = agent.best_mean_reward
        except FileNotFoundError:
            print("Error: no existe ningun modelo entrenado")
    episode = 0
    while global_step_num < agent_params['max_training_steps']:
        obs = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        while not done:
            if env_conf['render'] or args.render:
                env.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.memory.store(Experience(obs, action, reward, next_obs, done))

            obs = next_obs
            total_reward += reward
            step += 1
            global_step_num += 1
            if done is True:
                episode += 1
                episode_rewards.append(total_reward)
                if total_reward > agent.best_reward:
                    agent.best_reward = total_reward
                if np.mean(episode_rewards) > previous_checkpoint_mean_ep_rew:
                    num_improved_episodes_before_checkpoint += 1
                if num_improved_episodes_before_checkpoint >= agent_params['save_freq']:
                    previous_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                    agent.best_mean_reward = np.mean(episode_rewards)
                    agent.save(env_conf['env_name'])
                    num_improved_episodes_before_checkpoint = 0
                print(
                    "\nEpisodio #{} finalizado con {} iteraciones. Con {} estados. Recompensa = {}, Recompensa media = {:.2f}, Mejor recompensa = {}".format(
                        episode, step + 1, reward_type, total_reward, np.mean(episode_rewards), agent.best_reward))
                writer.add_scalar('main/ep_reward', total_reward, global_step_num)
                writer.add_scalar('main/mean_ep_reward', np.mean(episode_rewards), global_step_num)
                writer.add_scalar('main/max_ep_reward', agent.best_reward, global_step_num)
                if agent.memory.get_size() >= 2 * agent_params['replay_start_size'] and not args.test:
                    agent.replay_experience()
                break
    env.close()
    writer.close()
