import datetime
from argparse import ArgumentParser
from collections import namedtuple
from random import choice

import gym
import pybulletgym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

import environments.atari as Atari
from function_aproximator.deep import DeepActor, DeepDiscreteActor, DeepCritic
from function_aproximator.swallow import Actor, DiscreteActor, Critic
from utils.params_manager import ParamsManager

# Parseador de argumentos
args = ArgumentParser('PPO')
args.add_argument('--params-file', help='Path del fichero JSON',
                  default='parameters.json', metavar='PFILE')
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
summary_filename = summary_file_prefix + args.env + \
    datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
writer = SummaryWriter(summary_filename)
manager.export_agent_params(summary_filename + '/' + 'agent_params.json')
manager.export_env_params(summary_filename + '/' + 'environment_params.json')
# habilitar el uso de la grafica
torch.cuda.init() 
use_cuda = manager.get_agent_params()["use_cuda"]
device = torch.device('cuda:0')
torch.manual_seed(seed)
# np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)

# T_t = (st, at, rt, st+1)
Transition = namedtuple('Transition', ['state', 'value_s', 'action', 'log_prob_a'])
#Transition = Experience


class PPO(mp.Process):

    def __init__(self, id, env_name, agent_params, env_params):
        super(PPO, self).__init__()
        """
            Implementación de un agente usando el algoritmo Advantage Actor Critic
            :param id: identificador para indentificar al agente en caso de que existan varios agentes
            :param env_name: nombre del entorno
            :param agent_params: parámetros que ussará el agente
            :param env_params: parámetros del entorno
        """
        self.params = agent_params
        self.id = id
        self.actor_name = "Actor " + str(self.id)
        self.env_name = env_name
        self.env_params = env_params

        self.policy = self.multi_variate_gaussian_policy
        self.gamma = self.params['gamma']
        self.trajectory = []  # contiene la trayectoria del agente como secuencia de transiciones
        self.rewards = []  # Contiene las recompensas del entorno en cada paso
        self.global_step_num = 0
        #self.memory = ExperienceMemory(capacity=int(self.params['experience_memory_size']))
        self.best_mean_reward = -float('inf')
        self.best_reward = -float('inf')
        self.save_params = False  # saber si hay parámetros guardados junto con el modelo
        # indicar si el espacio de acciones es continuo o discreto
        self.continuos_action_space = True
        #self.advs = 0

    def multi_variate_gaussian_policy(self, obs):
        """
        Calcula una distribución gaussiana multivariada del tamaño de las acciones usando las observaciones
        :param obs: observaciones del agente
        :return: una distribución sobre las acciones dadas las observaciones actuales
        """
        mu, sigma = self.actor(obs)
        value = self.critic(obs)
        [mu[:, i].clamp_(float(self.env.action_space.low[i]), float(self.env.action_space.high[i])) for i in
         range(self.action_shape)]
        sigma = torch.nn.Softplus()(sigma).squeeze() + 1e-7
        self.mu = mu.to(device)
        self.sigma = sigma.to(device)
        self.value = value.to(device)
        if len(self.mu.shape) == 0:
            # evitar que la multivariante normal de un error
            self.mu.unsqueeze_(0)
        self.action_distribution = MultivariateNormal(self.mu, torch.eye(self.action_shape).cuda() * self.sigma,
                                                      validate_args=True)
        return self.action_distribution

    def preprocess_obs(self, obs):
        obs = np.array(obs)
        if len(obs.shape) == 3:
            obs = np.reshape(obs, (obs.shape[2], obs.shape[1], obs.shape[0]))
            obs = np.resize(obs, (obs.shape[0], 84, 84))
        obs = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        return obs

    def process_actions(self, action):
        """
        :param action:
        :return:
        """
        if self.continuos_action_space:
            [action[:, i].clamp_(float(self.env.action_space.low[i]), float(self.env.action_space.high[i])) for i in
             range(self.action_shape)]
        action = action.to(device)
        return action.squeeze(0)

    def discrete_policy(self, obs):
        """
        Calcula una distribucion discreta o categorica sobre las observaciones del agente
        :param obs: observaciones del agente
        :return: politica formada por una distribicion sobre las acciones a partir de las observaciones
        """
        logits = self.actor(obs)
        value = self.critic(obs)
        self.logits = logits.to(device)
        self.value = value.to(device)
        self.action_distribution = Categorical(logits=logits)
        return self.action_distribution

    def get_action(self, obs):
        observation = self.preprocess_obs(obs)
        action_distribution = self.policy(observation)
        value = self.value
        action = action_distribution.sample()
        log_prob_a = action_distribution.log_prob(action)
        action = self.process_actions(action)
        if not self.params['test']:
            self.trajectory.append(Transition(obs, value, action, log_prob_a))
        return action

    def calculate_n_steps(self, n_steps_reward, final_state, done, gamma):
        """
        Calcula el valor de retorno dados n-pasos para cada uno de los estados de entrada
        :param n_steps_reward: Lista de las recompensas obtenidas en cada uno de los n estados
        :param final_state: Estado final tras las n iteraciones
        :param done: Variable booleana con valor True si se ha alcanzado el estado final del entorno
        :param gamma: Factor de Descuento para el cálculo de la diferencia temporal.
        :return: El valor final de cada estado de los n ejecutados
        """
        result = list()
        with torch.no_grad():
            reward = torch.tensor([[0]]).float().to(device=device) if done else self.critic(
                self.preprocess_obs(final_state)).to(device=device)
            for r_t in n_steps_reward:
                reward = torch.tensor(r_t).float() + gamma * reward
                result.insert(0, reward)
        return result

    def calculate_loss(self, trajectory, td_targets):
        """ Calcula los valores de perdida del actor, critico y de la politica
        :param trajectory: trayectoria de los valores obtenidos durante un episodio
        :param td_targets: valores objetivo durante el episodio
        :return: devuelve las perdidas del actor y del critico
        """
        td_targe = torch.tensor(td_targets).cuda()
        clip_eps = self.params['policy_clip']
        n_step_trajectory = Transition(*zip(*trajectory))
        v_s = n_step_trajectory.value_s
        log_prob_a = torch.tensor(n_step_trajectory.log_prob_a).cuda()
        if self.continuos_action_space:
            action = [a.cpu().numpy() for a in n_step_trajectory.action]
            action = torch.tensor(action).cuda()
        else:
            action = torch.tensor(n_step_trajectory.action).cuda()
        old_log_prob_a = self.action_distribution.log_prob(action).cuda()
        assert log_prob_a.shape == old_log_prob_a.shape
        ratios = torch.exp(log_prob_a - old_log_prob_a).to(device=device)
        sur_1 = ratios * td_targe                                                      
        sur_2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * td_targe
        sur_2.cuda()
        clip_loss = -torch.min(sur_1, sur_2).mean()
        entropy = self.action_distribution.entropy().mean()
        ent_penalty = -self.params['entropy_coef'] * entropy
        policy_loss = clip_loss + ent_penalty
        # actor_losses = []
        critic_losses = []
        for td_target, critic_prediction, _ in zip(td_targets, v_s, log_prob_a):
            # td_error = td_target - critic_prediction
            # actor_losses.append(-log_p_a * td_error)
            critic_losses.append(F.smooth_l1_loss(critic_prediction, td_target))
        # if self.params['use_entropy_bonus']:
        #    actor_loss = torch.stack(actor_losses).mean() - self.action_distribution.entropy().mean()
        # else:
        #    actor_loss = torch.stack(actor_losses).mean()
        critic_loss = torch.stack(critic_losses).mean()
        writer.add_scalar(self.actor_name + '/critic_loss', critic_loss, self.global_step_num)
        # writer.add_scalar(self.actor_name + '/actor_loss', actor_loss, self.global_step_num)
        writer.add_scalar(self.actor_name + '/actor_loss', policy_loss, self.global_step_num)

        return policy_loss, critic_loss

    def learn(self, n_th_observation, done):
        """
            :param n_th_observation: n-sima observacion del entorno
            :param done:             estado final del entorno
        """
        td_targets = self.calculate_n_steps(self.rewards, n_th_observation, done, self.gamma)
        actor_loss, critic_loss, = self.calculate_loss(self.trajectory, td_targets)

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.trajectory.clear()
        self.rewards.clear()

    def save(self):
        file_name = self.params['model_dir'] + 'PPO_' + self.env_name + ".ptm"
        agent_state = {
            'Actor': self.actor.state_dict(),
            'Critic': self.critic.state_dict(),
            'best_mean_reward': self.best_mean_reward,
            'best_reward': self.best_reward
        }
        torch.save(agent_state, file_name)
        print("Estado del agente guardado en:", file_name)

        if not self.save_params:
            manager.export_agent_params(file_name + '.agent_params')
            print("Los parametros del agente se han guardado en:" +
                  file_name + '.agent_params')
            self.save_params = True

    def load(self):
        file_name = self.params['model_dir'] + 'PPO_' + self.env_name + '.ptm'
        agent_state = torch.load(
            file_name, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(agent_state['Actor'])
        self.actor.to(device)
        self.critic.load_state_dict(agent_state['Critic'])
        self.critic.to(device)
        self.best_mean_reward = agent_state['best_mean_reward']
        self.best_reward = agent_state['best_reward']
        print("cargando el modelo A2C desde:", file_name, "con recompensa media de:", self.best_mean_reward,
              "y recompensa maxima de:", self.best_reward)

    def run(self):
        # Cargar datos del entorno
        custom_region_available = False
        for key, value in self.env_params['useful_region'].items():
            if key in args.env:
                self.env_params['useful_region'] = value
                custom_region_available = True
        if custom_region_available is not True:
            self.env_params['useful_region']['Default']

        atari_env = False
        for game in Atari.get_games_list():
            if game.replace("_", '') in args.env.lower():
                atari_env = True
        if atari_env:
            self.env = Atari.make_env(self.env_name, self.env_params)
        else:
            self.env = gym.make(self.env_name)
        monitor_path = './trained_models'
        self.env = gym.wrappers.Monitor(self.env, monitor_path, force=True)
        # Configurar politicas y parametros del agente y del critico
        self.state_shape = self.env.observation_space.shape

        if isinstance(self.env.action_space.sample(), int):  # Espacio discreto
            self.action_shape = self.env.action_space.n
            self.policy = self.discrete_policy
            self.continuos_action_space = False
        else:  # Espacio continuo
            self.action_shape = self.env.action_space.shape[0]
            self.policy = self.multi_variate_gaussian_policy

        self.critic_shape = 1
        if len(self.state_shape) >= 3:
            if self.continuos_action_space:
                self.actor = DeepActor(
                    self.state_shape, self.action_shape, device).to(device)
            else:
                self.actor = DeepDiscreteActor(
                    self.state_shape, self.action_shape, device).to(device)
            self.critic = DeepCritic(
                self.state_shape, self.critic_shape, device).to(device)
        else:
            if self.continuos_action_space:
                self.actor = Actor(
                    self.state_shape, self.action_shape, device).to(device)
            else:
                self.actor = DiscreteActor(
                    self.state_shape, self.action_shape, device).to(device)
            self.critic = Critic(
                self.state_shape, self.critic_shape, device).to(device)
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=self.params['learning_rate'])
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=self.params['learning_rate'])
        # Fase de entrenamiento del agente
        episode_rewards = list()
        previous_checkpoint_mean_ep_rew = self.best_mean_reward
        num_improved_episodes_before_checkpoint = 0
        if self.params['load_trained_model']:
            try:
                self.load()
                previous_checkpoint_mean_ep_rew = self.best_mean_reward
            except FileNotFoundError:
                print("Error: no existe ningun modelo entrenado")
                if args.test:
                    print(
                        "FATAL: no hay modelo guardado y no se puede proceder al modo testing")
                else:
                    print('WARNING: no hay ningun modelo para este entorno')
        for episode in range(self.params['max_num_epsiodes']):
            if args.render:
                self.env.render()
            obs = self.env.reset()
            done = False
            ep_reward = 0.0
            step_num = 0
            while not done:
                action = self.get_action(obs)
                action = action.cpu().numpy()
                next_obs, reward, done, _ = self.env.step(action)
                self.rewards.append(reward)
                ep_reward += reward
                step_num += 1

                if not args.test and (step_num > self.params['learning_step_thresh'] or done):
                    self.learn(next_obs, done)
                    step_num = 0
                    if done:
                        episode_rewards.append(ep_reward)
                        if ep_reward > self.best_reward:
                            self.best_reward = ep_reward
                        if np.mean(episode_rewards) > previous_checkpoint_mean_ep_rew:
                            num_improved_episodes_before_checkpoint += 1
                        if num_improved_episodes_before_checkpoint >= self.params['save_freq']:
                            previous_checkpoint_mean_ep_rew = np.mean(
                                episode_rewards)
                            self.best_mean_reward = np.mean(episode_rewards)
                            self.save()
                            num_improved_episodes_before_checkpoint = 0
                obs = next_obs
                self.global_step_num += 1
                # if args.render:
                #    self.env.render()
                print(
                    "\n{}: Episodio #{} Con {} recompensa, Recompensa media = {:.2f}, Mejor recompensa = {}".format(
                        self.actor_name,
                        episode, ep_reward, np.mean(episode_rewards), self.best_reward))
                writer.add_scalar(self.actor_name + '/reward',
                                  reward, self.global_step_num)
                writer.add_scalar(self.actor_name + '/ep_reward',
                                  ep_reward, self.global_step_num)
                writer.add_scalar(self.actor_name + '/mean_reward',
                                  np.mean(episode_rewards), self.global_step_num)
                writer.add_scalar(self.actor_name + '/max_ep_reward',
                                  self.best_reward, self.global_step_num)


if __name__ == "__main__":
    agent_params = manager.get_agent_params()
    agent_params['model_dir'] = args.output_dir
    agent_params['test'] = args.test
    env_params = manager.get_environment_params()
    env_params['env_name'] = args.env
    mp.set_start_method('spawn')
    agent_proc = [PPO(id, args.env, agent_params, env_params) for id in
                  range(agent_params['num_agents'])]
    [p.start() for p in agent_proc]
    [p.join() for p in agent_proc]
