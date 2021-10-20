#import pybullet_envs
#import gym
import gfootball.env as football
import numpy as np
from sac import Agent
from plot import learning_curve
from gym import wrappers

if __name__ == '__main__':
    #env = gym.make('InvertedPendulumBulletEnv-v0')
    env = football.create_environment(
        env_name="11_vs_11_stochastic", representation='simple115v2', render=False, rewards='scoring,checkpoints',write_full_episode_dumps=True,logdir="tmp/video")
    # agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=1)
    n_games = 5
    env = wrappers.Monitor(env, 'SAC/tmp/video', force=True)
    filename = 'Google.png'
    step = 0
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn(step=step)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save()
        step += 1

        print('episodio: ', i, 'recompensa: %.1f' %
              score, 'recompensa media: %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        learning_curve(x, score_history, figure_file)
