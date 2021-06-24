import pybulletgym
import gym
import numpy as np
from sac import Agent
from plot import learning_curve

if __name__ == "__main__":
    env = gym.make("HumanoidPyBulletEnv-v0")
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
    n_games = 2000
    filename = 'Humanoid.png'
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load()
        env.render(mode='human')

    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(action)
            score += reward
            agent.remember(obs, action, reward, new_obs, done)
            if not load_checkpoint:
                agent.learn()
            obs = new_obs
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save()
        print('episodio {} recompesa {} media {}'.format(i, score, avg_score))

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        learning_curve(x, score_history, "plots/"+filename)
