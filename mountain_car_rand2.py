import gym

enviroment = gym.make("MountainCar-v0")
MAX_NUM_EPISODES = 1000

for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = enviroment.reset()
    total_reward = 0.0 #variable obtenida en casa episodio
    step = 0
    while not done:
        enviroment.render()
        action = enviroment.action_space.sample()# accion aleatoria
        next_state, reward, done, info = enviroment.step(action)
        total_reward += reward
        step += 1
        obs = next_state
    
    print("\n Episodio numero {} finalizado con {} iteraciones. Recompensa final {}".format(episode,step+1, total_reward))
enviroment.close()