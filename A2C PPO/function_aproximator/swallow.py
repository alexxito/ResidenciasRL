import torch


class Actor(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device=torch.device("cuda:0")):
        """
        Red neuronal que producirá dos valores continuos (media y desviación típica) para cada uno de los valores de
        output_shape
        Representa en papel de actor
        :param input_shape: Observaciones del actor
        :param output_shape: Acciones que debe producir el actor
        :param device: Dispositivo donde se ubicará la red neuronal
        """
        super(Actor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 64), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(32, output_shape)
        self.actor_sigma = torch.nn.Linear(32, output_shape)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        # self.to(self.device)

    def forward(self, x):
        """
        Dado el valor x calculamos la media y desviación
        :param x: observación
        :return: media (mu) y desviación (sigma) para una política gaussiana
        """
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        mu = self.actor_mu(x)
        sigma = self.actor_sigma(x)
        return mu, sigma


class DiscreteActor(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device=torch.device("cuda:0")):
        """
        Red neuronal que utilizará  una función logistica para discriminar la acción del espacio de acciones discreto
        Representa en papel de actor es espacio discreto
        :param input_shape: Observaciones del actor
        :param output_shape: Acciones que debe producir el actor
        :param device: Dispositivo donde se ubicará la red neuronal
        """
        super(DiscreteActor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 64), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU())
        self.actor_logist = torch.nn.Linear(32, output_shape)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        # self.to(self.device)

    def forward(self, x):
        """
        Dado el valor x calculamos la acción con la función logist
        :param x: observación
        :return: logistica según la política del agente
        """
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        logits = self.actor_logist(x)
        return logits


class Critic(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device=torch.device("cuda:0")):
        """
        Red neuronal que producirá un valor continuo
        Representa en papel del critico
        Estima el valor de la obseración/estado actual
        :param input_shape: Observaciones del actor
        :param output_shape: Representa el feedback que producirá el critico
        :param device: Dispositivo donde se ubicará la red neuronal
        """
        super(Critic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 64), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU())
        self.critic = torch.nn.Linear(32, output_shape)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        # self.to(self.device)

    def forward(self, x):
        """
        Dado el valor x devolvemos el valor estimado de salida como críticos
        :param x: observación
        :return: valor estimado
        """
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        critic = self.critic(x)
        return critic


class ActorCritic(torch.nn.Module):
    def __init__(self, input_shape, actor_shape, critic_shape, device=torch.device("cuda:0")):
        """
        Red neuronal que representará al actor y el crítico
        Representa en papel de actor
        :param input_shape: Observaciones del actor
        :param actor_shape: Forma de los datos del actor (acciones que producirá el actor)
        :param critic_shape: Forma de los datos de salida del crítico (suele ser un solo valor)
        :param device: Dispositivo donde se ubicará la red neuronal
        """
        super(ActorCritic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 32), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(16, actor_shape)
        self.actor_sigma = torch.nn.Linear(32, actor_shape)
        self.critic = torch.nn.Linear(16, critic_shape)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        # self.to(self.device)

    def forward(self, x):
        x.require_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        actor_mu = self.actor_mu(x)
        actor_sigma = self.actor_sigma(x)
        critic = self.critic(x)
        return actor_mu, actor_sigma, critic
