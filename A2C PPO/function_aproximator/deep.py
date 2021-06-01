import torch


class DeepActor(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device=torch.device("cuda:0")):
        """
        Red neuronal neuronal convolucional que producirá dos valores continuos (media y desviación típica) para cada
        uno de los valores de output_shape usando CNN
        Representa en papel de actor
        :param input_shape: Observaciones del actor
        :param output_shape: Acciones que debe producir el actor
        :param device: Dispositivo donde se ubicará la red neuronal
        """
        super(DeepActor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=0), torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1, padding=0), torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512), torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(512, output_shape)
        self.actor_sigma = torch.nn.Linear(512, output_shape)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        # self.to(self.device)

    def forward(self, x):
        """
        Dado el valor x calculamos la media y desviación
        :param x: observación
        :return: media (mu) y desviación (sigma) para una política gaussiana
        """
        x.requires_grad = True
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        mu = self.actor_mu(x)
        sigma = self.actor_sigma(x)
        return mu, sigma


class DeepDiscreteActor(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device=torch.device("cuda:0")):
        """
        Red neuronal convolucional que utilizará  una función logistica para discriminar la acción del espacio de acciones discreto
        Representa en papel de actor es espacio discreto
        :param input_shape: Observaciones del actor
        :param output_shape: Acciones que debe producir el actor
        :param device: Dispositivo donde se ubicará la red neuronal
        """
        super(DeepDiscreteActor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=0), torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1, padding=0), torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512), torch.nn.ReLU())
        self.actor_logist = torch.nn.Linear(512, output_shape)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        # self.to(self.device)

    def forward(self, x):
        """
        Dado el valor x calculamos la acción con la función logist
        :param x: observación
        :return: logistica según la política del agente
        """
        x = x.to(self.device)
        x.requires_grad = True
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        logits = self.actor_logist(x)
        return logits


class DeepCritic(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device=torch.device("cuda:0")):
        """
        Red neuronal convolucion que producirá un valor continuo
        Representa en papel del critico
        Estima el valor de la obseración/estado actual
        :param input_shape: Observaciones del actor
        :param output_shape: Representa el feedback que producirá el critico
        :param device: Dispositivo donde se ubicará la red neuronal
        """
        super(DeepCritic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=0), torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1, padding=0), torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512), torch.nn.ReLU())
        self.critic = torch.nn.Linear(512, output_shape)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        # self.to(self.device)

    def forward(self, x):
        """
        Dado el valor x devolvemos el valor estimado de salida como críticos
        :param x: observación
        :return: valor estimado
        """
        x.requires_grad = True
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        critic = self.critic(x)
        return critic


class DeepActorCritic(torch.nn.Module):
    def __init__(self, input_shape, actor_shape, critic_shape, device=torch.device("cuda:0")):
        """
        Red neuronal convolucional que representará al actor y el crítico
        Representa en papel de actor
        :param input_shape: Observaciones del actor
        :param actor_shape_shape: Forma de los datos del actor (acciones que producirá el actor)
        :param critic_shape: Forma de los datos de salida del crítico (suele ser un solo valor)
        :param device: Dispositivo donde se ubicará la red neuronal
        """
        super(DeepActorCritic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=0), torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1, padding=0), torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512), torch.nn.ReLU())

        self.actor_mu = torch.nn.Linear(512, actor_shape)
        self.actor_sigma = torch.nn.Linear(512, actor_shape)
        self.critic = torch.nn.Linear(512, critic_shape)

    def forward(self, x):
        x.requires_grad = True
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        actor_mu = self.actor_mu(x)
        actor_sigma = self.actor_sigma(x)
        critic = self.critic(x)
        return actor_mu, actor_sigma, critic
