from collections import namedtuple
import random

Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'next_obs', 'done'])


class ExperienceMemory(object):

    def __init__(self, capacity=int(1e6)):
        """Memmoria que acumulara las experiencias del agente

        Args:
            capacity ([int], optional): Capacidad de almacenamiento de la memoria cíclica del agente. Defaults to int(1e6).
        """
        self.capacity = capacity
        self.memory_idx = 0  # identificador de la experiencia actual
        self.memory = []

    def sample(self, batch_size):
        """[summary]

        Args:
            batch_size (int): Tamaño de la memoria a recuperar

        Returns:
            list: Una muestra aleatoria del tamaño de batch_size de experiencias de la memoria
        """
        assert batch_size <= self.get_size(), "El tamaño de la muestra es mayor al tamaño de la memoria"
        return random.sample(self.memory, batch_size)

    def get_size(self):
        """

        Returns:
            int: Devuelve el tamaño de la memoria
        """
        return len(self.memory)

    def store(self, exp):
        """

        Args:
            exp (object): Objeto experiencia a ser almacenado en memoria
        """
        self.memory.insert(self.memory_idx % self.capacity, exp)
        self.memory_idx += 1
