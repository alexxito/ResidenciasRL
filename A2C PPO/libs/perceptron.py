import torch

class SLP(torch.nn.Module):
    """
    Single Leyer Perceptron para aproximar funciones
    """
    def __init__(self, input_shape, output_shape, alpha, device=torch.device("cuda:0")):
        """

        Args:
            input_shape (Tuple): Tamaño de los datos de entrada
            output_shape (Tuple): Tamaño de los datos de salida
            device: Defaults to torch.device('CPU').
        """
        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape) 
        self.out = torch.nn.Linear(self.hidden_shape, output_shape)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, x):
        # x = torch.from_numpy(x).float().to(self.device)
        x = torch.Tensor(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x)) # Función de activación RELU
        x = self.out(x)
        return x
