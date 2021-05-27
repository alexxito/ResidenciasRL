import torch


class CNN(torch.nn.Module):

    def __init__(self, input_shape, output_shape, device=torch.device('cuda:0')):
        """[summary]

        Args:
            input_shape ([type]): Dimension de la imagen, que estara reescalada a 84x84 
            output_shape ([type]): dimension de la salida
            device ([type], optional): [description]. Defaults to torch.device('cpu').
        """
        super(CNN, self).__init__()
        self.device = device

        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(
            input_shape[0], 64, kernel_size=4, stride=2, padding=1), torch.nn.ReLU())

        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(
            64, 32, kernel_size=4, stride=2, padding=0), torch.nn.ReLU())

        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=0), torch.nn.ReLU())
        
        self.out = torch.nn.Linear(18*18*32, output_shape)

    def forward(self, x):
        # x = torch.from_numpy(x).float().to(self.device)
        x = torch.Tensor(x).float().to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x
