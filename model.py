import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyNet(nn.Module):

    def __init__(self, K=256):
        super(ToyNet, self).__init__()
        self.K = K

        self.encode = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2*self.K)
            )

        self.decode = nn.Sequential(
            nn.Linear(self.K, 10)
            )
        
        self.apply(self._init_weights) #Not always necessary

    def forward(self, x, 
                num_sample=1, stab_factor = 5,
                beta = 1):
        # Flatten the input for the MLP 
        x = x.flatten(start_dim = 1)

        # Encode the inpute
        statistics = self.encode(x)

        # Extract the Distribution Discriptor
        mu, raw_std = statistics[:,:self.K], statistics[:,self.K:]
        std = F.softplus(raw_std-stab_factor,
                         beta=beta)
        
        # Encode (Random Sampling) -> Decode
        encoding = self.reparametrize_n(mu,std,
                                        num_sample)
        logits = self.decode(encoding)

        if num_sample > 1:
            logits = F.softmax(logits, dim=-1).mean(dim=0)

        return (mu, std), logits

    def reparametrize_n(self, mu, std, n=1):
        if n == 1:
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            mu_exp = mu.unsqueeze(0).expand(n, *mu.shape)
            std_exp = std.unsqueeze(0).expand(n, *std.shape)
            eps = torch.randn_like(std_exp)
            return mu_exp + eps * std_exp

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
