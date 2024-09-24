import torch
from torch import nn

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class GenerativeRepresentation(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        rep_dim: int = 768,
        emb_dim: int = 3072,
        batch_size: int = 32,
        lambd: float = 0.0051
    ):
        super().__init__()
        self.encoder = encoder
        self.rep_dim = rep_dim
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.lambd = lambd
        self.projector = nn.Sequential(
            nn.Linear(self.rep_dim, self.emb_dim, bias=False),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_dim, self.emb_dim)
        )
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(self.emb_dim, affine=False)

    def forward(self, x1, x2): # x1, x2 is augmented
        
        z1 = self.encoder(x1) # representation
        z2 = self.encoder(x2)

        y1 = self.projector(z1) # embedding
        y2 = self.projector(z2) 

        # empirical cross-correlation matrix
        c = self.bn(y1).T @ self.bn(y2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
    
    def get_representation(self, x):
        """Get the representation from the encoder."""
        with torch.no_grad():  # Disable gradient calculation
            representation = self.encoder(x)
        return representation