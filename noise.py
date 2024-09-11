import torch
import torch.nn.functional as F


class ValueNoise(torch.nn.Module):
    def __init__(self, n_dims, n_fields, res, seed=0, periodic=False):
        super().__init__()
        self.n_dims = n_dims
        self.n_fields = n_fields
        self.res = res
        self.seed = seed
        self.periodic = periodic
        self.values = self.init_values()
        self.corners = self.init_corners()

    def init_corners(self):
        zero_one = torch.tensor([0,1],dtype=torch.long)
        return torch.meshgrid((zero_one,)*self.n_dims, indexing='ij')

    def init_values(self):
        torch.manual_seed(self.seed)
        shape = (*([self.res+1,]*self.n_dims), self.n_fields)
        g = torch.normal(torch.zeros(*shape), torch.ones(*shape))
        return g

    def forward(self, x):
        # x is shape (Npts, dims)
        x = torch.fmod(x*self.res, self.res) # Wrap points

        locs = torch.frac(x)              # Local coordinates of x
        idxA = torch.floor(x).long()      # indices of vertices below x
        idxB = idxA + 1                   # indices of vertices above x
        if self.periodic:
            idxB[idxB == self.res] = 0    # -> wrap for periodic boundary
        idx = torch.stack((idxA, idxB), dim=-1)

        idx_verts = []
        for i in range(self.n_dims):
            idx_verts.append(idx[:,i,self.corners[i]])
        idx_verts = torch.stack(idx_verts, dim=-1)
        vals = self.values[*[idx_verts[...,i] for i in range(self.n_dims)], :]

        for i in range(self.n_dims):
            weights = locs[:,i, *(None,)*(self.n_dims - i)]
            vals = torch.lerp(vals[:,0,...], vals[:,1,...], weights)

        return vals










class PerlinNoise(torch.nn.Module):
    def __init__(self, n_dims, n_fields, res, seed=0):
        super().__init__()
        self.n_dims = n_dims
        self.n_fields = n_fields
        self.res = res
        self.seed = seed
        self.gradients = self.init_gradients()
        self.verts = self.init_verts()

    def init_gradients(self):
        torch.manual_seed(self.seed)
        shape = (self.res, self.res, self.n_dims, self.n_fields)
        g = torch.normal(torch.zeros(*shape), torch.ones(*shape))
        return F.normalize(g, dim=2)
    
    def forward(self, x):
        return NotImplementedError