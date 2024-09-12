import torch
from torch import nn
import torch.nn.functional as F


class ValueNoise(nn.Module):
    def __init__(self, n_dims, n_fields, res, seed=None, periodic=False, smoothness=1, trainable=False):
        super().__init__()
        self.smoothness = int(smoothness)
        assert self.smoothness > 0 and self.smoothness < 4, "Smoothness must be 1, 2, or 3."
        self.wfun = self.init_wfun()
        self.n_dims = n_dims
        self.n_fields = n_fields
        self.res = res
        if seed is not None:
            self.seed = seed
        else:
            self.seed = torch.seed()
        self.periodic = periodic
        self.trainable = trainable
        if trainable:
            self.values = nn.Parameter(self.init_values())
        else:
            self.values = self.init_values()
        self.corners = self.init_corners()
        self.name = "Value Noise"

    def init_wfun(self):
        if self.smoothness == 1:
            # Linear interpolation AKA linstep
            return lambda w: w
        elif self.smoothness == 2:
            # Cubic Hermite interpolation AKA smoothstep
            return lambda w: (3. - 2.*w)*w*w
        elif self.smoothness == 3:
            # Smootherstep
            return lambda w: (10. + w*(6.*w - 15.))*w*w*w

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
            vals = torch.lerp(vals[:,0,...], vals[:,1,...], self.wfun(weights))

        return vals


class PerlinNoise(nn.Module):
    def __init__(self, n_dims, n_fields, res, seed=None, periodic=False, smoothness=1, trainable=False):
        super().__init__()
        self.smoothness = int(smoothness)
        assert self.smoothness > 0 and self.smoothness < 4, "Smoothness must be 1, 2, or 3."
        self.wfun = self.init_wfun()
        self.n_dims = n_dims
        self.n_fields = n_fields
        self.res = res
        if seed is not None:
            self.seed = seed
        else:
            self.seed = torch.seed()
        self.periodic = periodic
        self.trainable = trainable
        if trainable:
            self.grads = nn.Parameter(self.init_grads())
        else:
            self.grads = self.init_grads()
        self.corners = self.init_corners()
        self.name = "Perlin Noise"

    def init_wfun(self):
        if self.smoothness == 1:
            # Linear interpolation AKA linstep
            return lambda w: w
        elif self.smoothness == 2:
            # Cubic Hermite interpolation AKA smoothstep
            return lambda w: (3. - 2.*w)*w*w
        elif self.smoothness == 3:
            # Smootherstep
            return lambda w: (10. + w*(6.*w - 15.))*w*w*w

    def init_corners(self):
        zero_one = torch.tensor([0,1],dtype=torch.long)
        return torch.meshgrid((zero_one,)*self.n_dims, indexing='ij')

    def init_grads(self):
        torch.manual_seed(self.seed)
        shape = (*([self.res+1,]*self.n_dims), self.n_fields, self.n_dims)
        g = torch.normal(torch.zeros(*shape), torch.ones(*shape))
        return F.normalize(g, dim=-1)

    def forward(self, x):
        # x is shape (Npts, dims)
        x = torch.fmod(x*self.res, self.res) # Wrap points

        locs = torch.frac(x)              # Local coordinates of x
        idxA = torch.floor(x).long()      # indices of vertices below x
        idxB = idxA + 1                   # indices of vertices above x
        if self.periodic:
            idxB[idxB == self.res] = 0    # -> wrap for periodic boundary
        idx = torch.stack((idxA, idxB), dim=-1)
        offsets = torch.stack((locs, 1.0-locs), dim=-1)

        idx_verts = []
        offsets_verts = []
        for i in range(self.n_dims):
            idx_verts.append(idx[:,i,self.corners[i]])
            offsets_verts.append(offsets[:,i,self.corners[i]])
        idx_verts = torch.stack(idx_verts, dim=-1)
        offsets_verts = torch.stack(offsets_verts, dim=-1).unsqueeze(-2)
        grads = self.grads[*[idx_verts[...,i] for i in range(self.n_dims)], ...]

        vals = torch.sum(grads * offsets_verts, dim=-1)
        for i in range(self.n_dims):
            weights = locs[:,i, *(None,)*(self.n_dims - i)]
            vals = torch.lerp(vals[:,0,...], vals[:,1,...], self.wfun(weights))
        return vals
