import math

import torch


def _to_tensor(value, ref_tensor):
    if isinstance(value, torch.Tensor):
        return value
    return torch.full((1,), float(value), dtype=ref_tensor.dtype, device=ref_tensor.device)


class gaussian:
    def __init__(self, data=None, center=torch.zeros(1), scale=torch.ones(1)):
        if data is None:
            center_t = center if isinstance(center, torch.Tensor) else torch.tensor([float(center)])
            scale_t = _to_tensor(scale, center_t)
            self.mu = center_t
            self.sigma = scale_t
        else:
            self.mu = torch.mean(data)
            self.sigma = torch.sqrt((data - self.mu) ** 2 / (data.size(0) - 1))

    def cdf(self, x):
        return 0.5 * (1 + torch.erf((x - self.mu) / self.sigma / math.sqrt(2)))

    def ppf(self, q):
        q_t = _to_tensor(q, self.mu)
        return self.mu + self.sigma * math.sqrt(2) * torch.erfinv(2 * q_t - 1)

    def logL(self, x):
        return -1 * (torch.log(self.sigma * math.sqrt(2 * math.pi)) + 0.5 * (((x - self.mu) ** 2) / self.sigma))


class laplace:
    def __init__(self, data=None, center=torch.zeros(1), scale=torch.ones(1)):
        if data is None:
            center_t = center if isinstance(center, torch.Tensor) else torch.tensor([float(center)])
            scale_t = _to_tensor(scale, center_t)
            self.mu = center_t
            self.b = scale_t
        else:
            self.mu = torch.mean(data)
            self.b = torch.mean(torch.abs(data - self.mu))

    def cdf(self, x):
        return torch.where(
            x <= self.mu,
            0.5 * torch.exp((x - self.mu) / self.b),
            1 - 0.5 * torch.exp(-(x - self.mu) / self.b),
        )

    def ppf(self, q):
        q_t = _to_tensor(q, self.mu)
        return torch.where(
            q_t <= 0.5,
            self.mu + self.b * torch.log(2 * q_t),
            self.mu - self.b * torch.log(2 - 2 * q_t),
        )

    def logL(self, x):
        return -1 * (torch.log(2 * self.b) + torch.abs(x - self.mu) / self.b)


class gumbel:
    def __init__(self, data):
        mean = torch.mean(data)
        std = torch.sum((data - mean) ** 2) / (data.size(0) - 1)
        self.beta = std * math.sqrt(6) / math.pi
        self.mu = mean - 0.57721 * self.beta

    def cdf(self, x):
        return torch.exp(-torch.exp(-(x - self.mu) / self.beta))

    def ppf(self, q):
        q_t = _to_tensor(q, self.mu)
        return self.mu - self.beta * torch.log(-torch.log(q_t))


def return_distribution_dict():
    return {
        'gaussian': gaussian,
        'laplace': laplace,
        'gumbel': gumbel,
    }

