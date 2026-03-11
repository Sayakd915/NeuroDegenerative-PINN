import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_size):
        super(BaselineLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        predictions = self.fc(out)
        return predictions


class AtrophyPINN(nn.Module):
    """PINN model for atrophy prediction"""
    def __init__(self, input_size, hidden_size=64):
        super(AtrophyPINN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.k = nn.Parameter(torch.tensor([0.01]))

    def forward(self, x, t):
        inputs = torch.cat  ([x, t], dim=1)
        return self.net(inputs)

    def compute_physics_loss(self, x, t):
        t.requires_grad_(True)
        V = self.forward(x, t)

        dV_dt = torch.autograd.grad(
            outputs=V,
            inputs=t,
            grad_outputs=torch.ones_like(V),
            create_graph=True
        )[0]

        ode_residual = dV_dt + self.k * V
        return torch.mean(ode_residual**2)


class KineticPINN(nn.Module):
    """PINN model for kinetic modeling"""
    def __init__(self, input_size, output_size=3, hidden_size=64):
        super(KineticPINN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

        self.k = nn.Parameter(torch.ones(output_size) * 0.01)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

    def compute_physics_loss(self, x, t):
        t.requires_grad_(True)
        P = self.forward(x,t)

        physics_loss = 0
        for i in range(P.size(1)):
            P_i = P[:, i].unsqueeze(1)
            dP_dt = torch.autograd.grad(
                outputs=P_i,
                inputs=t,
                grad_outputs=torch.ones_like(P_i),
                create_graph=True
            )[0]

            ode_residual = dP_dt - self.k[i] * P_i
            physics_loss += torch.mean(ode_residual**2)
        
        return physics_loss/P.size(1)


class CoupledCJDPINN(nn.Module):
    """Used for the Synthetic CJD Dataset
    Governing Equations:
    dM/dt = lambda - gamma * M - k * M * P
    dP/dt = k * M * P - mu * P
    """

    def __init__(self, input_size=2, hidden_size=64):
        super(CoupledCJDPINN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size + 1, hidden_size), # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2) # Output 0 : PrPc, Output 1 : PrPSc
        )

        self.k_net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),
            nn.Linear(16,1),
            nn.Softplus()
        )

        self.lam = 1.0
        self.gamma = 0.1
        self.mu = 0.05

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        predictions = self.net(inputs)
        return nn.functional.softplus(predictions)

    def get_patient_k(self, x):
        return self.k_net(x)

    def compute_physics_loss(self, x, t):
        t.requires_grad_(True)
        predictions = self.forward(x,t)

        M = predictions[:, 0].unsqueeze(1)
        P = predictions[:, 1].unsqueeze(1)

        k = self.get_patient_k(x)

        dM_dt = torch.autograd.grad(
            outputs=M,
            inputs=t,
            grad_outputs=torch.ones_like(M),
            create_graph=True
        )[0]

        dP_dt = torch.autograd.grad(
            outputs=P,
            inputs=t,
            grad_outputs=torch.ones_like(P),
            create_graph=True
        )[0]

        ode_M = dM_dt - (self.lam - (self.gamma * M) - (k * M * P))
        ode_P = dP_dt - ((k * M * P) - (self.mu * P))

        physics_loss = torch.mean(ode_M**2) + torch.mean(ode_P**2)
        return physics_loss