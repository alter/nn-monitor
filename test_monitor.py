import torch
import torch.nn as nn
from typing import Dict, Any

from nn_monitor.core import TrainingMonitor, ActivationMonitor
from nn_monitor.sanity import check_receptive_field_gradients
from nn_monitor.metrics import compute_psi, compute_attention_entropy

# Test basic metrics 
psi = compute_psi(torch.randn(100).numpy(), torch.randn(100).numpy())
print("PSI OK:", psi)

att = compute_attention_entropy(torch.softmax(torch.randn(2, 4, 10, 10), dim=-1))
print("Att Entropy OK:", att)

# Test sanity
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 10, kernel_size=3, padding=1)
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=-1)
        return self.fc(x)

device = "cpu"
model = DummyModel().to(device)
batch = (torch.randn(4, 3, 20), torch.empty(4, dtype=torch.long).random_(2))
loader = [batch]

res = check_receptive_field_gradients(model, loader, nn.CrossEntropyLoss(), device)
print("Sanity OK:", res)

# Test core
monitor = TrainingMonitor('./output', detect_anomalies=True)
act_mon = ActivationMonitor(model)
out = model(batch[0])
act_mon.summary()
res_epoch = monitor.log_epoch(0, model, torch.softmax(out, dim=1).detach().numpy(), batch[1].numpy(), 0.5, 0.4, 80.0, 85.0, 0.01)
monitor.save_summary()
print("Core OK.")

