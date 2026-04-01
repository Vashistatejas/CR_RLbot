import torch
from rl import A2CNet, act, encode_state
from sim import MatchState

DEVICE = torch.device("cpu")

net = A2CNet().to(DEVICE)

checkpoint = torch.load(
    "clash_a2c.pth",
    map_location=DEVICE
)

# If you saved full checkpoint dict
if "model_state_dict" in checkpoint:
    net.load_state_dict(checkpoint["model_state_dict"])
else:
    net.load_state_dict(checkpoint)

net.eval()
print("✅ Model loaded")



with torch.no_grad():
    card, x, y, _, _ = act(net, state, owner=1)
