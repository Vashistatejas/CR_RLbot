# CR_RLbot: Clash Royale Reinforcement Learning Bot

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Bot-red.svg)
![Roboflow](https://img.shields.io/badge/Roboflow-Vision-purple.svg)

CR_RLbot is an end-to-end framework for training and running a Reinforcement Learning agent to play **Clash Royale**. It features a custom-built lightweight game simulator, an Advantage Actor-Critic (A2C) training pipeline, and a real-time computer vision module that extracts game states straight from live gameplay screen captures.

---

## 🌟 Key Features

* **Custom Python Simulator (`sim.py`)**: A fast, grid-based Python simulator that accurately mimics Clash Royale rules. It models lane mechanics, bridge crossing, troop movement, tower targeting, and splash damage. 
* **A2C RL Core (`a2c.py`)**: A fully implemented Actor-Critic neural network in PyTorch. The agent learns strategies via self-play, directly interpreting global elixir, tower healths, cards in hand, and exact unit positions.
* **Computer Vision Extraction (`eyes.py` & `test.py`)**: Uses `mss` for ultra-fast screen capturing and a custom **Roboflow** inference model to translate live screen pixels into structured JSON containing exact game state (units, positions, health, elixir).
* **Bot Inference (`bot.py`)**: Ready-to-use script that loads trained `.pth` PyTorch weights to evaluate states and make live card placement decisions.

## 📂 Project Structure

| File | Description |
|------|-------------|
| `sim.py` | The Clash Royale simulated emulation logic and mechanics. |
| `a2c.py` | PyTorch Actor-Critic neural network architecture & self-play training loop. |
| `bot.py` | Inference script to predict optimal moves given a loaded model state. |
| `eyes.py` / `test.py` | Screenshot capturing and Roboflow API configuration. |
| `vision_prompt.txt` | Vision prompting specifications for state extraction. |
| `*.pth` | Pre-trained PyTorch model weights (e.g., `clash_a2c_update1.pth`). |

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. The project heavily relies on PyTorch and some computer vision libraries.

```bash
pip install torch torchvision mss numpy pillow python-dotenv inference-sdk
```

### Configuration

If you want to use the vision module in `test.py`, create a `.env` file in the root directory and add your Roboflow Inference API key:

```env
ROBOFLOW_API_KEY=your_api_key_here
```

## 🧠 Usage

### 1. Training the Agent
You can train the agent via simulated self-play to improve its heuristics.

```bash
python a2c.py
```
This runs the `train_self_play()` loop and continuously optimizes the model based on reward functions (tower kills, elixir hoarding, unit advantage).

### 2. Live State Perception
To test out the screen-capture and vision extraction on a live game:
(Note: Make sure your Clash Royale window matches the configured `CLASH_BOX` coordinates).

```bash
python eyes.py
# or if using the serverless remote API
python test.py
```

### 3. Inference / Playing
For letting the bot evaluate a single action based on a state vector:

```bash
python bot.py
```

---

## 🤝 Contributing

This project is built for experimental machine learning applied to complex tower defense games. Feel free to open issues or contribute PRs (especially regarding expanding `sim.py` troops, or fine-tuning the reward functions in `a2c.py`!).

## 📄 License
This original code is open source. (Note: Clash Royale is a trademark of Supercell).
