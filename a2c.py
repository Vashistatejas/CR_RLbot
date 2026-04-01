# =========================
# RL CORE FOR CLASH SIM
# =========================

from sim import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import deque

# -------------------------
# Constants
# -------------------------
MAX_TIME = 180.0
MAX_UNITS = 10
PASS_ID = 4               # 4 cards + PASS
NUM_HAND = 5              # 4 cards + PASS

GRID_W = GRID_W
GRID_H = GRID_H

CARD_ID = {name: i for i, name in enumerate(CARD_POOL.keys())}
NUM_CARDS = len(CARD_ID)

STATE_DIM = (
    3 +          # global
    4 +          # hand
    6 +          # towers
    2 * MAX_UNITS * 5  # units
)

# =====================================================
# 1. STATE ENCODER
# =====================================================

def encode_state(state: MatchState):
    features = []

    # ----- Global -----
    features.append((state.timestep * DT) / MAX_TIME)
    features.append(state.elixir_p1 / 10.0)
    features.append(state.elixir_p2 / 10.0)

    # ----- Hand (P1 perspective) -----
    for c in state.cards_p1.hand:
        features.append(CARD_ID[c.name] / (NUM_CARDS - 1))

    # ----- Towers -----
    # P1
    features.append(state.towers_p1[0].hp / 4000.0)
    features.append(state.towers_p1[1].hp / 2500.0)
    features.append(state.towers_p1[2].hp / 2500.0)
    # P2
    features.append(state.towers_p2[0].hp / 4000.0)
    features.append(state.towers_p2[1].hp / 2500.0)
    features.append(state.towers_p2[2].hp / 2500.0)

    # ----- Units -----
    def encode_units(units, owner_flag):
        enc = []
        for u in units[:MAX_UNITS]:
            enc.extend([
                u.x / GRID_W,
                u.y / GRID_H,
                u.hp / 5000.0,
                CARD_ID[u.name] / (NUM_CARDS - 1),
                owner_flag
            ])
        while len(enc) < MAX_UNITS * 5:
            enc.append(0.0)
        return enc

    features.extend(encode_units(state.units_p1, 1.0))
    features.extend(encode_units(state.units_p2, 0.0))

    return torch.tensor(features, dtype=torch.float32)

# =====================================================
# 2. ACTION MASKING
# =====================================================

def card_action_mask(state, owner):
    mask = torch.zeros(NUM_HAND, dtype=torch.bool)

    hand = state.cards_p1.hand if owner == 1 else state.cards_p2.hand
    elixir = state.elixir_p1 if owner == 1 else state.elixir_p2

    for i, card in enumerate(hand):
        if card.elixir_cost <= elixir:
            mask[i] = True

    mask[PASS_ID] = True
    return mask

def x_action_mask():
    return torch.ones(GRID_W, dtype=torch.bool)

def y_action_mask(owner):
    mask = torch.zeros(GRID_H, dtype=torch.bool)
    if owner == 1:
        mask[:RIVER_MIN] = True
    else:
        mask[RIVER_MAX + 1:] = True
    return mask

def get_action_masks(state, owner):
    return {
        "card": card_action_mask(state, owner),
        "x": x_action_mask(),
        "y": y_action_mask(owner)
    }

# =====================================================
# 3. MASKED SOFTMAX + SAMPLING
# =====================================================

def masked_softmax(logits, mask):
    masked_logits = logits.clone()
    masked_logits[~mask] = -1e9
    return F.softmax(masked_logits, dim=-1)

def sample_action(net, state_tensor, masks):
    pi_card, pi_x, pi_y, value = net(state_tensor)

    p_card = masked_softmax(pi_card, masks["card"])
    p_x = masked_softmax(pi_x, masks["x"])
    p_y = masked_softmax(pi_y, masks["y"])

    card = torch.distributions.Categorical(p_card).sample()
    x = torch.distributions.Categorical(p_x).sample()
    y = torch.distributions.Categorical(p_y).sample()

    log_prob = (
        torch.log(p_card[card]) +
        torch.log(p_x[x]) +
        torch.log(p_y[y])
    )

    dist_card = torch.distributions.Categorical(p_card)
    dist_x    = torch.distributions.Categorical(p_x)
    dist_y    = torch.distributions.Categorical(p_y)

    entropy = dist_card.entropy() + dist_x.entropy() + dist_y.entropy()

    return card.item(), x.item(), y.item(), log_prob, value, entropy

    

# =====================================================
# 4. A2C NEURAL NETWORK
# =====================================================

class A2CNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.pi_card = nn.Linear(128, NUM_HAND)
        self.pi_x = nn.Linear(128, GRID_W)
        self.pi_y = nn.Linear(128, GRID_H)

        self.value = nn.Linear(128, 1)

    def forward(self, state):
        h = self.shared(state)
        return (
            self.pi_card(h),
            self.pi_x(h),
            self.pi_y(h),
            self.value(h).squeeze(-1)
        )

        

def flip_state(state: MatchState):
    s = copy.deepcopy(state)

    # swap units
    s.units_p1, s.units_p2 = s.units_p2, s.units_p1

    # swap towers
    s.towers_p1, s.towers_p2 = s.towers_p2, s.towers_p1

    # swap elixir
    s.elixir_p1, s.elixir_p2 = s.elixir_p2, s.elixir_p1

    #swap the decks in hand 
    s.cards_p1, s.cards_p2 = s.cards_p2, s.cards_p1

    # mirror y coordinates
    for u in s.units_p1 + s.units_p2:
        u.y = GRID_H - 1 - u.y

    for t in s.towers_p1 + s.towers_p2:
        t.y = GRID_H - 1 - t.y

    return s

def act(net, state, owner):
    state_tensor = encode_state(state)
    masks = get_action_masks(state, owner)

    card, x, y, log_prob, value,entropy = sample_action(
        net, state_tensor, masks
    )

    return card, x, y, log_prob, value,entropy


import torch
import torch.optim as optim

GAMMA = 0.99
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4
ROLLOUT_LEN = 20
MAX_EPISODES = 1500

TMAX = 2500 * 3   # 3 towers per side

W_TOWER = 0.1
W_UNIT  = 0.2
W_ELIX  = 0.01
W_TOWER_KILL = 1.0
W_ELIX_HOARD = 0.075

def count_dead_towers(towers):
    return sum(1 for t in towers if t.hp <= 0)


def elixir_hoard_penalty(Em, threshold=7.0, max_elix=10.0):
    excess = max(0.0, Em - threshold)
    return -excess / (max_elix - threshold)

def phi(state):
    Tm = sum(t.hp for t in state.towers_p1)
    Te = sum(t.hp for t in state.towers_p2)

    dead_enemy = count_dead_towers(state.towers_p2)
    dead_mine  = count_dead_towers(state.towers_p1)

    tower_term = ((TMAX - Te) - (TMAX - Tm)) / TMAX
    kill_term  = W_TOWER_KILL * (dead_enemy - dead_mine)
    elix_term  = W_ELIX_HOARD * elixir_hoard_penalty(state.elixir_p1)
    return tower_term + kill_term + elix_term


def train_self_play():
    net = A2CNet().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR)

    GAMMA = 0.99
    LAMBDA = 0.95
    log_window = 10

    win_history = deque(maxlen=log_window)
    tower_delta_hist = deque(maxlen=log_window)
    unit_delta_hist = deque(maxlen=log_window)
    elixir_delta_hist = deque(maxlen=log_window)
    tower_kill_hist = deque(maxlen=log_window)
    phi_end_hist = deque(maxlen=log_window)

    for episode in range(MAX_EPISODES):
        state = MatchState()
        episode_reward = 0.0

        buffers = []   # (log_prob, value, reward, entropy)
        start_Tm = sum(t.hp for t in state.towers_p1)
        start_Te = sum(t.hp for t in state.towers_p2)
        start_units_m = 0
        start_units_e = 0
        start_elix_m = state.elixir_p1
        start_elix_e = state.elixir_p2
        start_dead_m = count_dead_towers(state.towers_p1)
        start_dead_e = count_dead_towers(state.towers_p2)
        while not state.game_over:

            for _ in range(ROLLOUT_LEN):
                if state.game_over:
                    break

                start_units_m += sum(u.hp for u in state.units_p1)
                start_units_e += sum(u.hp for u in state.units_p2)

                # -----------------
                # POTENTIAL BEFORE
                # -----------------
                pre_phi = phi(state)

                # -----------------
                # PLAYER 1
                # -----------------
                c1, x1, y1, lp1, v1, ent1 = act(net, state, owner=1)
                if c1 != PASS_ID:
                    play_card(state, 1, state.cards_p1.hand[c1], x1, y1)

                # -----------------
                # PLAYER 2 (SELF-PLAY)
                # -----------------
                flipped = flip_state(state)
                c2, x2, y2, lp2, v2, ent2 = act(net, flipped, owner=1)
                y2 = GRID_H - 1 - y2
                if c2 != PASS_ID:
                    play_card(state, 2, state.cards_p2.hand[c2], x2, y2)

                # -----------------
                # ENV STEP
                # -----------------
                step(state)

                # -----------------
                # POTENTIAL AFTER
                # -----------------
                post_phi = phi(state)

                r1 = post_phi - pre_phi
                r2 = -r1

                if c1 == PASS_ID:
                    r1 -= 0.01*DT
                    r2 += 0.01*DT

                if c2 == PASS_ID:
                    r2 -= 0.01*DT
                    r1 += 0.01*DT

                if state.game_over:
                    if state.towers_p2[0].hp <= 0:
                        r1 += 2.0
                        r2 -= 2.0
                    elif state.towers_p1[0].hp <= 0:
                        r1 -= 2.0
                        r2 += 2.0

                episode_reward += r1

                buffers.append((lp1, v1, r1, ent1))
                buffers.append((lp2, v2, r2, ent2))

            # -----------------
            # BOOTSTRAP VALUE
            # -----------------
            if state.game_over:
                next_value = 0.0
            else:
                with torch.no_grad():
                    s = encode_state(state)
                    _, _, _, next_value = net(s)

            # -----------------
            # EXTRACT
            # -----------------
            log_probs = torch.stack([b[0] for b in buffers])
            values = torch.stack([b[1] for b in buffers])
            rewards = [b[2] for b in buffers]
            entropies = torch.stack([b[3] for b in buffers])

            # -----------------
            # GAE
            # -----------------
            values_ext = list(values.detach().cpu().numpy()) + [next_value]
            gae = 0
            advantages = []

            for t in reversed(range(len(rewards))):
                delta = rewards[t] + GAMMA * values_ext[t+1] - values_ext[t]
                gae = delta + GAMMA * LAMBDA * gae
                advantages.insert(0, gae)

            advantages = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
            returns = advantages + values.detach()

            # -----------------
            # NORMALIZE ADV
            # -----------------

            critic_loss = VALUE_COEF * (returns - values).pow(2).mean()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # -----------------
            # LOSSES
            # -----------------
            actor_loss = -(log_probs * advantages.detach()).mean()
            
            entropy_loss = ENTROPY_COEF * entropies.mean()

            loss = actor_loss + critic_loss - entropy_loss

            # -----------------
            # UPDATE
            # -----------------
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            buffers.clear()


        end_Tm = sum(t.hp for t in state.towers_p1)
        end_Te = sum(t.hp for t in state.towers_p2)

        end_units_m = sum(u.hp for u in state.units if u.owner == 1)
        end_units_e = sum(u.hp for u in state.units if u.owner == 2)

        end_elix_m = state.elixir_p1
        end_elix_e = state.elixir_p2

        end_dead_m = count_dead_towers(state.towers_p1)
        end_dead_e = count_dead_towers(state.towers_p2)
        win = 1 if state.winner == 1 else 0
            
        win_history.append(win)
        tower_delta_hist.append(end_Tm - start_Tm)
        unit_delta_hist.append(end_units_m - start_units_m)
        elixir_delta_hist.append(end_elix_m - start_elix_m)
        tower_kill_hist.append((end_dead_e - start_dead_e) - (end_dead_m - start_dead_m))
        phi_end_hist.append(phi(state))
        
        if episode % 10 == 0:
            print(f"[EP {episode}] "
            f"WinRate={sum(win_history)/len(win_history):.2f} | "
            f"TowerΔ={sum(tower_delta_hist)/len(tower_delta_hist):.1f} | "
            f"UnitΔ={sum(unit_delta_hist)/len(unit_delta_hist):.1f} | "
            f"ElixirΔ={sum(elixir_delta_hist)/len(elixir_delta_hist):.2f} | "
            f"TowersKilled={sum(tower_kill_hist)/len(tower_kill_hist):.2f} | "
            f"Φ_end={sum(phi_end_hist)/len(phi_end_hist):.2f} | "
            f"Loss={loss.item():.3f}")



if __name__ == "__main__":  
    train_self_play()