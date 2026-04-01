"""
Clash-Royale-like simulator core
- Grid: width=18, height=32 (x in [0..17], y in [0..31])
- River: horizontal band that blocks crossing except at bridge centers
- Troops spawn in playable area (owner=1 bottom, owner=2 top)
- Troop movement: lane-following, bridge crossing, default advance when no enemies
- Combat: vision-based targeting, attack cooldown (hit speed), towers also have hit speed
- Time: DT seconds per tick, MAX_STEPS total ticks
"""

import math
import random


# -------------------------
# Configuration / constants
# -------------------------
GRID_W = 18
GRID_H = 32

# River horizontal band (inclusive indexes)
RIVER_MIN = 15
RIVER_MAX = 16

# Bridge centers (x coordinates) where crossing is allowed
BRIDGE_CENTERS = [5.0, 12.0]
BRIDGE_WIDTH = 1.5  # how close to center is considered "on the bridge"

# Lane centers (x positions troops prefer)
LANE_CENTERS = [5.0, 12.0]  # left and right lane centers

# Timing
DT = 0.25  # seconds per tick
SIM_SECONDS = 180.0  # total simulated seconds per match
MAX_STEPS = int(SIM_SECONDS / DT)

# Spawn probabilities for random spawning (optional)
RANDOM_SPAWN_PROB = 0.5

# Movement thresholds
CROSSING_THRESHOLD = 0.05  # small distance threshold to consider "reached" a point



# -------------------------
# Helper classes & helpers
# -------------------------
class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


from collections import deque

class Card:
    def __init__(self, name, elixir_cost, unit_kwargs,count):
        self.name = name
        self.elixir_cost = elixir_cost
        self.unit_kwargs = unit_kwargs  # passed to place_unit
        self.count = count


class Tower:
    def __init__(self, name, x, y, hp, damage, attack_range, hit_speed=1.0):
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.hp = float(hp)
        self.damage = float(damage)
        self.attack_range = float(attack_range)
        self.hit_speed = float(hit_speed)
        self._cooldown = 0.0  # time until next attack allowed

    @property
    def alive(self):
        return self.hp > 0

    def tick_cooldown(self):
        self._cooldown = max(0.0, self._cooldown - DT)

    def can_attack(self):
        return self._cooldown <= 1e-9

    def register_attack(self):
        self._cooldown = self.hit_speed


class Unit:
    def __init__(self, name, x, y, hp, damage, speed, attack_range, vision_range, owner, hit_speed=1.0,target_type="any", splash_radius=0.0,
        splash_targets="units"):
        self.name = str(name)
        self.x = float(x)
        self.y = float(y)
        self.hp = float(hp)
        self.damage = float(damage)
        self.speed = float(speed)  # units per tick distance (in world units)
        self.attack_range = float(attack_range)
        self.vision_range = float(vision_range)
        self.owner = int(owner)  # 1 (bottom) or 2 (top)
        self.hit_speed = float(hit_speed)
        self._cooldown = 0.0
        self.target_type = target_type
        self.splash_radius = float(splash_radius)
        self.splash_targets = splash_targets

        # pathing state
        self.target = None            # actual combat target (Unit or Tower)
        self.crossing_phase = None    # None / "to_entry" / "to_exit"
        self.crossing_point = None    # Point used for crossing phase

    @property
    def alive(self):
        return self.hp > 0

    def tick_cooldown(self):
        self._cooldown = max(0.0, self._cooldown - DT)

    def can_attack(self):
        return self._cooldown <= 1e-9

    def register_attack(self):
        self._cooldown = self.hit_speed


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def clamp_step(unit, dx, dy):
    """Return (step_x, step_y) normalized and clamped by speed and remaining distance."""
    dist_rem = math.hypot(dx, dy)
    if dist_rem == 0:
        return 0.0, 0.0
    step = min(unit.speed * DT, dist_rem)  # speed scaled per tick
    return (dx / dist_rem) * step, (dy / dist_rem) * step


def nearest_value(x, arr):
    return min(arr, key=lambda v: abs(v - x))


def is_bottom_side(y):
    return y <= (RIVER_MIN - 1)


def is_top_side(y):
    return y >= (RIVER_MAX + 1)


def needs_to_cross(unit_y, target_y):
    """True if unit and target are on opposite sides of the river (crossing required)."""
    return (is_bottom_side(unit_y) and is_top_side(target_y)) or (is_top_side(unit_y) and is_bottom_side(target_y))


def count_alive_towers(towers):
    return sum(1 for t in towers if t.alive)

# -------------------------
# CARD POOL
# -------------------------

CARD_POOL = {
    "knight": Card(
        "knight", 3,
        dict(hp=1200, damage=120, speed=1.0, attack_range=1.2, vision_range=6.0, hit_speed=1.5,target_type="any"),
        count =1
    ),
    "giant": Card(
        "giant", 5,
        dict(hp=3000, damage=150, speed=1.0, attack_range=1.0, vision_range=6.0, hit_speed=1.0,target_type="building"),
        count =1
    ),
    "mini_pekka": Card(
        "mini_pekka", 4,
        dict(hp=900, damage=450, speed=1.0, attack_range=1.0, vision_range=8.0, hit_speed=1.0,target_type="any"),
        count =1
    ),
    "archer": Card(
        "archer", 3,
        dict(hp=304, damage=112, speed=1.2, attack_range=5.0, vision_range=7.0, hit_speed=1.2,target_type="any"),
        count =2
    ),
    "valkyrie": Card(
        "valkyrie", 4,
        dict(hp=1100, damage=150, speed=1.0, attack_range=1.0, vision_range=6.0, hit_speed=1.0,target_type="any",splash_radius=1.5, splash_targets="units"),
        count =1
    ),
    "goblin": Card(
        "goblin", 2,
        dict(hp=202, damage=120, speed=1.6, attack_range=1.0, vision_range=6.0, hit_speed=1.5,target_type="any"),
        count=4
    ),
    "musketeer": Card(
        "musketeer", 4,
        dict(hp=450, damage=130, speed=1.0, attack_range=6.0, vision_range=6.0, hit_speed=1.0,target_type="any"),
        count =1
    ),
    "spear goblins": Card(
        "spear goblins", 2,
        dict(hp=120, damage=90, speed=1.0, attack_range=5.0, vision_range=6.0, hit_speed=1.2,target_type="any"),
        count=3
    ),
}

class PlayerCards:
    def __init__(self, deck_cards):
        assert len(deck_cards) == 8
        self.deck = deck_cards[:]
        random.shuffle(self.deck)

        self.hand = self.deck[:4]
        self.queue = deque(self.deck[4:])

    def play(self, hand_index):
        card = self.hand.pop(hand_index)
        self.hand.append(self.queue.popleft())
        self.queue.append(card)
        return card


def formation_offsets(n, spacing=0.6):
    offsets = []
    if n == 1:
        return [(0.0, 0.0)]

    angle_step = 2 * math.pi / n
    for i in range(n):
        a = i * angle_step
        offsets.append((spacing * math.cos(a), spacing * math.sin(a)))
    return offsets
    
# -------------------------
# Movement & pathing
# -------------------------
def move_towards_point(unit, point):
    dx = point.x - unit.x
    dy = point.y - unit.y
    sx, sy = clamp_step(unit, dx, dy)
    unit.x += sx
    unit.y += sy


def start_crossing_phase(unit, bridge_x):
    """
    Initialize crossing by setting entry and exit points and setting crossing_phase = 'to_entry'.
    Entry is at the near edge of the river for the unit's side.
    Exit is on the opposite side just past the river.
    """
    if is_bottom_side(unit.y):
        entry = Point(bridge_x, float(RIVER_MIN))        # approach from bottom edge
        exit = Point(bridge_x, float(RIVER_MAX + 0.5))   # step slightly into top side
    else:
        entry = Point(bridge_x, float(RIVER_MAX))        # approach from top edge
        exit = Point(bridge_x, float(RIVER_MIN - 0.5))   # step slightly into bottom side

    unit.crossing_phase = "to_entry"
    unit.crossing_point = entry
    unit._crossing_exit = exit  # temp storage on unit


def update_crossing(unit):
    """Drive crossing phases: move to entry, then to exit, then clear crossing state."""
    if unit.crossing_phase is None:
        return

    # Move toward the current crossing point (entry or exit)
    move_towards_point(unit, unit.crossing_point)

    # If we reached the crossing point, advance the phase
    if dist(unit, unit.crossing_point) <= CROSSING_THRESHOLD:
        if unit.crossing_phase == "to_entry":
            # switch to exit
            unit.crossing_phase = "to_exit"
            unit.crossing_point = unit._crossing_exit
        else:
            # finished crossing
            unit.crossing_phase = None
            unit.crossing_point = None
            unit._crossing_exit = None


def move_towards(unit, goal):
    """
    Move the unit toward a goal (Unit, Tower or Point).
    If the goal is on the opposite side of the river, set up a bridge crossing (entry->exit).
    Otherwise simply move toward the goal.
    """

    # If currently in crossing, continue it
    if unit.crossing_phase is not None:
        update_crossing(unit)
        return

    # If crossing is required to reach goal, schedule crossing via nearest bridge
    if needs_to_cross(unit.y, goal.y):
        bridge_x = nearest_value(unit.x, BRIDGE_CENTERS)
        # Start crossing (will move to entry, then exit)
        start_crossing_phase(unit, bridge_x)
        # Immediately perform one tick toward entry (inside start_crossing_phase we set crossing_point)
        update_crossing(unit)
        return

    # Otherwise move directly toward the goal (clamped)
    move_towards_point(unit, goal)


# -------------------------
# Combat & targeting
# -------------------------
def acquire_target(unit, enemy_units, enemy_towers):
    # GIANT / building-only logic
    if unit.target_type == "building":
        buildings = [t for t in enemy_towers if t.alive]
        if not buildings:
            return None
        buildings.sort(key=lambda t: dist(unit, t))
        return buildings[0]
    
    # 1) visible enemy units (within vision)
    visible = []
    for u in enemy_units:
        if not u.alive:
            continue
        d = dist(unit, u)
        if d <= unit.vision_range:
            visible.append((d, u))
    if visible:
        visible.sort(key=lambda t: t[0])
        return visible[0][1]

    # 2) if no units seen, target nearest alive tower
    alive_towers = [t for t in enemy_towers if t.alive]
    if not alive_towers:
        return None
    alive_towers.sort(key=lambda t: dist(unit, t))
    return alive_towers[0]


def attack(attacker, primary_target, enemy_units, enemy_towers):
    # No splash → normal hit
    if attacker.splash_radius <= 0:
        primary_target.hp -= attacker.damage
        return

    # Splash damage
    if attacker.splash_targets in ("units", "all"):
        for u in enemy_units:
            if not u.alive:
                continue
            if dist(attacker, u) <= attacker.splash_radius:
                u.hp -= attacker.damage

    if attacker.splash_targets == "all":
        for t in enemy_towers:
            if not t.alive:
                continue
            if dist(attacker, t) <= attacker.splash_radius:
                t.hp -= attacker.damage

def tower_attack(tower, target):
    target.hp -= tower.damage

def unit_default_advance(unit):
    """
    Provide a default navigation goal (Point) for a unit when it has no combat target.
    The unit moves toward the nearest lane center and advances toward the far edge of the map.
    """
    lane_x = nearest_value(unit.x, LANE_CENTERS)
    if unit.owner == 1:
        return Point(lane_x, float(GRID_H - 1))
    else:
        return Point(lane_x, 0.0)


def unit_step(unit, enemy_units, enemy_towers):
    """Single tick update for a unit: cooldown, target acquisition, attack or movement."""
    if not unit.alive:
        return

    # cooldown tick
    unit.tick_cooldown()

    # Acquire combat target
    unit.target = acquire_target(unit, enemy_units, enemy_towers)

    # ---------- ATTACK CHECK FIRST ----------
    if unit.target is not None:
        d = dist(unit, unit.target)
        if d <= unit.attack_range and unit.can_attack():
            attack(unit, unit.target, enemy_units, enemy_towers)
            unit.register_attack()
            return  # attacking consumes the tick

    # ---------- MOVEMENT AFTER ----------
    goal = unit.target if unit.target is not None else unit_default_advance(unit)
    move_towards(unit, goal)


# -------------------------
# Placement & spawning
# -------------------------
def in_playable_area(owner, x, y):
    """True if (x,y) is within grid, not in river, and on owner's half."""
    if not (0 <= x < GRID_W and 0 <= y < GRID_H):
        return False
    if RIVER_MIN <= y <= RIVER_MAX:
        return False
    if owner == 1:
        return is_bottom_side(y)
    elif owner == 2:
        return is_top_side(y)
    return False


def place_unit(state, owner, x, y, **kwargs):
    if not in_playable_area(owner, x, y):
        return None

    unit = Unit(
        name=kwargs["name"],
        x=x,
        y=y,
        hp=kwargs["hp"],
        damage=kwargs["damage"],
        speed=kwargs["speed"],
        attack_range=kwargs["attack_range"],
        vision_range=kwargs["vision_range"],
        owner=owner,
        hit_speed=kwargs["hit_speed"],
        target_type=kwargs.get("target_type", "any")
    )

    if owner == 1:
        state.units_p1.append(unit)
    else:
        state.units_p2.append(unit)

    return unit


def play_card(state, owner, card, x, y):
    # 1. legality
    if not in_playable_area(owner, x, y):
        return False

    # 2. elixir
    if owner == 1:
        if state.elixir_p1 < card.elixir_cost:
            return False
        state.elixir_p1 -= card.elixir_cost
    else:
        if state.elixir_p2 < card.elixir_cost:
            return False
        state.elixir_p2 -= card.elixir_cost

    # 3. spawn units
    for dx, dy in formation_offsets(card.count):
        place_unit(
            state,
            owner,
            x + dx,
            y + dy,
            name=card.name,
            **card.unit_kwargs
        )

    return True




def random_play_card(state, owner):
    cards = state.cards_p1 if owner == 1 else state.cards_p2
    elixir = state.elixir_p1 if owner == 1 else state.elixir_p2

    playable = [
        i for i, c in enumerate(cards.hand)
        if c.elixir_cost <= elixir
    ]

    if not playable:
        return

    if random.random() > 0.5:  # 50% chance to play
        return

    hand_index = random.choice(playable)
    card = cards.play(hand_index)

    x = random.randint(0, GRID_W - 1)
    y = random.randint(0, RIVER_MIN - 1) if owner == 1 else random.randint(RIVER_MAX + 1, GRID_H - 1)

    play_card(state, owner, card, x, y)


# -------------------------
# Match state & step loop
# -------------------------
class MatchState:
    def __init__(self):
        self.timestep = 0
        self.winner = 0
        self.units_p1 = []
        self.units_p2 = []
        self.elixir_p1 = 5
        self.elixir_p2 = 5
        self._elixir_timer_p1 = 0.0
        self._elixir_timer_p2 = 0.0
        self.towers_p1 = [
            Tower("king", 9, 2, 4000, 100, 7, hit_speed=1.0),
            Tower("princess_left", 4, 6, 2500, 90, 7, hit_speed=1.0),
            Tower("princess_right", 13, 6, 2500, 90, 7, hit_speed=1.0),
        ]
        self.towers_p2 = [
            Tower("king", 9, 29, 4000, 100, 7, hit_speed=1.0),
            Tower("princess_left", 4, 25, 2500, 90, 7, hit_speed=1.0),
            Tower("princess_right", 13, 25, 2500, 90, 7, hit_speed=1.0),
        ]
        self.game_over = False

        deck1 = [
            CARD_POOL["knight"], CARD_POOL["giant"], CARD_POOL["mini_pekka"],
            CARD_POOL["archer"], CARD_POOL["valkyrie"], CARD_POOL["goblin"],
            CARD_POOL["musketeer"], CARD_POOL["spear goblins"]
        ]
        deck2 = deck1[:]  # symmetric for now

        self.cards_p1 = PlayerCards(deck1)
        self.cards_p2 = PlayerCards(deck2)

        
def update_elixir(state):
    state._elixir_timer_p1 += DT
    state._elixir_timer_p2 += DT

    if state._elixir_timer_p1 >= 2.8:
        state.elixir_p1 = min(10, state.elixir_p1 + 1)
        state._elixir_timer_p1 -= 2.8

    if state._elixir_timer_p2 >= 2.8:
        state.elixir_p2 = min(10, state.elixir_p2 + 1)
        state._elixir_timer_p2 -= 2.8

def step(state: MatchState):
    state.timestep += 1
    update_elixir(state)

    # Random placement (optional)
    #random_play_card(state, 1)
    #random_play_card(state, 2)

    # Units update (copy lists to allow safe removal during iteration)
    for u in list(state.units_p1):
        unit_step(u, state.units_p2, state.towers_p2)

    for u in list(state.units_p2):
        unit_step(u, state.units_p1, state.towers_p1)

    # Tower attacks (with cooldown)
    for tower in state.towers_p1:
        if not tower.alive:
            continue
        tower.tick_cooldown()
        if not tower.can_attack():
            continue
        # find nearest enemy unit in range
        target = None
        best_d = float("inf")
        for u in state.units_p2:
            if not u.alive:
                continue
            d = dist(tower, u)
            if d <= tower.attack_range and d < best_d:
                best_d = d
                target = u
        if target is not None:
            tower_attack(tower, target)
            tower.register_attack()

    for tower in state.towers_p2:
        if not tower.alive:
            continue
        tower.tick_cooldown()
        if not tower.can_attack():
            continue
        target = None
        best_d = float("inf")
        for u in state.units_p1:
            if not u.alive:
                continue
            d = dist(tower, u)
            if d <= tower.attack_range and d < best_d:
                best_d = d
                target = u
        if target is not None:
            tower_attack(tower, target)
            tower.register_attack()

    # Cleanup dead units
    state.units_p1 = [u for u in state.units_p1 if u.alive]
    state.units_p2 = [u for u in state.units_p2 if u.alive]

    # Terminal conditions
    # King tower destroyed → immediate game over
    if not state.towers_p1[0].alive:
        state.game_over = True
        state.winner = 2
        return

    if not state.towers_p2[0].alive:
        state.game_over = True
        state.winner = 1
        return

    # Time limit reached
    if state.timestep >= MAX_STEPS:
        p1_towers = count_alive_towers(state.towers_p1)
        p2_towers = count_alive_towers(state.towers_p2)

        state.game_over = True

    if p1_towers > p2_towers:
        state.winner = 1
    elif p2_towers > p1_towers:
        state.winner = 2
    else:
        state.winner = 0  # draw





'''class ExcelLogger:
    def __init__(self, filename, max_units=30):
        self.wb = Workbook()
        self.ws = self.wb.active
        self.ws.title = "MatchLog"

        self.max_units = max_units
        self._write_header()
        self.filename = filename
        self.row = 2  # data starts from row 2

    def _write_header(self):
        headers = ["time_sec"]

        # Player 1 units
        for i in range(self.max_units):
            headers += [
                f"P1_u{i}_x", f"P1_u{i}_y", f"P1_u{i}_hp"
            ]

        # Player 2 units
        for i in range(self.max_units):
            headers += [
                f"P2_u{i}_x", f"P2_u{i}_y", f"P2_u{i}_hp"
            ]

        # Towers
        headers += [
            "P1_king_hp", "P1_princess_L_hp", "P1_princess_R_hp",
            "P2_king_hp", "P2_princess_L_hp", "P2_princess_R_hp"
        ]

        self.ws.append(headers)

    def log(self, state):
        row = [state.timestep * DT]

        # Player 1 units
        for i in range(self.max_units):
            if i < len(state.units_p1):
                u = state.units_p1[i]
                row += [u.x, u.y, u.hp]
            else:
                row += ["", "", ""]

        # Player 2 units
        for i in range(self.max_units):
            if i < len(state.units_p2):
                u = state.units_p2[i]
                row += [u.x, u.y, u.hp]
            else:
                row += ["", "", ""]

        # Towers
        t1 = state.towers_p1
        t2 = state.towers_p2

        row += [
            t1[0].hp, t1[1].hp, t1[2].hp,
            t2[0].hp, t2[1].hp, t2[2].hp
        ]

        self.ws.append(row)
        self.row += 1

    def save(self):
        self.wb.save(self.filename)


# -------------------------
# Demo / run example
# -------------------------
if __name__ == "__main__":
    #random.seed(1)

    logger = ExcelLogger("match_log.xlsx", max_units=10) 
    
    state = MatchState()

    # Example manual placements:
    # place a single troop at (7,10) (on P1 side) -> it should find the nearest lane and advance/cross
    
    # Extra random spawns will also happen according to RANDOM_SPAWN_PROB

    print("Starting simulation for", SIM_SECONDS, "sim seconds (DT=", DT, "s) =>", MAX_STEPS, "steps")
    
    while not state.game_over:
        step(state)
        if state.timestep % int(1.0 / DT) == 0:  # every 1 simulated second
            print(
                f"t={state.timestep} | sec={state.timestep*DT:.1f} | "
                f"P1_units={len(state.units_p1)} P2_units={len(state.units_p2)} | "
                f"P1_king_hp={state.towers_p1[0].hp:.0f} P2_king_hp={state.towers_p2[0].hp:.0f}"
                f"unit location {state.units_p2[0].x, state.units_p2[0].y}"
            )
    while not state.game_over:
        step(state)
        if state.timestep % int(1.0 / DT) == 0:  # every 3 simulated seconds
            logger.log(state)
            print("P1 hand:", [c.name for c in state.cards_p1.hand], "elixir", state.elixir_p1)
            print("P2 hand:", [c.name for c in state.cards_p2.hand], "elixir", state.elixir_p2)
    
    logger.save()

    print("Game Over at sec=", state.timestep * DT)'''