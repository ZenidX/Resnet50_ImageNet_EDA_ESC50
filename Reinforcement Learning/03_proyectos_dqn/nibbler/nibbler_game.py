"""
Nibbler (Snake) con Visualización Pygame + Reinforcement Learning
==================================================================

VARIANTES DE ENTRENAMIENTO:
  A -- Estándar   (--variant standard):  DQN básico, reward por comida/muerte
  B -- Shaping    (--variant shaped):    Penaliza bucles, premia exploración
  C -- Curiosidad (--variant curiosity): Bonus intrínseco por estados nuevos

Ejecutar:
    python nibbler_game.py              # Jugar manualmente
    python nibbler_game.py --train      # Entrenar agente (variante A por defecto)
    python nibbler_game.py --train --variant shaped    # Entrenar variante B
    python nibbler_game.py --train --variant curiosity # Entrenar variante C
    python nibbler_game.py --demo       # Ver agente entrenado

Controles manuales:
    Flechas: Mover serpiente
    R: Reiniciar
    ESC: Salir
"""

import pygame
import numpy as np
import random
from collections import deque
import argparse
import os

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
RED = (255, 50, 50)
GRAY = (40, 40, 40)
BLUE = (50, 100, 200)

# Configuración
GRID_SIZE = 15
CELL_SIZE = 40
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS_PLAY = 10
FPS_TRAIN = 0  # Sin límite durante entrenamiento
FPS_DEMO = 8


class NibblerGame:
    """Juego Nibbler con visualización Pygame."""

    def __init__(self, grid_size=GRID_SIZE, render=True):
        self.grid_size = grid_size
        self.cell_size = CELL_SIZE
        self.render_enabled = render

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((grid_size * CELL_SIZE, grid_size * CELL_SIZE + 50))
            pygame.display.set_caption("Nibbler - Reinforcement Learning")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        # Direcciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.reset()

    def reset(self):
        """Reinicia el juego."""
        center = self.grid_size // 2
        self.snake = deque([(center, center)])
        self.direction = 1  # Empezar yendo a la derecha
        self.score = 0
        self.steps = 0
        self.game_over = False
        self._place_food()
        return self._get_state()

    def _place_food(self):
        """Coloca comida en posición aleatoria."""
        empty = [(i, j) for i in range(self.grid_size)
                 for j in range(self.grid_size) if (i, j) not in self.snake]
        self.food = random.choice(empty) if empty else None

    def _get_state(self):
        """Devuelve el estado como array para el agente."""
        head = self.snake[0]
        food = self.food if self.food else (0, 0)

        # Estado: [peligro_arriba, peligro_derecha, peligro_abajo, peligro_izquierda,
        #          dir_arriba, dir_derecha, dir_abajo, dir_izquierda,
        #          comida_arriba, comida_derecha, comida_abajo, comida_izquierda]
        state = []

        # Peligro en cada dirección
        for d in range(4):
            delta = self.directions[d]
            next_pos = (head[0] + delta[0], head[1] + delta[1])
            danger = (next_pos[0] < 0 or next_pos[0] >= self.grid_size or
                     next_pos[1] < 0 or next_pos[1] >= self.grid_size or
                     next_pos in self.snake)
            state.append(1 if danger else 0)

        # Dirección actual (one-hot)
        for d in range(4):
            state.append(1 if self.direction == d else 0)

        # Dirección de la comida
        state.append(1 if food[0] < head[0] else 0)  # Comida arriba
        state.append(1 if food[1] > head[1] else 0)  # Comida derecha
        state.append(1 if food[0] > head[0] else 0)  # Comida abajo
        state.append(1 if food[1] < head[1] else 0)  # Comida izquierda

        return np.array(state, dtype=np.float32)

    def step(self, action):
        """Ejecuta una acción y devuelve (state, reward, done)."""
        self.steps += 1

        # Evitar ir en dirección opuesta
        opposite = (self.direction + 2) % 4
        if action == opposite and len(self.snake) > 1:
            action = self.direction

        self.direction = action
        head = self.snake[0]
        delta = self.directions[action]
        new_head = (head[0] + delta[0], head[1] + delta[1])

        # Colisiones
        reward = 0
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            self.game_over = True
            reward = -10
            return self._get_state(), reward, True

        self.snake.appendleft(new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            # Pequeña recompensa por acercarse a la comida
            old_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 0.1 if new_dist < old_dist else -0.1

        # Límite de pasos
        if self.steps > self.grid_size * self.grid_size * 2:
            self.game_over = True

        return self._get_state(), reward, self.game_over

    def render(self):
        """Dibuja el juego."""
        if not self.render_enabled:
            return

        self.screen.fill(BLACK)

        # Dibujar cuadrícula
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.screen, GRAY,
                           (i * self.cell_size, 0),
                           (i * self.cell_size, self.grid_size * self.cell_size))
            pygame.draw.line(self.screen, GRAY,
                           (0, i * self.cell_size),
                           (self.grid_size * self.cell_size, i * self.cell_size))

        # Dibujar serpiente
        for i, (row, col) in enumerate(self.snake):
            color = DARK_GREEN if i == 0 else GREEN
            rect = pygame.Rect(col * self.cell_size + 2, row * self.cell_size + 2,
                             self.cell_size - 4, self.cell_size - 4)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)

        # Dibujar comida
        if self.food:
            center = (self.food[1] * self.cell_size + self.cell_size // 2,
                     self.food[0] * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.screen, RED, center, self.cell_size // 2 - 4)

        # Dibujar puntuación
        score_text = self.font.render(f"Score: {self.score}  Length: {len(self.snake)}",
                                      True, WHITE)
        self.screen.blit(score_text, (10, self.grid_size * self.cell_size + 10))

        pygame.display.flip()

    def close(self):
        if self.render_enabled:
            pygame.quit()


# ============================================================================
# AGENTE DQN
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch no disponible. Instalar con: pip install torch")


if TORCH_AVAILABLE:
    class DQN(nn.Module):
        def __init__(self, input_size=12, n_actions=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, n_actions)
            )

        def forward(self, x):
            return self.net(x)


    class DQNAgent:
        def __init__(self, state_size=12, n_actions=4, lr=0.001, gamma=0.95,
                     epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
            self.n_actions = n_actions
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.q_network = DQN(state_size, n_actions).to(self.device)
            self.target_network = DQN(state_size, n_actions).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())

            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
            self.memory = deque(maxlen=100000)
            self.batch_size = 64

        def act(self, state):
            if random.random() < self.epsilon:
                return random.randint(0, self.n_actions - 1)

            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.q_network(state_t).argmax().item()

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def replay(self):
            if len(self.memory) < self.batch_size:
                return 0

            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones_t = torch.FloatTensor(dones).to(self.device)

            q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_q = self.target_network(next_states_t).max(1)[0]
                target = rewards_t + (1 - dones_t) * self.gamma * next_q

            loss = nn.MSELoss()(q_values, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def update_target(self):
            self.target_network.load_state_dict(self.q_network.state_dict())

        def decay_epsilon(self):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        def save(self, path):
            torch.save({
                'q_network': self.q_network.state_dict(),
                'epsilon': self.epsilon
            }, path)

        def load(self, path):
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['q_network'])
            self.epsilon = checkpoint.get('epsilon', 0.01)


if TORCH_AVAILABLE:
    class CuriosityModule(nn.Module):
        """
        Módulo de Curiosidad Intrínseca (Intrinsic Curiosity Module - ICM).

        Aprende a predecir el siguiente estado dado el estado actual y la acción.
        El error de predicción sirve como recompensa intrínseca: cuanto más
        sorprendente es la transición (estado poco visto), mayor la recompensa.

        Referencia: Pathak et al., 2017 - "Curiosity-driven Exploration"
        """
        def __init__(self, state_size=12, n_actions=4, hidden=64):
            super().__init__()
            # Modelo de transición: predice next_state dado (state, action)
            self.forward_model = nn.Sequential(
                nn.Linear(state_size + n_actions, hidden),
                nn.ReLU(),
                nn.Linear(hidden, state_size)
            )
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        def predict(self, state, action_onehot):
            """Predice el siguiente estado."""
            x = torch.cat([state, action_onehot], dim=-1)
            return self.forward_model(x)

        def intrinsic_reward(self, state, action, next_state, n_actions=4):
            """
            Calcula la recompensa intrínseca como error de predicción.
            Mayor error = estado más sorprendente = mayor curiosidad.
            """
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
                action_onehot = torch.zeros(1, n_actions)
                action_onehot[0, action] = 1.0
                predicted = self.predict(state_t, action_onehot)
                error = torch.mean((predicted - next_state_t) ** 2).item()
            return error  # Error = recompensa intrínseca

        def train_step(self, state, action, next_state, n_actions=4):
            """Actualiza el modelo de transición."""
            state_t = torch.FloatTensor(state).unsqueeze(0)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
            action_onehot = torch.zeros(1, n_actions)
            action_onehot[0, action] = 1.0
            predicted = self.predict(state_t, action_onehot)
            loss = nn.MSELoss()(predicted, next_state_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()


def play_manual():
    """Modo de juego manual con flechas."""
    game = NibblerGame(grid_size=GRID_SIZE)
    game.reset()

    running = True
    action = 1  # Empezar yendo a la derecha

    print("=== NIBBLER - Modo Manual ===")
    print("Controles: Flechas para mover, R para reiniciar, ESC para salir")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                elif event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3

        if not game.game_over:
            _, _, done = game.step(action)
            if done:
                print(f"Game Over! Score: {game.score}")

        game.render()
        game.clock.tick(FPS_PLAY)

    game.close()


def train_agent(episodes=500, render_every=50):
    """Entrena el agente DQN con visualización periódica."""
    if not TORCH_AVAILABLE:
        print("Error: PyTorch no disponible")
        return

    print("=== ENTRENAMIENTO DQN ===")
    print(f"Episodios: {episodes}")
    print(f"Visualización cada {render_every} episodios")

    agent = DQNAgent()
    scores = []
    best_score = 0

    for ep in range(episodes):
        # Renderizar periódicamente
        render = (ep % render_every == 0)
        game = NibblerGame(grid_size=GRID_SIZE, render=render)

        state = game.reset()
        total_reward = 0

        while not game.game_over:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()
                        agent.save("nibbler_agent.pth")
                        return agent, scores

            action = agent.act(state)
            next_state, reward, done = game.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

            if render:
                game.render()
                game.clock.tick(30)  # Más rápido durante entrenamiento visual

        agent.decay_epsilon()

        if ep % 10 == 0:
            agent.update_target()

        scores.append(game.score)

        if game.score > best_score:
            best_score = game.score
            agent.save("nibbler_best.pth")

        if (ep + 1) % 10 == 0:
            avg = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            print(f"Ep {ep+1:4d} | Score: {game.score:2d} | Avg: {avg:.1f} | "
                  f"Best: {best_score} | ε: {agent.epsilon:.3f}")

        if render:
            game.close()

    agent.save("nibbler_agent.pth")
    print(f"\n✅ Entrenamiento completado. Mejor score: {best_score}")

    return agent, scores


def demo_agent():
    """Muestra el agente entrenado jugando."""
    if not TORCH_AVAILABLE:
        print("Error: PyTorch no disponible")
        return

    model_path = "nibbler_best.pth"
    if not os.path.exists(model_path):
        model_path = "nibbler_agent.pth"

    if not os.path.exists(model_path):
        print("Error: No hay modelo entrenado. Ejecuta primero --train")
        return

    print("=== DEMO AGENTE ENTRENADO ===")
    print("Presiona ESC para salir, R para reiniciar")

    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0  # Sin exploración

    game = NibblerGame(grid_size=GRID_SIZE)

    running = True
    while running:
        state = game.reset()

        while not game.game_over and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset()

            action = agent.act(state)
            state, _, done = game.step(action)

            game.render()
            game.clock.tick(FPS_DEMO)

        if running:
            print(f"Score: {game.score} | Length: {len(game.snake)}")
            pygame.time.wait(1000)

    game.close()


def train_shaped(episodes=500, render_every=50):
    """
    Variante B: Entrenamiento con Reward Shaping.

    El reward shaping modifica la función de recompensa para guiar
    mejor al agente. En lugar de solo premiar comer y penalizar morir,
    añadimos señales que desalientan comportamientos no deseados:
    - Penalizar dar vueltas en círculos (visitar posiciones repetidas)
    - Penalizar quedarse sin comer durante mucho tiempo
    - Premiar explorar celdas no visitadas antes

    Clave: El reward shaping NO cambia la política óptima si se diseña
    correctamente (potential-based shaping), pero puede acelerar enormemente
    el aprendizaje inicial.

    Ejecutar: python nibbler_game.py --train --variant shaped
    """
    if not TORCH_AVAILABLE:
        print("Error: PyTorch no disponible")
        return

    print("=== VARIANTE B: REWARD SHAPING ===")
    print("Penaliza bucles, premia exploración")
    print(f"Episodios: {episodes}\n")

    agent = DQNAgent()
    scores = []
    best_score = 0
    recent_positions = deque(maxlen=20)  # Para detectar bucles

    for ep in range(episodes):
        render = (ep % render_every == 0)
        game = NibblerGame(grid_size=GRID_SIZE, render=render)
        state = game.reset()

        visited_cells = set()
        visited_cells.add(game.snake[0])
        steps_since_food = 0
        total_reward = 0
        recent_positions.clear()

        while not game.game_over:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()
                        agent.save("nibbler_shaped.pth")
                        return agent, scores

            action = agent.act(state)
            next_state, base_reward, done = game.step(action)

            head = game.snake[0]

            # --- Reward Shaping ---
            shaped_reward = base_reward

            if base_reward == 10:
                # Comió: resetear contador y limpiar celdas visitadas
                steps_since_food = 0
                visited_cells = set(game.snake)
            elif not done:
                steps_since_food += 1

                # Penalizar bucles: posición repetida recientemente
                if head in recent_positions:
                    shaped_reward -= 2.0

                # Penalizar no comer durante mucho tiempo
                if steps_since_food > 50:
                    shaped_reward -= 0.5

                # Premiar explorar celdas nuevas
                if head not in visited_cells:
                    shaped_reward += 0.2
                    visited_cells.add(head)

            recent_positions.append(head)

            agent.remember(state, action, shaped_reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += shaped_reward

            if render:
                game.render()
                game.clock.tick(30)

        agent.decay_epsilon()
        if ep % 10 == 0:
            agent.update_target()

        scores.append(game.score)
        if game.score > best_score:
            best_score = game.score
            agent.save("nibbler_shaped_best.pth")

        if (ep + 1) % 10 == 0:
            avg = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            print(f"Ep {ep+1:4d} | Score: {game.score:2d} | Avg: {avg:.1f} | "
                  f"Best: {best_score} | eps: {agent.epsilon:.3f}")

        if render:
            game.close()

    agent.save("nibbler_shaped.pth")
    print(f"\nVariante B completada. Mejor score: {best_score}")
    return agent, scores


def train_curiosity(episodes=500, render_every=50, curiosity_scale=0.5):
    """
    Variante C: Entrenamiento con Curiosidad Intrínseca.

    Añade un módulo de curiosidad que genera recompensa extra cuando
    el agente visita estados poco frecuentes o sorprendentes.

    El módulo aprende a predecir el siguiente estado. Cuando falla
    mucho en su predicción (estado raro/nuevo), da una recompensa
    adicional al agente por "explorar lo desconocido".

    Resultado: El agente tiende a explorar todo el tablero antes de
    centrarse solo en buscar comida.

    Ejecutar: python nibbler_game.py --train --variant curiosity
    """
    if not TORCH_AVAILABLE:
        print("Error: PyTorch no disponible")
        return

    print("=== VARIANTE C: CURIOSIDAD INTRÍNSECA ===")
    print("El agente recibe bonus por explorar estados nuevos")
    print(f"Escala de curiosidad: {curiosity_scale}")
    print(f"Episodios: {episodes}\n")

    agent = DQNAgent()
    curiosity = CuriosityModule(state_size=12, n_actions=4)
    scores = []
    best_score = 0
    intrinsic_rewards_history = []

    for ep in range(episodes):
        render = (ep % render_every == 0)
        game = NibblerGame(grid_size=GRID_SIZE, render=render)
        state = game.reset()
        total_intrinsic = 0

        while not game.game_over:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()
                        agent.save("nibbler_curiosity.pth")
                        return agent, scores

            action = agent.act(state)
            next_state, extrinsic_reward, done = game.step(action)

            # Recompensa intrínseca por curiosidad
            intrinsic = curiosity.intrinsic_reward(state, action, next_state)
            total_reward = extrinsic_reward + curiosity_scale * intrinsic
            total_intrinsic += intrinsic

            # Entrenar módulo de curiosidad
            curiosity.train_step(state, action, next_state)

            agent.remember(state, action, total_reward, next_state, done)
            agent.replay()
            state = next_state

            if render:
                game.render()
                game.clock.tick(30)

        agent.decay_epsilon()
        if ep % 10 == 0:
            agent.update_target()

        scores.append(game.score)
        intrinsic_rewards_history.append(total_intrinsic)

        if game.score > best_score:
            best_score = game.score
            agent.save("nibbler_curiosity_best.pth")

        if (ep + 1) % 10 == 0:
            avg = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            avg_intr = np.mean(intrinsic_rewards_history[-10:])
            print(f"Ep {ep+1:4d} | Score: {game.score:2d} | Avg: {avg:.1f} | "
                  f"Best: {best_score} | Intrínseca: {avg_intr:.3f} | eps: {agent.epsilon:.3f}")

        if render:
            game.close()

    agent.save("nibbler_curiosity.pth")
    print(f"\nVariante C completada. Mejor score: {best_score}")
    return agent, scores, intrinsic_rewards_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nibbler con RL")
    parser.add_argument("--train", action="store_true", help="Entrenar agente")
    parser.add_argument("--demo", action="store_true", help="Ver agente entrenado")
    parser.add_argument("--episodes", type=int, default=500, help="Episodios de entrenamiento")
    parser.add_argument("--variant", type=str, default="standard",
                        choices=["standard", "shaped", "curiosity"],
                        help=(
                            "Variante de entrenamiento:\n"
                            "  standard  - (A) DQN con reward básico\n"
                            "  shaped    - (B) Reward shaping: penaliza bucles, premia exploración\n"
                            "  curiosity - (C) Curiosidad intrínseca: bonus por estados nuevos"
                        ))
    args = parser.parse_args()

    if args.train:
        if args.variant == "standard":
            print("Variante A: DQN Estándar")
            train_agent(episodes=args.episodes)
        elif args.variant == "shaped":
            print("Variante B: Reward Shaping")
            train_shaped(episodes=args.episodes)
        elif args.variant == "curiosity":
            print("Variante C: Curiosidad Intrínseca")
            train_curiosity(episodes=args.episodes)
    elif args.demo:
        demo_agent()
    else:
        play_manual()
