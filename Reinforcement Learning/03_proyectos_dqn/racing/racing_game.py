"""
Racing Game con Múltiples Agentes RL
=====================================

Un juego de carreras donde varios coches (agentes) compiten simultáneamente.
Cada coche aprende a evitar obstáculos y mantenerse en la pista.

Ejecutar:
    python racing_game.py                    # Ver 4 agentes aleatorios
    python racing_game.py --train            # Entrenar agentes
    python racing_game.py --demo             # Ver agentes entrenados
    python racing_game.py --cars 8           # Número de coches

Controles:
    ESC: Salir
    R: Reiniciar carrera
    +/-: Ajustar velocidad
"""

import pygame
import numpy as np
import random
from collections import deque
import argparse
import os
import math

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 100, 255)
YELLOW = (255, 255, 50)
ORANGE = (255, 150, 50)
PURPLE = (150, 50, 255)
CYAN = (50, 255, 255)
PINK = (255, 100, 150)

CAR_COLORS = [RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, CYAN, PINK]

# Configuración
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
TRACK_WIDTH = 120
CAR_WIDTH = 20
CAR_HEIGHT = 35
FPS = 60


class Car:
    """Un coche individual en la carrera."""

    def __init__(self, car_id, color, start_pos, start_angle=0):
        self.id = car_id
        self.color = color
        self.x, self.y = start_pos
        self.angle = start_angle  # En grados
        self.speed = 0
        self.max_speed = 8
        self.acceleration = 0.3
        self.turn_speed = 4
        self.friction = 0.98

        self.alive = True
        self.distance = 0
        self.laps = 0
        self.checkpoint = 0
        self.time_alive = 0
        self.fitness = 0

        # Sensores (raycast)
        self.sensor_angles = [-60, -30, 0, 30, 60]  # Ángulos relativos
        self.sensor_length = 150
        self.sensor_readings = [0] * len(self.sensor_angles)

    def reset(self, pos, angle=0):
        self.x, self.y = pos
        self.angle = angle
        self.speed = 0
        self.alive = True
        self.distance = 0
        self.laps = 0
        self.checkpoint = 0
        self.time_alive = 0
        self.fitness = 0
        self.sensor_readings = [0] * len(self.sensor_angles)

    def get_state(self):
        """Estado para el agente RL."""
        state = list(self.sensor_readings)  # 5 sensores normalizados
        state.append(self.speed / self.max_speed)  # Velocidad normalizada
        return np.array(state, dtype=np.float32)

    def update(self, action, track):
        """
        Actualiza el coche según la acción.
        Acciones: 0=nada, 1=acelerar, 2=frenar, 3=izquierda, 4=derecha
                  5=acelerar+izq, 6=acelerar+der, 7=frenar+izq, 8=frenar+der

        Física realista:
        - Solo gira si se está moviendo
        - Giro proporcional a la velocidad
        - Marcha atrás más lenta
        """
        if not self.alive:
            return

        # Procesar acción: Acelerar/Frenar (sin marcha atrás)
        if action in [1, 5, 6]:  # Acelerar
            self.speed = min(self.max_speed, self.speed + self.acceleration)
        elif action in [2, 7, 8]:  # Frenar (solo frenar, no marcha atrás)
            self.speed = max(0, self.speed - self.acceleration * 2.0)

        # Procesar acción: Girar (SOLO si hay velocidad)
        # El giro es proporcional a la velocidad (más rápido = gira menos cerrado)
        if self.speed > 0.1:  # Necesita velocidad mínima para girar
            # Factor de giro: más efectivo a velocidades medias
            turn_factor = min(1.0, self.speed / 3.0)  # Máximo giro a velocidad 3+
            actual_turn = self.turn_speed * turn_factor

            if action in [3, 5, 7]:  # Girar izquierda
                self.angle += actual_turn
            elif action in [4, 6, 8]:  # Girar derecha
                self.angle -= actual_turn

        # Aplicar fricción
        self.speed *= self.friction

        # Mover
        rad = math.radians(self.angle)
        self.x += self.speed * math.sin(rad)
        self.y -= self.speed * math.cos(rad)

        self.time_alive += 1
        self.distance += abs(self.speed)

        # Actualizar sensores
        self._update_sensors(track)

        # Verificar colisiones
        if track.check_collision(self.x, self.y):
            self.alive = False

        # Calcular fitness
        self.fitness = self.distance + self.laps * 1000 + self.checkpoint * 100

    def _update_sensors(self, track):
        """Actualiza las lecturas de los sensores."""
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = math.radians(self.angle + sensor_angle)
            dist = 0

            for d in range(1, self.sensor_length + 1, 5):
                check_x = self.x + d * math.sin(angle)
                check_y = self.y - d * math.cos(angle)

                if track.check_collision(check_x, check_y):
                    dist = d
                    break
                dist = d

            self.sensor_readings[i] = dist / self.sensor_length

    def draw(self, screen):
        """Dibuja el coche y sus sensores."""
        if not self.alive:
            color = GRAY
        else:
            color = self.color

        # Dibujar sensores (líneas) - misma dirección que el movimiento
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = math.radians(self.angle + sensor_angle)
            dist = self.sensor_readings[i] * self.sensor_length
            end_x = self.x + dist * math.sin(angle)
            end_y = self.y - dist * math.cos(angle)

            # Color según distancia (rojo = cerca, verde = lejos)
            if self.alive:
                r = int(255 * (1 - self.sensor_readings[i]))
                g = int(255 * self.sensor_readings[i])
                sensor_color = (r, g, 0)
            else:
                sensor_color = (50, 50, 50)
            pygame.draw.line(screen, sensor_color, (self.x, self.y), (end_x, end_y), 2)

        # Dibujar coche (rectángulo rotado)
        car_surface = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, color, (0, 0, CAR_WIDTH, CAR_HEIGHT), border_radius=3)
        # Indicador de frente (parte delantera del coche)
        pygame.draw.rect(car_surface, WHITE, (CAR_WIDTH//4, 2, CAR_WIDTH//2, 5))

        # Rotar en sentido contrario para alinear visual con movimiento
        rotated = pygame.transform.rotate(car_surface, -self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect)


class Track:
    """Pista de carreras."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.track_width = TRACK_WIDTH

        # Definir la pista como un circuito
        self.center_points = self._create_track()
        self.checkpoints = self._create_checkpoints()

        # Pre-renderizar la pista
        self.surface = self._render_track()

    def _create_track(self):
        """Crea los puntos centrales de la pista (circuito ovalado)."""
        cx, cy = self.width // 2, self.height // 2
        rx, ry = 350, 250  # Radios del óvalo

        points = []
        for angle in range(0, 360, 5):
            rad = math.radians(angle)
            x = cx + rx * math.cos(rad)
            y = cy + ry * math.sin(rad)
            points.append((x, y))

        return points

    def _create_checkpoints(self):
        """Crea checkpoints para medir progreso."""
        return self.center_points[::18]  # Cada 18 puntos

    def _render_track(self):
        """Renderiza la pista en una superficie."""
        surface = pygame.Surface((self.width, self.height))
        surface.fill(DARK_GRAY)

        # Dibujar bordes de la pista
        if len(self.center_points) > 2:
            # Pista exterior e interior
            outer_points = []
            inner_points = []

            for i, (x, y) in enumerate(self.center_points):
                # Calcular normal
                next_i = (i + 1) % len(self.center_points)
                prev_i = (i - 1) % len(self.center_points)

                dx = self.center_points[next_i][0] - self.center_points[prev_i][0]
                dy = self.center_points[next_i][1] - self.center_points[prev_i][1]
                length = math.sqrt(dx*dx + dy*dy)

                if length > 0:
                    nx, ny = -dy/length, dx/length
                else:
                    nx, ny = 0, 1

                outer_points.append((x + nx * self.track_width/2, y + ny * self.track_width/2))
                inner_points.append((x - nx * self.track_width/2, y - ny * self.track_width/2))

            # Dibujar pista (verde oscuro)
            pygame.draw.polygon(surface, (30, 80, 30), outer_points)
            pygame.draw.polygon(surface, DARK_GRAY, inner_points)

            # Bordes blancos
            pygame.draw.lines(surface, WHITE, True, outer_points, 3)
            pygame.draw.lines(surface, WHITE, True, inner_points, 3)

            # Línea central discontinua
            for i in range(0, len(self.center_points), 4):
                if i + 2 < len(self.center_points):
                    pygame.draw.line(surface, YELLOW,
                                   self.center_points[i],
                                   self.center_points[i+2], 2)

        return surface

    def check_collision(self, x, y):
        """Verifica si un punto está fuera de la pista."""
        # Verificar límites de ventana
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        # Verificar si está en la pista (punto más cercano al centro)
        min_dist = float('inf')
        for cx, cy in self.center_points:
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            min_dist = min(min_dist, dist)

        return min_dist > self.track_width / 2

    def get_start_positions(self, n_cars):
        """Devuelve posiciones de salida para n coches."""
        positions = []
        start_point = self.center_points[0]
        next_point = self.center_points[1]

        # Calcular dirección inicial
        dx = next_point[0] - start_point[0]
        dy = next_point[1] - start_point[1]
        angle = math.degrees(math.atan2(dx, -dy))

        # Distribuir coches
        for i in range(n_cars):
            row = i // 2
            col = i % 2
            offset_x = (col - 0.5) * 30
            offset_y = row * 40

            # Rotar offset según ángulo
            rad = math.radians(angle)
            rx = offset_x * math.cos(rad) - offset_y * math.sin(rad)
            ry = offset_x * math.sin(rad) + offset_y * math.cos(rad)

            pos = (start_point[0] + rx, start_point[1] + ry)
            positions.append((pos, angle))

        return positions

    def draw(self, screen):
        screen.blit(self.surface, (0, 0))


class RacingGame:
    """Juego de carreras con múltiples coches."""

    def __init__(self, n_cars=4, render=True):
        self.n_cars = n_cars
        self.render_enabled = render

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Racing RL - Múltiples Agentes")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        self.track = Track(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.cars = []

        self.reset()

    def reset(self):
        """Reinicia la carrera."""
        start_positions = self.track.get_start_positions(self.n_cars)

        self.cars = []
        for i in range(self.n_cars):
            color = CAR_COLORS[i % len(CAR_COLORS)]
            pos, angle = start_positions[i]
            car = Car(i, color, pos, angle)
            self.cars.append(car)

        return [car.get_state() for car in self.cars]

    def step(self, actions):
        """
        Ejecuta un paso para todos los coches.
        actions: lista de acciones, una por coche
        """
        states = []
        rewards = []
        dones = []

        for car, action in zip(self.cars, actions):
            old_distance = car.distance

            car.update(action, self.track)

            # Calcular recompensa
            if not car.alive:
                reward = -10
            else:
                reward = (car.distance - old_distance) * 0.1  # Recompensa por avanzar
                if car.speed > 0:
                    reward += 0.1  # Bonus por moverse

            states.append(car.get_state())
            rewards.append(reward)
            dones.append(not car.alive)

        all_done = all(not car.alive for car in self.cars)

        return states, rewards, dones, all_done

    def render(self):
        if not self.render_enabled:
            return

        self.screen.fill(DARK_GRAY)
        self.track.draw(self.screen)

        # Dibujar coches
        for car in self.cars:
            car.draw(self.screen)

        # Panel de información
        y_offset = 10
        for car in self.cars:
            status = "ALIVE" if car.alive else "DEAD"
            color = car.color if car.alive else GRAY
            text = f"Car {car.id}: {status} | Dist: {car.distance:.0f} | Fitness: {car.fitness:.0f}"
            rendered = self.font.render(text, True, color)
            self.screen.blit(rendered, (10, y_offset))
            y_offset += 20

        pygame.display.flip()

    def close(self):
        if self.render_enabled:
            pygame.quit()


# ============================================================================
# AGENTE DQN PARA COCHES
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CarDQN(nn.Module):
        def __init__(self, state_size=6, n_actions=9):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions)
            )

        def forward(self, x):
            return self.net(x)


    class CarAgent:
        def __init__(self, agent_id, state_size=6, n_actions=9):
            self.id = agent_id
            self.n_actions = n_actions
            self.gamma = 0.95
            self.epsilon = 1.0
            self.epsilon_min = 0.05
            self.epsilon_decay = 0.995

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.q_network = CarDQN(state_size, n_actions).to(self.device)
            self.target_network = CarDQN(state_size, n_actions).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())

            self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
            self.memory = deque(maxlen=50000)
            self.batch_size = 32

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


def run_random(n_cars=4):
    """Ejecuta con agentes aleatorios para demostración."""
    print(f"=== RACING GAME - {n_cars} Coches Aleatorios ===")
    print("Controles: ESC=Salir, R=Reiniciar")

    game = RacingGame(n_cars=n_cars)

    running = True
    while running:
        game.reset()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        break  # Reiniciar

            # Acciones aleatorias para cada coche
            actions = [random.randint(0, 8) for _ in range(n_cars)]
            _, _, _, all_done = game.step(actions)

            game.render()
            game.clock.tick(FPS)

            if all_done:
                pygame.time.wait(1000)
                break

    game.close()


def train_agents(n_cars=4, episodes=200):
    """Entrena múltiples agentes simultáneamente."""
    if not TORCH_AVAILABLE:
        print("Error: PyTorch no disponible")
        return

    print(f"=== ENTRENAMIENTO - {n_cars} Coches ===")
    print(f"Episodios: {episodes}")

    game = RacingGame(n_cars=n_cars, render=True)
    agents = [CarAgent(i) for i in range(n_cars)]

    best_fitness = [0] * n_cars

    for ep in range(episodes):
        states = game.reset()

        step_count = 0
        max_steps = 2000

        while step_count < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    for i, agent in enumerate(agents):
                        torch.save(agent.q_network.state_dict(), f"car_agent_{i}.pth")
                    game.close()
                    return

            # Cada agente decide su acción
            actions = [agent.act(state) for agent, state in zip(agents, states)]

            # Ejecutar paso
            next_states, rewards, dones, all_done = game.step(actions)

            # Almacenar experiencias
            for i, agent in enumerate(agents):
                agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                agent.replay()

            states = next_states
            step_count += 1

            # Renderizar cada pocos frames
            if step_count % 3 == 0:
                game.render()
                game.clock.tick(120)

            if all_done:
                break

        # Actualizar agentes
        for i, agent in enumerate(agents):
            agent.decay_epsilon()
            if ep % 10 == 0:
                agent.update_target()

            if game.cars[i].fitness > best_fitness[i]:
                best_fitness[i] = game.cars[i].fitness

        if (ep + 1) % 10 == 0:
            avg_fitness = np.mean([car.fitness for car in game.cars])
            print(f"Ep {ep+1:3d} | Avg Fitness: {avg_fitness:.0f} | "
                  f"Best: {max(best_fitness):.0f} | eps: {agents[0].epsilon:.3f}")

    # Guardar modelos
    for i, agent in enumerate(agents):
        torch.save(agent.q_network.state_dict(), f"car_agent_{i}.pth")

    print("\nEntrenamiento completado")
    game.close()


def demo_agents(n_cars=4):
    """Muestra los agentes entrenados."""
    if not TORCH_AVAILABLE:
        print("Error: PyTorch no disponible")
        return

    print(f"=== DEMO - {n_cars} Coches Entrenados ===")

    game = RacingGame(n_cars=n_cars)
    agents = []

    for i in range(n_cars):
        agent = CarAgent(i)
        path = f"car_agent_{i}.pth"
        if os.path.exists(path):
            agent.q_network.load_state_dict(torch.load(path, map_location=agent.device))
            agent.epsilon = 0
            print(f"Coche {i}: Modelo cargado")
        else:
            print(f"Coche {i}: Sin modelo, usando aleatorio")
        agents.append(agent)

    running = True
    while running:
        states = game.reset()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        break

            actions = [agent.act(state) for agent, state in zip(agents, states)]
            states, _, _, all_done = game.step(actions)

            game.render()
            game.clock.tick(FPS)

            if all_done:
                pygame.time.wait(1500)
                break

    game.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Racing Game con RL")
    parser.add_argument("--train", action="store_true", help="Entrenar agentes")
    parser.add_argument("--demo", action="store_true", help="Ver agentes entrenados")
    parser.add_argument("--cars", type=int, default=4, help="Número de coches")
    parser.add_argument("--episodes", type=int, default=200, help="Episodios de entrenamiento")
    args = parser.parse_args()

    if args.train:
        train_agents(n_cars=args.cars, episodes=args.episodes)
    elif args.demo:
        demo_agents(n_cars=args.cars)
    else:
        run_random(n_cars=args.cars)
