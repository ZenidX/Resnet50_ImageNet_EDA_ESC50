"""
MiniGrid - Navegación en Laberintos 2D
======================================
Entorno de cuadrícula donde el agente aprende a navegar,
abrir puertas y recoger objetos.

VARIANTES:
  A — MLP + Flat (--variant flat):   Vector aplanado, sin estructura espacial
  B — CNN        (--variant cnn):    CNN que aprovecha la estructura de la imagen
  C — Curriculum (--curriculum):     Empty → FourRooms → DoorKey → LavaCrossing

Instalación:
    pip install minigrid stable-baselines3

Uso:
    python minigrid_navegacion.py                     # Entrenar (var. B por defecto)
    python minigrid_navegacion.py --variant flat      # Var. A: MLP
    python minigrid_navegacion.py --variant cnn       # Var. B: CNN
    python minigrid_navegacion.py --curriculum        # Var. C: curriculum
    python minigrid_navegacion.py --demo              # Ver agente
    python minigrid_navegacion.py --env DoorKey       # Otro entorno
    python minigrid_navegacion.py --manual            # Jugar manualmente
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, FullyObsWrapper

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy

import torch
import torch.nn as nn


# =============================================================================
# WRAPPERS PERSONALIZADOS
# =============================================================================

class FlatObsWrapper(gym.ObservationWrapper):
    """
    Aplana la observación de MiniGrid para usarla con MLP.
    """
    def __init__(self, env):
        super().__init__(env)
        # Observación original: (7, 7, 3) - vista parcial del agente
        obs_shape = env.observation_space['image'].shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(np.prod(obs_shape),),
            dtype=np.float32
        )

    def observation(self, obs):
        return obs['image'].flatten().astype(np.float32) / 255.0


class SimpleObsWrapper(gym.ObservationWrapper):
    """
    Simplifica la observación a solo la imagen normalizada.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space['image'].shape,
            dtype=np.float32
        )

    def observation(self, obs):
        return obs['image'].astype(np.float32) / 10.0  # Normalizar


# =============================================================================
# EXTRACTOR DE CARACTERÍSTICAS CNN
# =============================================================================

class MinigridCNN(BaseFeaturesExtractor):
    """
    CNN para procesar las observaciones de MiniGrid.
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]  # 3 canales

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcular tamaño de salida
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, *observation_space.shape[:2])
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Cambiar de (B, H, W, C) a (B, C, H, W)
        x = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(x))


# =============================================================================
# CALLBACK DE ENTRENAMIENTO
# =============================================================================

class MinigridCallback(BaseCallback):
    """Callback para registrar progreso en MiniGrid."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []

    def _on_step(self) -> bool:
        # Registrar episodios completados
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                # Éxito si llegó a la meta (recompensa > 0)
                self.successes.append(1 if info['episode']['r'] > 0 else 0)
        return True


# =============================================================================
# ENTORNOS DISPONIBLES
# =============================================================================

ENTORNOS = {
    "Empty": "MiniGrid-Empty-5x5-v0",           # Vacío, ir a la meta
    "Empty8": "MiniGrid-Empty-8x8-v0",          # Vacío más grande
    "FourRooms": "MiniGrid-FourRooms-v0",       # 4 habitaciones
    "DoorKey": "MiniGrid-DoorKey-5x5-v0",       # Abrir puerta con llave
    "DoorKey8": "MiniGrid-DoorKey-8x8-v0",      # Más difícil
    "Unlock": "MiniGrid-Unlock-v0",             # Encontrar llave
    "GoToDoor": "MiniGrid-GoToDoor-5x5-v0",     # Ir a puerta específica
    "SimpleCrossing": "MiniGrid-SimpleCrossingS9N1-v0",  # Cruzar obstáculos
    "LavaCrossing": "MiniGrid-LavaCrossingS9N1-v0",      # Evitar lava
    "DistShift": "MiniGrid-DistShift1-v0",      # Distribución cambiante
}


def crear_entorno(nombre="Empty", render=False):
    """
    Crea un entorno de MiniGrid con wrappers apropiados.

    Args:
        nombre: Nombre del entorno (ver ENTORNOS)
        render: Si renderizar visualmente

    Returns:
        Entorno wrapped
    """
    env_id = ENTORNOS.get(nombre, nombre)
    render_mode = "human" if render else None

    env = gym.make(env_id, render_mode=render_mode)
    env = SimpleObsWrapper(env)

    return env


def entrenar_flat(env_name="Empty", timesteps=50000, algorithm="PPO"):
    """
    Variante A: PPO/DQN con observación APLANADA (MLP).

    La imagen 7×7×3 del agente se aplana a un vector de 147 valores.
    Se usa una red neuronal densa (MLP) para aprender la política.

    Desventaja: el MLP no aprovecha la estructura espacial de la imagen.
    No sabe que los píxeles adyacentes están relacionados espacialmente.

    Ejecutar: python minigrid_navegacion.py --variant flat
    """
    print(f"\n{'='*60}")
    print(f"  Variante A: MLP + Observación Plana")
    print(f"  Entorno: {env_name} | Algoritmo: {algorithm}")
    print(f"  Observación: 7×7×3 = 147 valores aplanados")
    print(f"  Red: MLP (sin estructura espacial)")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'='*60}\n")

    env_id = ENTORNOS.get(env_name, env_name)
    env = gym.make(env_id)
    env = FlatObsWrapper(env)  # Aplana la imagen

    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",  # Red densa
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [256, 256]},  # MLP simple
        )
    else:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0001,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            policy_kwargs={"net_arch": [256, 256]},
        )

    callback = MinigridCallback()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

    save_path = f"minigrid_{env_name.lower()}_flat_{algorithm.lower()}"
    model.save(save_path)
    print(f"\nModelo guardado: {save_path}.zip")
    env.close()
    return model, callback


def entrenar_minigrid(env_name="Empty", timesteps=50000, algorithm="PPO"):
    """
    Variante B: PPO/DQN con CNN personalizada (MinigridCNN).

    Usa convoluciones para procesar la imagen 7×7×3, preservando
    la estructura espacial. La CNN aprende qué características
    visuales son relevantes (paredes, puertas, objetos).

    Ventaja vs Variante A: entiende relaciones espaciales entre píxeles.

    Arquitectura CNN:
        Conv2d(3→16, k=2) → Conv2d(16→32, k=2) → Conv2d(32→64, k=2)
        → Flatten → Linear(64) → política

    Ejecutar: python minigrid_navegacion.py --variant cnn

    Args:
        env_name: Nombre del entorno (ver ENTORNOS)
        timesteps: Pasos de entrenamiento
        algorithm: "PPO" o "DQN"

    Returns:
        Modelo entrenado y callback
    """
    print(f"\n{'='*60}")
    print(f"  Entrenando {algorithm} en MiniGrid-{env_name}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'='*60}\n")

    env = crear_entorno(env_name, render=False)

    # Política con CNN personalizada
    policy_kwargs = {
        "features_extractor_class": MinigridCNN,
        "features_extractor_kwargs": {"features_dim": 64}
    }

    if algorithm == "PPO":
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0003,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            ent_coef=0.01,  # Fomentar exploración
        )
    else:
        model = DQN(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0001,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.2,
        )

    callback = MinigridCallback()

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True
    )

    # Guardar
    save_path = f"minigrid_{env_name.lower()}_{algorithm.lower()}"
    model.save(save_path)
    print(f"\nModelo guardado en: {save_path}.zip")

    env.close()
    return model, callback


def demo_agente(env_name="Empty", model_path=None, n_episodios=5):
    """
    Muestra el agente navegando el laberinto.
    """
    if model_path is None:
        model_path = f"minigrid_{env_name.lower()}_ppo"

    print(f"\nCargando modelo: {model_path}")
    model = PPO.load(model_path)

    env = crear_entorno(env_name, render=True)

    print(f"\nMostrando {n_episodios} episodios en {env_name}...")
    print("Acciones: 0=izq, 1=der, 2=adelante, 3=recoger, 4=soltar, 5=toggle")

    exitos = 0
    for ep in range(n_episodios):
        obs, info = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        exito = "META" if total_reward > 0 else "FALLO"
        if total_reward > 0:
            exitos += 1
        print(f"Episodio {ep+1}: {total_reward:.3f} ({steps} pasos) - {exito}")

    print(f"\nTasa de éxito: {exitos}/{n_episodios} ({100*exitos/n_episodios:.0f}%)")
    env.close()


def jugar_manual(env_name="Empty"):
    """
    Permite jugar manualmente para entender el entorno.
    """
    import pygame

    print(f"\n{'='*60}")
    print(f"  Modo Manual - MiniGrid {env_name}")
    print(f"{'='*60}")
    print("\nControles:")
    print("  Flechas: Girar izq/der, Avanzar")
    print("  Espacio: Recoger/Soltar objeto")
    print("  Enter: Toggle (abrir puerta)")
    print("  R: Reiniciar")
    print("  ESC: Salir\n")

    env = crear_entorno(env_name, render=True)
    obs, info = env.reset()

    total_reward = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = None

                if event.key == pygame.K_LEFT:
                    action = 0  # Girar izquierda
                elif event.key == pygame.K_RIGHT:
                    action = 1  # Girar derecha
                elif event.key == pygame.K_UP:
                    action = 2  # Avanzar
                elif event.key == pygame.K_SPACE:
                    action = 3  # Recoger
                elif event.key == pygame.K_RETURN:
                    action = 5  # Toggle (abrir puerta)
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("Reiniciado")
                elif event.key == pygame.K_ESCAPE:
                    running = False

                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward

                    if reward != 0:
                        print(f"Recompensa: {reward:.3f} | Total: {total_reward:.3f}")

                    if terminated or truncated:
                        resultado = "GANASTE" if total_reward > 0 else "Fallaste"
                        print(f"{resultado}! Recompensa final: {total_reward:.3f}")
                        obs, info = env.reset()
                        total_reward = 0

    env.close()


def plot_training(callback, save_path="minigrid_training.png"):
    """Grafica el progreso del entrenamiento."""
    if not callback.episode_rewards:
        print("No hay datos para graficar")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Recompensas
    rewards = callback.episode_rewards
    window = min(50, len(rewards) // 4) if len(rewards) > 4 else 1

    axes[0].plot(rewards, alpha=0.3, color='blue')
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), smoothed, 'r-', linewidth=2)
    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_title('Recompensa por Episodio')
    axes[0].grid(True, alpha=0.3)

    # Longitud de episodios
    lengths = callback.episode_lengths
    axes[1].plot(lengths, alpha=0.3, color='green')
    if window > 1:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(lengths)), smoothed, 'r-', linewidth=2)
    axes[1].set_xlabel('Episodio')
    axes[1].set_ylabel('Pasos')
    axes[1].set_title('Longitud de Episodio')
    axes[1].grid(True, alpha=0.3)

    # Tasa de éxito
    if callback.successes:
        successes = callback.successes
        cumsum = np.cumsum(successes)
        rate = cumsum / (np.arange(len(successes)) + 1)
        axes[2].plot(rate, 'b-', linewidth=2)
        axes[2].set_xlabel('Episodio')
        axes[2].set_ylabel('Tasa de Éxito')
        axes[2].set_title('Tasa de Éxito Acumulada')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfica guardada: {save_path}")


def curriculum_minigrid(timesteps_por_nivel=30000, algorithm="PPO"):
    """
    Variante C: Curriculum Learning en MiniGrid.

    El agente aprende navegación progresivamente, comenzando por
    entornos simples y avanzando a más complejos.

    Progresión:
      1. Empty-5x5     — Solo ir a la meta (trivial)
      2. Empty-8x8     — Meta más lejos
      3. FourRooms     — Navegar entre habitaciones
      4. DoorKey       — Encontrar llave y abrir puerta
      5. LavaCrossing  — Evitar obstáculos de lava

    Cada nivel transfiere los pesos al siguiente entorno.
    El agente acumula habilidades: primero navega, luego gestiona
    objetos, finalmente evita peligros.

    Ejecutar: python minigrid_navegacion.py --curriculum
    """
    curriculum = [
        ("Empty",         "Solo llegar a la meta — sin obstáculos"),
        ("Empty8",        "Meta más lejos — planificación básica"),
        ("FourRooms",     "Navegar entre habitaciones"),
        ("DoorKey",       "Encontrar llave y abrir puerta"),
        ("LavaCrossing",  "Evitar lava — planificación con riesgo"),
    ]

    print(f"\n{'='*60}")
    print(f"  Variante C: Curriculum Learning MiniGrid")
    print(f"  {len(curriculum)} niveles de dificultad")
    print(f"  {timesteps_por_nivel:,} steps por nivel")
    print(f"  Algoritmo: {algorithm}")
    print(f"{'='*60}")

    for i, (env_name, desc) in enumerate(curriculum):
        print(f"  Nivel {i+1}: {env_name} — {desc}")
    print()

    model = None
    historial = {}

    for nivel, (env_name, desc) in enumerate(curriculum):
        print(f"\n--- NIVEL {nivel+1}: {env_name} ---")
        print(f"Descripción: {desc}")

        env = crear_entorno(env_name, render=False)

        if model is None:
            # Primer nivel: modelo desde cero con CNN
            policy_kwargs = {
                "features_extractor_class": MinigridCNN,
                "features_extractor_kwargs": {"features_dim": 64}
            }
            if algorithm == "PPO":
                model = PPO(
                    "CnnPolicy", env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    learning_rate=0.0003,
                    n_steps=128, batch_size=64,
                    n_epochs=4, gamma=0.99, ent_coef=0.01,
                )
            else:
                model = DQN(
                    "CnnPolicy", env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    learning_rate=0.0001,
                    buffer_size=50000, learning_starts=1000,
                    batch_size=32, gamma=0.99,
                )
        else:
            # Niveles siguientes: transferir pesos
            print(f"Transfiriendo pesos del nivel {nivel} al nuevo entorno...")
            model.set_env(env)

        callback = MinigridCallback()
        model.learn(
            total_timesteps=timesteps_por_nivel,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=(nivel == 0)
        )

        tasa_exito = np.mean(callback.successes) if callback.successes else 0
        historial[env_name] = {
            "rewards": callback.episode_rewards,
            "successes": callback.successes,
            "tasa_exito": tasa_exito
        }

        save_path = f"minigrid_curriculum_nivel{nivel+1}_{env_name.lower()}"
        model.save(save_path)
        print(f"Nivel {nivel+1} completado. Tasa de éxito: {tasa_exito*100:.1f}%")
        print(f"Modelo guardado: {save_path}.zip")

        env.close()

    # Graficar progreso de todos los niveles
    n_niveles = len(curriculum)
    fig, axes = plt.subplots(2, n_niveles, figsize=(4*n_niveles, 8))

    for idx, (env_name, desc) in enumerate(curriculum):
        data = historial.get(env_name, {})
        rewards = data.get("rewards", [])
        successes = data.get("successes", [])

        # Recompensas
        ax_r = axes[0][idx]
        if rewards:
            ax_r.plot(rewards, alpha=0.3, color="blue")
            window = min(20, len(rewards) // 4) if len(rewards) > 4 else 1
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
                ax_r.plot(range(window-1, len(rewards)), smoothed, "r-", linewidth=2)
        ax_r.set_title(f"Nivel {idx+1}: {env_name}")
        ax_r.set_ylabel("Recompensa" if idx == 0 else "")
        ax_r.grid(True, alpha=0.3)

        # Tasa de éxito
        ax_s = axes[1][idx]
        if successes:
            cumsum = np.cumsum(successes)
            rate = cumsum / (np.arange(len(successes)) + 1)
            ax_s.plot(rate, "g-", linewidth=2)
            ax_s.set_ylim(0, 1)
        ax_s.set_xlabel("Episodio")
        ax_s.set_ylabel("Tasa de Éxito" if idx == 0 else "")
        ax_s.grid(True, alpha=0.3)

    plt.suptitle(f"Curriculum Learning — MiniGrid ({algorithm})")
    plt.tight_layout()
    plt.savefig("minigrid_curriculum.png", dpi=150)
    plt.show()
    print("\nGráfica guardada: minigrid_curriculum.png")

    # Resumen
    print("\n" + "="*60)
    print("  RESUMEN CURRICULUM")
    print("="*60)
    for nivel, (env_name, _) in enumerate(curriculum):
        data = historial.get(env_name, {})
        print(f"  Nivel {nivel+1} ({env_name}): éxito = {data.get('tasa_exito', 0)*100:.1f}%")

    return model, historial


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGrid - Navegación con RL")
    parser.add_argument("--demo", action="store_true", help="Ver agente entrenado")
    parser.add_argument("--manual", action="store_true", help="Jugar manualmente")
    parser.add_argument("--curriculum", action="store_true",
                        help="Variante C: Curriculum learning progresivo")
    parser.add_argument("--variant", type=str, default="cnn",
                        choices=["flat", "cnn"],
                        help=(
                            "Variante de red:\n"
                            "  flat - (A) MLP con obs. aplanada (sin estructura espacial)\n"
                            "  cnn  - (B) CNN personalizada (aprovecha estructura espacial)"
                        ))
    parser.add_argument("--env", type=str, default="Empty",
                        choices=list(ENTORNOS.keys()), help="Entorno")
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "DQN"], help="Algoritmo")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Timesteps de entrenamiento")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodios para demo")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\nEntornos disponibles:")
    for name, env_id in ENTORNOS.items():
        print(f"  {name}: {env_id}")

    if args.manual:
        jugar_manual(args.env)
    elif args.demo:
        demo_agente(args.env, n_episodios=args.episodes)
    elif args.curriculum:
        curriculum_minigrid(
            timesteps_por_nivel=args.timesteps // 5,
            algorithm=args.algorithm
        )
    elif args.variant == "flat":
        print("Variante A: MLP + Observación Plana")
        model, callback = entrenar_flat(
            env_name=args.env,
            timesteps=args.timesteps,
            algorithm=args.algorithm
        )
        plot_training(callback)
        demo_agente(args.env, n_episodios=3)
    else:
        print("Variante B: CNN personalizada")
        model, callback = entrenar_minigrid(
            env_name=args.env,
            timesteps=args.timesteps,
            algorithm=args.algorithm
        )
        plot_training(callback)
        demo_agente(args.env, n_episodios=3)
