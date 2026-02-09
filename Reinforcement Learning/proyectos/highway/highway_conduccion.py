"""
Highway-Env - Conducción Autónoma con RL
========================================
Simulación 2D de conducción: autopista, parking, intersecciones.

Instalación:
    pip install highway-env stable-baselines3

Uso:
    python highway_conduccion.py                    # Entrenar autopista
    python highway_conduccion.py --env parking      # Entrenar parking
    python highway_conduccion.py --demo             # Ver agente
    python highway_conduccion.py --manual           # Conducir manualmente
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import highway_env

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# =============================================================================
# ENTORNOS DISPONIBLES
# =============================================================================

ENTORNOS = {
    "highway": {
        "id": "highway-v0",
        "desc": "Autopista: adelantar coches, mantener velocidad",
        "obs_type": "Kinematics",
    },
    "highway-fast": {
        "id": "highway-fast-v0",
        "desc": "Autopista simplificada (más rápida)",
        "obs_type": "Kinematics",
    },
    "merge": {
        "id": "merge-v0",
        "desc": "Incorporarse a autopista",
        "obs_type": "Kinematics",
    },
    "roundabout": {
        "id": "roundabout-v0",
        "desc": "Rotonda",
        "obs_type": "Kinematics",
    },
    "parking": {
        "id": "parking-v0",
        "desc": "Aparcar en plaza",
        "obs_type": "KinematicsGoal",
    },
    "intersection": {
        "id": "intersection-v0",
        "desc": "Cruzar intersección",
        "obs_type": "Kinematics",
    },
    "racetrack": {
        "id": "racetrack-v0",
        "desc": "Circuito de carreras",
        "obs_type": "Kinematics",
    },
}


# =============================================================================
# CALLBACK
# =============================================================================

class HighwayCallback(BaseCallback):
    """Callback para registrar progreso en highway-env."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.crashes = []
        self.goals_reached = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                # Algunas métricas específicas
                self.crashes.append(info.get('crashed', False))
                self.goals_reached.append(info.get('is_success', False))
        return True


# =============================================================================
# CREAR ENTORNO
# =============================================================================

def crear_entorno(env_name="highway", render=False, config_override=None):
    """
    Crea un entorno de highway-env con configuración.

    Args:
        env_name: Nombre del entorno
        render: Si renderizar
        config_override: Dict con configuración extra

    Returns:
        Entorno configurado
    """
    env_info = ENTORNOS.get(env_name, ENTORNOS["highway"])
    env_id = env_info["id"]

    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    # Configuración base
    config = {
        "observation": {
            "type": env_info["obs_type"],
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "duration": 40,  # Pasos por episodio
        "collision_reward": -1,
        "high_speed_reward": 0.4,
        "right_lane_reward": 0.1,
    }

    # Configuración específica por entorno
    if env_name == "parking":
        config = {
            "observation": {
                "type": "KinematicsGoal",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False,
            },
            "action": {
                "type": "ContinuousAction",
            },
        }

    if config_override:
        config.update(config_override)

    env.unwrapped.configure(config)

    return env


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def entrenar_highway(env_name="highway", timesteps=100000, algorithm="DQN"):
    """
    Entrena un agente en highway-env.

    Args:
        env_name: Nombre del entorno
        timesteps: Pasos de entrenamiento
        algorithm: "DQN" o "PPO"

    Returns:
        Modelo y callback
    """
    print(f"\n{'='*60}")
    print(f"  Entrenando {algorithm} en {env_name}")
    env_info = ENTORNOS.get(env_name, ENTORNOS["highway"])
    print(f"  Descripción: {env_info['desc']}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'='*60}\n")

    env = crear_entorno(env_name, render=False)

    # Determinar tipo de política
    if env_name == "parking":
        # Parking usa acciones continuas, solo PPO/SAC
        algorithm = "PPO"
        print("Nota: Parking requiere acciones continuas, usando PPO")

    if algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=5e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.8,  # Horizonte corto para conducción
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            target_update_interval=50,
            policy_kwargs={"net_arch": [256, 256]},
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=5e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.8,
            gae_lambda=0.95,
            policy_kwargs={"net_arch": [256, 256]},
        )

    callback = HighwayCallback()

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True
    )

    # Guardar
    save_path = f"highway_{env_name}_{algorithm.lower()}"
    model.save(save_path)
    print(f"\nModelo guardado: {save_path}.zip")

    env.close()
    return model, callback


def demo_highway(env_name="highway", model_path=None, n_episodios=5):
    """
    Muestra el agente conduciendo.
    """
    if model_path is None:
        model_path = f"highway_{env_name}_dqn"

    print(f"\nCargando modelo: {model_path}")

    # Intentar cargar como DQN, si falla como PPO
    try:
        model = DQN.load(model_path)
    except:
        model = PPO.load(model_path)

    env = crear_entorno(env_name, render=True)

    print(f"\nMostrando {n_episodios} episodios...")
    print("\nAcciones disponibles:")
    print("  0: LANE_LEFT")
    print("  1: IDLE")
    print("  2: LANE_RIGHT")
    print("  3: FASTER")
    print("  4: SLOWER\n")

    stats = {"crashes": 0, "success": 0, "rewards": []}

    for ep in range(n_episodios):
        obs, info = env.reset()
        total_reward = 0
        done = False
        crashed = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            crashed = info.get('crashed', False)

        stats["rewards"].append(total_reward)
        if crashed:
            stats["crashes"] += 1
            resultado = "CHOQUE"
        elif info.get('is_success', False):
            stats["success"] += 1
            resultado = "EXITO"
        else:
            resultado = "TIEMPO"

        print(f"Episodio {ep+1}: {total_reward:.2f} puntos - {resultado}")

    print(f"\nResumen:")
    print(f"  Recompensa media: {np.mean(stats['rewards']):.2f}")
    print(f"  Choques: {stats['crashes']}/{n_episodios}")
    print(f"  Éxitos: {stats['success']}/{n_episodios}")

    env.close()


def conducir_manual(env_name="highway"):
    """
    Permite conducir manualmente.
    """
    import pygame

    print(f"\n{'='*60}")
    print(f"  Conducción Manual - {env_name}")
    print(f"{'='*60}")
    print("\nControles:")
    print("  Flechas Arriba/Abajo: Acelerar/Frenar")
    print("  Flechas Izq/Der: Cambiar carril")
    print("  R: Reiniciar")
    print("  ESC: Salir\n")

    env = crear_entorno(env_name, render=True)
    obs, info = env.reset()

    total_reward = 0
    running = True
    clock = pygame.time.Clock()

    # Mapeo de teclas a acciones
    # 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
    action = 1  # IDLE por defecto

    while running:
        keys = pygame.key.get_pressed()

        # Determinar acción basada en teclas presionadas
        if keys[pygame.K_LEFT]:
            action = 0  # LANE_LEFT
        elif keys[pygame.K_RIGHT]:
            action = 2  # LANE_RIGHT
        elif keys[pygame.K_UP]:
            action = 3  # FASTER
        elif keys[pygame.K_DOWN]:
            action = 4  # SLOWER
        else:
            action = 1  # IDLE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("Reiniciado")
                elif event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            crashed = info.get('crashed', False)
            resultado = "CHOQUE!" if crashed else "Fin del episodio"
            print(f"{resultado} - Recompensa: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(15)  # 15 FPS

    env.close()


def plot_training(callback, save_path="highway_training.png"):
    """Grafica el progreso."""
    if not callback.episode_rewards:
        print("No hay datos")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    rewards = callback.episode_rewards
    window = min(50, len(rewards) // 4) if len(rewards) > 4 else 1

    # Recompensas
    axes[0].plot(rewards, alpha=0.3, color='blue')
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), smoothed, 'r-', linewidth=2)
    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_title('Recompensa por Episodio')
    axes[0].grid(True, alpha=0.3)

    # Longitud
    lengths = callback.episode_lengths
    axes[1].plot(lengths, alpha=0.3, color='green')
    if window > 1:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(lengths)), smoothed, 'r-', linewidth=2)
    axes[1].set_xlabel('Episodio')
    axes[1].set_ylabel('Pasos')
    axes[1].set_title('Duración del Episodio')
    axes[1].grid(True, alpha=0.3)

    # Tasa de choques
    if callback.crashes:
        crashes = [1 if c else 0 for c in callback.crashes]
        crash_rate = 1 - np.cumsum(crashes) / (np.arange(len(crashes)) + 1)
        axes[2].plot(crash_rate, 'b-', linewidth=2)
        axes[2].set_xlabel('Episodio')
        axes[2].set_ylabel('Tasa de Supervivencia')
        axes[2].set_title('Supervivencia Acumulada')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfica guardada: {save_path}")


def entrenar_todos(timesteps=50000):
    """
    Entrena en todos los entornos disponibles.
    """
    print("\n" + "="*60)
    print("  Entrenando en TODOS los entornos")
    print("="*60)

    resultados = {}

    for env_name in ["highway", "merge", "intersection", "roundabout"]:
        print(f"\n>>> {env_name.upper()} <<<")
        try:
            model, callback = entrenar_highway(
                env_name=env_name,
                timesteps=timesteps,
                algorithm="DQN"
            )
            resultados[env_name] = {
                "mean_reward": np.mean(callback.episode_rewards[-50:]),
                "crash_rate": sum(callback.crashes) / len(callback.crashes) if callback.crashes else 0
            }
        except Exception as e:
            print(f"Error en {env_name}: {e}")
            resultados[env_name] = {"error": str(e)}

    # Resumen
    print("\n" + "="*60)
    print("  RESUMEN")
    print("="*60)
    for env_name, data in resultados.items():
        if "error" in data:
            print(f"{env_name}: ERROR - {data['error']}")
        else:
            print(f"{env_name}: reward={data['mean_reward']:.2f}, crashes={data['crash_rate']*100:.1f}%")

    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Highway-Env - Conducción con RL")
    parser.add_argument("--demo", action="store_true", help="Ver agente")
    parser.add_argument("--manual", action="store_true", help="Conducir")
    parser.add_argument("--all", action="store_true", help="Entrenar todos")
    parser.add_argument("--env", type=str, default="highway",
                        choices=list(ENTORNOS.keys()), help="Entorno")
    parser.add_argument("--algorithm", type=str, default="DQN",
                        choices=["DQN", "PPO"], help="Algoritmo")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Timesteps")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodios demo")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\nEntornos disponibles:")
    for name, info in ENTORNOS.items():
        print(f"  {name}: {info['desc']}")

    if args.manual:
        conducir_manual(args.env)
    elif args.demo:
        demo_highway(args.env, n_episodios=args.episodes)
    elif args.all:
        entrenar_todos(args.timesteps)
    else:
        model, callback = entrenar_highway(
            env_name=args.env,
            timesteps=args.timesteps,
            algorithm=args.algorithm
        )
        plot_training(callback)
        demo_highway(args.env, n_episodios=3)
