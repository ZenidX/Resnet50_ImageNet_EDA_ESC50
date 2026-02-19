"""
PyBullet - Robótica y Física 3D con RL
======================================
Simulación 3D de robots aprendiendo a caminar, nadar, saltar.

VARIANTES:
  A — PPO por robot       (--algorithm PPO):       On-policy, general
  B — SAC por robot       (--algorithm SAC):       Off-policy, máxima entropía
  C — TD3 por robot       (--algorithm TD3):       Off-policy, estable
  D — Matriz Algo×Robot  (--compare-matrix):       Comparativa cruzada

Instalación:
    pip install pybullet stable-baselines3

Uso:
    python pybullet_robotica.py                      # Entrenar Ant con PPO
    python pybullet_robotica.py --env hopper         # Entrenar Hopper
    python pybullet_robotica.py --algorithm SAC      # Usar SAC
    python pybullet_robotica.py --compare            # Comparar PPO vs SAC vs TD3 en un robot
    python pybullet_robotica.py --compare-robots     # Mismo algo en diferentes robots
    python pybullet_robotica.py --compare-matrix     # Var. D: Matriz Algoritmo × Robot
    python pybullet_robotica.py --demo               # Ver agente
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import gymnasium as gym

# Registrar entornos de PyBullet
try:
    import pybullet_envs
    PYBULLET_LEGACY = True
except ImportError:
    PYBULLET_LEGACY = False

# Intentar con gymnasium-robotics o shimmy
try:
    import shimmy
    SHIMMY_AVAILABLE = True
except ImportError:
    SHIMMY_AVAILABLE = False

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# =============================================================================
# ENTORNOS DISPONIBLES
# =============================================================================

ENTORNOS = {
    # Locomotion básica
    "ant": {
        "id": "AntBulletEnv-v0" if PYBULLET_LEGACY else "Ant-v4",
        "desc": "Hormiga de 4 patas - Aprender a caminar",
        "timesteps": 500000,
    },
    "halfcheetah": {
        "id": "HalfCheetahBulletEnv-v0" if PYBULLET_LEGACY else "HalfCheetah-v4",
        "desc": "Guepardo 2D - Correr lo más rápido posible",
        "timesteps": 500000,
    },
    "hopper": {
        "id": "HopperBulletEnv-v0" if PYBULLET_LEGACY else "Hopper-v4",
        "desc": "Saltador de una pierna",
        "timesteps": 300000,
    },
    "walker": {
        "id": "Walker2DBulletEnv-v0" if PYBULLET_LEGACY else "Walker2d-v4",
        "desc": "Bípedo 2D - Caminar",
        "timesteps": 500000,
    },
    "humanoid": {
        "id": "HumanoidBulletEnv-v0" if PYBULLET_LEGACY else "Humanoid-v4",
        "desc": "Humanoide 3D - El más difícil",
        "timesteps": 2000000,
    },
    # Otros
    "inverted_pendulum": {
        "id": "InvertedPendulumBulletEnv-v0" if PYBULLET_LEGACY else "InvertedPendulum-v4",
        "desc": "Péndulo invertido - Equilibrar",
        "timesteps": 100000,
    },
    "inverted_double": {
        "id": "InvertedDoublePendulumBulletEnv-v0" if PYBULLET_LEGACY else "InvertedDoublePendulum-v4",
        "desc": "Doble péndulo - Muy inestable",
        "timesteps": 200000,
    },
    "reacher": {
        "id": "ReacherBulletEnv-v0" if PYBULLET_LEGACY else "Reacher-v4",
        "desc": "Brazo robótico - Alcanzar objetivo",
        "timesteps": 200000,
    },
}


# =============================================================================
# CALLBACK
# =============================================================================

class RoboticsCallback(BaseCallback):
    """Callback para registrar métricas de robots."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                reward = info['episode']['r']
                self.episode_rewards.append(reward)
                self.episode_lengths.append(info['episode']['l'])

                if reward > self.best_reward:
                    self.best_reward = reward
                    if self.verbose > 0:
                        print(f"  Nuevo mejor: {reward:.1f}")
        return True


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def crear_entorno(env_name="ant", render=False):
    """
    Crea un entorno de PyBullet.
    """
    env_info = ENTORNOS.get(env_name, ENTORNOS["ant"])
    env_id = env_info["id"]

    render_mode = "human" if render else None

    try:
        env = gym.make(env_id, render_mode=render_mode)
    except:
        # Fallback si el entorno no existe
        print(f"Entorno {env_id} no disponible, usando InvertedPendulum")
        env = gym.make("InvertedPendulum-v4", render_mode=render_mode)

    return env


def entrenar_robot(env_name="ant", timesteps=None, algorithm="PPO"):
    """
    Entrena un agente en un entorno de robótica.

    Args:
        env_name: Nombre del robot
        timesteps: Pasos (None = usar recomendado)
        algorithm: "PPO", "SAC" o "TD3"

    Returns:
        Modelo y callback
    """
    env_info = ENTORNOS.get(env_name, ENTORNOS["ant"])

    if timesteps is None:
        timesteps = env_info["timesteps"]

    print(f"\n{'='*60}")
    print(f"  Entrenando {algorithm} en {env_name}")
    print(f"  Descripción: {env_info['desc']}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'='*60}\n")

    env = crear_entorno(env_name, render=False)

    # Elegir algoritmo
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [dict(pi=[256, 256], vf=[256, 256])]},
        )
    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs={"net_arch": [256, 256]},
        )
    else:  # TD3
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_kwargs={"net_arch": [256, 256]},
        )

    callback = RoboticsCallback(verbose=1)

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True
    )

    # Guardar
    save_path = f"robot_{env_name}_{algorithm.lower()}"
    model.save(save_path)
    print(f"\nModelo guardado: {save_path}.zip")
    print(f"Mejor recompensa: {callback.best_reward:.1f}")

    env.close()
    return model, callback


def demo_robot(env_name="ant", model_path=None, n_episodios=3, slow_motion=False):
    """
    Muestra el robot entrenado.

    Args:
        env_name: Nombre del robot
        model_path: Ruta del modelo
        n_episodios: Episodios a mostrar
        slow_motion: Si reducir velocidad
    """
    if model_path is None:
        model_path = f"robot_{env_name}_ppo"

    print(f"\nCargando modelo: {model_path}")

    # Intentar cargar con diferentes algoritmos
    for ModelClass in [PPO, SAC, TD3]:
        try:
            model = ModelClass.load(model_path)
            break
        except:
            continue
    else:
        print("No se pudo cargar el modelo")
        return

    env = crear_entorno(env_name, render=True)

    print(f"\nMostrando {n_episodios} episodios de {env_name}...")
    if slow_motion:
        print("Modo cámara lenta activado")

    rewards = []
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

            if slow_motion:
                time.sleep(0.02)

        rewards.append(total_reward)
        print(f"Episodio {ep+1}: {total_reward:.1f} puntos ({steps} pasos)")

    print(f"\nPromedio: {np.mean(rewards):.1f}")
    print(f"Mejor: {max(rewards):.1f}")

    env.close()


def comparar_algoritmos(env_name="ant", timesteps=100000):
    """
    Compara PPO, SAC y TD3 en un robot.
    """
    print("\n" + "="*60)
    print(f"  Comparación de Algoritmos en {env_name}")
    print("="*60)

    resultados = {}

    for algo in ["PPO", "SAC", "TD3"]:
        print(f"\n--- {algo} ---")
        try:
            model, callback = entrenar_robot(
                env_name=env_name,
                timesteps=timesteps,
                algorithm=algo
            )
            resultados[algo] = {
                "rewards": callback.episode_rewards,
                "best": callback.best_reward,
                "mean_last": np.mean(callback.episode_rewards[-20:]) if len(callback.episode_rewards) >= 20 else np.mean(callback.episode_rewards)
            }
        except Exception as e:
            print(f"Error con {algo}: {e}")
            resultados[algo] = {"error": str(e)}

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 5))

    for algo, data in resultados.items():
        if "error" not in data and data["rewards"]:
            rewards = data["rewards"]
            window = min(20, len(rewards) // 4) if len(rewards) > 4 else 1
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=f"{algo} (best: {data['best']:.0f})")

    ax.set_xlabel('Episodio')
    ax.set_ylabel('Recompensa')
    ax.set_title(f'Comparación de Algoritmos - {env_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"robot_{env_name}_comparacion.png", dpi=150)
    plt.show()

    # Tabla
    print("\n" + "="*60)
    print("  RESULTADOS")
    print("="*60)
    for algo, data in resultados.items():
        if "error" in data:
            print(f"{algo}: ERROR")
        else:
            print(f"{algo}: mejor={data['best']:.1f}, media={data['mean_last']:.1f}")

    return resultados


def comparar_robots(timesteps=100000, algorithm="PPO"):
    """
    Entrena varios robots y compara rendimiento.
    """
    print("\n" + "="*60)
    print("  Comparación de Robots")
    print("="*60)

    robots = ["inverted_pendulum", "hopper", "halfcheetah", "ant"]
    resultados = {}

    for robot in robots:
        print(f"\n>>> {robot.upper()} <<<")
        try:
            model, callback = entrenar_robot(
                env_name=robot,
                timesteps=timesteps,
                algorithm=algorithm
            )
            resultados[robot] = {
                "rewards": callback.episode_rewards,
                "best": callback.best_reward
            }
        except Exception as e:
            print(f"Error: {e}")
            resultados[robot] = {"error": str(e)}

    # Graficar
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (robot, data) in enumerate(resultados.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        if "error" not in data and data["rewards"]:
            rewards = data["rewards"]
            ax.plot(rewards, alpha=0.3)
            window = min(20, len(rewards) // 4) if len(rewards) > 4 else 1
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(rewards)), smoothed, 'r-', linewidth=2)
        ax.set_title(f"{robot} (best: {data.get('best', 'N/A')})")
        ax.set_xlabel('Episodio')
        ax.set_ylabel('Recompensa')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("robot_comparacion_todos.png", dpi=150)
    plt.show()

    return resultados


def plot_training(callback, env_name, save_path=None):
    """Grafica el entrenamiento."""
    if not callback.episode_rewards:
        print("No hay datos")
        return

    if save_path is None:
        save_path = f"robot_{env_name}_training.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    rewards = callback.episode_rewards
    window = min(50, len(rewards) // 4) if len(rewards) > 4 else 1

    # Recompensas
    axes[0].plot(rewards, alpha=0.3, color='blue')
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), smoothed, 'r-', linewidth=2)
    axes[0].axhline(y=callback.best_reward, color='green', linestyle='--',
                    label=f'Mejor: {callback.best_reward:.0f}')
    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_title(f'Aprendizaje - {env_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Longitud
    lengths = callback.episode_lengths
    axes[1].plot(lengths, alpha=0.3, color='green')
    if window > 1:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(lengths)), smoothed, 'r-', linewidth=2)
    axes[1].set_xlabel('Episodio')
    axes[1].set_ylabel('Pasos')
    axes[1].set_title('Duración de Episodio')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfica guardada: {save_path}")


def comparar_algoritmos_por_robot(timesteps=50000):
    """
    Variante D: Matriz de comparación Algoritmo × Robot.

    Entrena cada combinación de algoritmo y robot, generando
    una tabla comparativa de qué algoritmo funciona mejor
    para cada tipo de robot.

    Robots evaluados (de menor a mayor complejidad):
      - inverted_pendulum: péndulo invertido (equilibrio simple)
      - hopper:            saltador de una pierna
      - halfcheetah:       guepardo 2D (correr rápido)
      - ant:               hormiga 4 patas (locomoción compleja)

    Algoritmos:
      - PPO: on-policy, general, más simple
      - SAC: off-policy, máxima entropía, muy explorador
      - TD3: off-policy, determinista, estable

    Insight esperado: SAC y TD3 suelen superar a PPO en entornos
    complejos de control continuo, pero PPO converge más rápido
    en entornos simples.

    Ejecutar: python pybullet_robotica.py --compare-matrix
    """
    robots = ["inverted_pendulum", "hopper", "halfcheetah", "ant"]
    algoritmos = ["PPO", "SAC", "TD3"]

    print(f"\n{'='*70}")
    print(f"  Variante D: Matriz Algoritmo × Robot")
    print(f"  {len(algoritmos)} algoritmos × {len(robots)} robots = {len(algoritmos)*len(robots)} combinaciones")
    print(f"  {timesteps:,} steps por combinación")
    print(f"{'='*70}\n")

    # Tabla de resultados: resultados[robot][algoritmo] = {mean, best}
    resultados = {robot: {} for robot in robots}

    for robot in robots:
        env_info = ENTORNOS.get(robot, ENTORNOS["ant"])
        print(f"\n{'='*50}")
        print(f"  ROBOT: {robot.upper()} — {env_info['desc']}")
        print(f"{'='*50}")

        for algo in algoritmos:
            print(f"\n  --- {algo} en {robot} ---")
            try:
                model, callback = entrenar_robot(
                    env_name=robot,
                    timesteps=timesteps,
                    algorithm=algo
                )
                mean_last = np.mean(callback.episode_rewards[-20:]) if len(callback.episode_rewards) >= 20 else (np.mean(callback.episode_rewards) if callback.episode_rewards else -999)
                resultados[robot][algo] = {
                    "rewards": callback.episode_rewards,
                    "best": callback.best_reward,
                    "mean_last": mean_last
                }
                print(f"  {algo} en {robot}: mejor={callback.best_reward:.1f}, media_final={mean_last:.1f}")
            except Exception as e:
                print(f"  Error {algo} en {robot}: {e}")
                resultados[robot][algo] = {"rewards": [], "best": -999, "mean_last": -999, "error": str(e)}

    # Graficar: grid de subplots (robots x algoritmos)
    fig, axes = plt.subplots(len(robots), len(algoritmos),
                             figsize=(5*len(algoritmos), 4*len(robots)))

    colores = {"PPO": "blue", "SAC": "orange", "TD3": "green"}

    for r_idx, robot in enumerate(robots):
        for a_idx, algo in enumerate(algoritmos):
            ax = axes[r_idx][a_idx]
            data = resultados[robot].get(algo, {})
            rewards = data.get("rewards", [])

            if rewards and "error" not in data:
                ax.plot(rewards, alpha=0.3, color=colores[algo])
                window = min(20, len(rewards) // 4) if len(rewards) > 4 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
                    ax.plot(range(window-1, len(rewards)), smoothed,
                            color=colores[algo], linewidth=2)
                best = data.get("best", 0)
                ax.axhline(y=best, color="red", linestyle="--", alpha=0.5,
                           label=f"best={best:.0f}")
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "Error / No disponible",
                        ha="center", va="center", transform=ax.transAxes)

            if r_idx == 0:
                ax.set_title(algo, fontsize=12, fontweight="bold")
            if a_idx == 0:
                ax.set_ylabel(f"{robot}\nRecompensa", fontsize=9)
            if r_idx == len(robots) - 1:
                ax.set_xlabel("Episodio")
            ax.grid(True, alpha=0.3)

    plt.suptitle(f"Comparación Algoritmo × Robot ({timesteps:,} steps)", fontsize=14)
    plt.tight_layout()
    plt.savefig("robot_matriz_comparacion.png", dpi=150)
    plt.show()
    print("\nGráfica guardada: robot_matriz_comparacion.png")

    # Tabla resumen
    print("\n" + "="*70)
    print("  TABLA RESUMEN: Recompensa Media Final")
    print("="*70)
    header = f"{'Robot':<20}" + "".join(f"{algo:>12}" for algo in algoritmos)
    print(header)
    print("-"*70)
    for robot in robots:
        row = f"{robot:<20}"
        for algo in algoritmos:
            data = resultados[robot].get(algo, {})
            if "error" in data:
                row += f"{'ERROR':>12}"
            else:
                row += f"{data.get('mean_last', -999):>12.1f}"
        print(row)

    print("\n  Interpretación:")
    print("  PPO: general, converge rápido en entornos simples")
    print("  SAC: explorador, mejor en espacios de estado complejos")
    print("  TD3: estable, reduce sobreestimación de Q-values")

    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyBullet - Robótica con RL")
    parser.add_argument("--demo", action="store_true", help="Ver robot entrenado")
    parser.add_argument("--compare", action="store_true", help="Comparar algoritmos en un robot")
    parser.add_argument("--compare-robots", action="store_true", help="Comparar robots con un algoritmo")
    parser.add_argument("--compare-matrix", action="store_true",
                        help="Variante D: Matriz comparativa Algoritmo × Robot")
    parser.add_argument("--slow", action="store_true", help="Cámara lenta en demo")
    parser.add_argument("--env", type=str, default="ant",
                        choices=list(ENTORNOS.keys()), help="Robot")
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "SAC", "TD3"], help="Algoritmo")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Timesteps (None=recomendado)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodios demo")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\nRobots disponibles:")
    for name, info in ENTORNOS.items():
        print(f"  {name}: {info['desc']} ({info['timesteps']:,} steps rec.)")

    if not PYBULLET_LEGACY:
        print("\nNota: Usando entornos de gymnasium (MuJoCo backend)")
        print("Para PyBullet puro: pip install pybullet gym==0.21")

    if args.demo:
        demo_robot(args.env, n_episodios=args.episodes, slow_motion=args.slow)
    elif args.compare:
        comparar_algoritmos(args.env, args.timesteps or 100000)
    elif args.compare_robots:
        comparar_robots(args.timesteps or 100000, args.algorithm)
    elif args.compare_matrix:
        comparar_algoritmos_por_robot(args.timesteps or 50000)
    else:
        model, callback = entrenar_robot(
            env_name=args.env,
            timesteps=args.timesteps,
            algorithm=args.algorithm
        )
        plot_training(callback, args.env)
        demo_robot(args.env, n_episodios=2, slow_motion=True)
