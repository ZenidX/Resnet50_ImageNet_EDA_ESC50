"""
LunarLander con Stable-Baselines3
=================================
Ejemplo completo de entrenamiento y visualización.

Instalación:
    pip install stable-baselines3 gymnasium[box2d]

Uso:
    python lunarlander_sb3.py              # Entrenar desde cero
    python lunarlander_sb3.py --demo       # Ver agente entrenado
    python lunarlander_sb3.py --episodes 200  # Entrenar más episodios
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# Stable-Baselines3
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym


class RewardLoggerCallback(BaseCallback):
    """Callback para registrar recompensas durante el entrenamiento."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        # Registrar recompensas de episodios completados
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if info not in self.current_rewards:
                    self.episode_rewards.append(info['r'])
                    self.episode_lengths.append(info['l'])
                    self.current_rewards.append(info)
        return True


def entrenar_lunarlander(algoritmo="PPO", timesteps=100000, save_path="lunarlander_model"):
    """
    Entrena un agente en LunarLander-v3.

    Args:
        algoritmo: "PPO", "DQN" o "A2C"
        timesteps: Número total de pasos de entrenamiento
        save_path: Ruta para guardar el modelo

    Returns:
        Modelo entrenado y callback con historial
    """
    print(f"\n{'='*60}")
    print(f"  Entrenando {algoritmo} en LunarLander-v3")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'='*60}\n")

    # Crear entorno con monitor
    env = gym.make("LunarLander-v3")
    env = Monitor(env)

    # Seleccionar algoritmo
    if algoritmo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./logs/"
        )
    elif algoritmo == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0001,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            tensorboard_log="./logs/"
        )
    elif algoritmo == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            tensorboard_log="./logs/"
        )
    else:
        raise ValueError(f"Algoritmo no soportado: {algoritmo}")

    # Callback para registrar progreso
    reward_callback = RewardLoggerCallback()

    # Entrenar
    model.learn(
        total_timesteps=timesteps,
        callback=reward_callback,
        progress_bar=True
    )

    # Guardar modelo
    model.save(save_path)
    print(f"\nModelo guardado en: {save_path}.zip")

    env.close()
    return model, reward_callback


def evaluar_modelo(model, n_episodios=10, render=False):
    """
    Evalúa un modelo entrenado.

    Args:
        model: Modelo de Stable-Baselines3
        n_episodios: Número de episodios para evaluar
        render: Si mostrar visualización

    Returns:
        Recompensa media y desviación estándar
    """
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_episodios, deterministic=True
    )

    print(f"\nEvaluación ({n_episodios} episodios):")
    print(f"  Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Criterio de éxito: >200 es resolver el entorno
    if mean_reward > 200:
        print("  Estado: RESUELTO (>200 puntos)")
    elif mean_reward > 0:
        print("  Estado: Aprendiendo...")
    else:
        print("  Estado: Necesita más entrenamiento")

    env.close()
    return mean_reward, std_reward


def demo_agente(model_path="lunarlander_model", n_episodios=5):
    """
    Muestra el agente entrenado jugando.

    Args:
        model_path: Ruta del modelo guardado
        n_episodios: Número de episodios a mostrar
    """
    print(f"\nCargando modelo desde: {model_path}")

    # Detectar algoritmo por el archivo (o usar PPO por defecto)
    model = PPO.load(model_path)

    env = gym.make("LunarLander-v3", render_mode="human")

    print(f"\nMostrando {n_episodios} episodios...")
    print("Controles: Cierra la ventana para terminar\n")

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

        resultado = "ATERRIZADO" if total_reward > 200 else "CRASH" if total_reward < 0 else "OK"
        print(f"Episodio {ep+1}: {total_reward:.1f} puntos ({steps} pasos) - {resultado}")

    env.close()


def plot_training(callback, save_path="training_curve.png"):
    """Grafica las curvas de entrenamiento."""
    if not callback.episode_rewards:
        print("No hay datos de entrenamiento para graficar")
        return

    rewards = callback.episode_rewards

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Recompensas por episodio
    axes[0].plot(rewards, alpha=0.3, color='blue', label='Por episodio')

    # Media móvil
    window = min(50, len(rewards) // 4) if len(rewards) > 4 else 1
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), moving_avg, color='red',
                     linewidth=2, label=f'Media móvil ({window} eps)')

    axes[0].axhline(y=200, color='green', linestyle='--', label='Meta (200)')
    axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_title('Curva de Aprendizaje - LunarLander')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histograma de recompensas
    axes[1].hist(rewards, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=200, color='green', linestyle='--', label='Meta (200)')
    axes[1].axvline(x=np.mean(rewards), color='red', linestyle='-',
                    label=f'Media: {np.mean(rewards):.1f}')
    axes[1].set_xlabel('Recompensa')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Recompensas')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfica guardada en: {save_path}")


def comparar_algoritmos(timesteps=50000):
    """
    Compara PPO, DQN y A2C en LunarLander.
    """
    print("\n" + "="*60)
    print("  Comparación de Algoritmos en LunarLander")
    print("="*60)

    resultados = {}

    for algo in ["PPO", "A2C", "DQN"]:
        print(f"\n--- Entrenando {algo} ---")
        model, callback = entrenar_lunarlander(
            algoritmo=algo,
            timesteps=timesteps,
            save_path=f"lunarlander_{algo.lower()}"
        )

        mean_reward, std_reward = evaluar_modelo(model, n_episodios=10)
        resultados[algo] = {
            'mean': mean_reward,
            'std': std_reward,
            'rewards': callback.episode_rewards
        }

    # Graficar comparación
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, data in resultados.items():
        rewards = data['rewards']
        window = min(20, len(rewards) // 4) if len(rewards) > 4 else 1
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=f"{algo} (final: {data['mean']:.1f})")

    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Meta')
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Recompensa (suavizada)')
    ax.set_title(f'Comparación de Algoritmos ({timesteps:,} timesteps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparacion_algoritmos.png", dpi=150)
    plt.show()

    # Tabla de resultados
    print("\n" + "="*60)
    print("  RESULTADOS FINALES")
    print("="*60)
    print(f"{'Algoritmo':<10} {'Recompensa Media':>18} {'Desv. Est.':>12}")
    print("-"*42)
    for algo, data in resultados.items():
        print(f"{algo:<10} {data['mean']:>18.2f} {data['std']:>12.2f}")

    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LunarLander con Stable-Baselines3")
    parser.add_argument("--demo", action="store_true", help="Ver agente entrenado")
    parser.add_argument("--compare", action="store_true", help="Comparar PPO, DQN, A2C")
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "DQN", "A2C"], help="Algoritmo a usar")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Timesteps de entrenamiento")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodios para demo")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.demo:
        demo_agente(n_episodios=args.episodes)
    elif args.compare:
        comparar_algoritmos(timesteps=args.timesteps)
    else:
        model, callback = entrenar_lunarlander(
            algoritmo=args.algorithm,
            timesteps=args.timesteps
        )
        plot_training(callback)
        evaluar_modelo(model, n_episodios=10, render=True)
