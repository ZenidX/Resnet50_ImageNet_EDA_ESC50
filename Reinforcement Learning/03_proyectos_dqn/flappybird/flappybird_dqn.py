"""
Flappy Bird con DQN/PPO
========================
Entrena un agente para jugar Flappy Bird.

VARIANTES:
  A — DQN + Simple  (--algorithm DQN --simple):  Off-policy, obs. 12D LIDAR
  B — PPO + Simple  (--algorithm PPO --simple):  On-policy, obs. 12D LIDAR
  C — DQN + RGB     (--algorithm DQN):           Off-policy, imagen completa
  D — Comparativa   (--compare-algorithms):      DQN vs PPO, misma obs.

  Comparativa completa (--compare-all): las 4 combinaciones A/B/C/D

Instalación:
    pip install flappy-bird-gymnasium stable-baselines3

Uso:
    python flappybird_dqn.py                          # Entrenar DQN+simple (var. A)
    python flappybird_dqn.py --algorithm PPO --simple # Entrenar PPO+simple (var. B)
    python flappybird_dqn.py --algorithm DQN          # Entrenar DQN+RGB (var. C)
    python flappybird_dqn.py --compare-algorithms     # Var. D: DQN vs PPO
    python flappybird_dqn.py --compare-all            # Las 4 combinaciones
    python flappybird_dqn.py --demo                   # Ver agente entrenado
    python flappybird_dqn.py --manual                 # Jugar manualmente
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import flappy_bird_gymnasium

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


# =============================================================================
# CALLBACK PERSONALIZADO
# =============================================================================

class FlappyCallback(BaseCallback):
    """Callback para registrar scores en Flappy Bird."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.scores = []
        self.episode_rewards = []
        self.best_score = 0

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                reward = info['episode']['r']
                self.episode_rewards.append(reward)

                # El score aproximado es la recompensa / 1 (cada tubo = +1)
                score = max(0, int(reward))
                self.scores.append(score)

                if score > self.best_score:
                    self.best_score = score
                    if self.verbose > 0:
                        print(f"  Nuevo récord: {score} tubos!")
        return True


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def entrenar_flappy(timesteps=500000, algorithm="DQN", use_simple_obs=False):
    """
    Entrena un agente en Flappy Bird.

    Args:
        timesteps: Pasos de entrenamiento
        algorithm: "DQN" o "PPO"
        use_simple_obs: Usar observación simplificada (12D) vs imagen

    Returns:
        Modelo entrenado y callback
    """
    print(f"\n{'='*60}")
    print(f"  Entrenando {algorithm} en Flappy Bird")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Observación: {'Simple (12D)' if use_simple_obs else 'RGB (288x512)'}")
    print(f"{'='*60}\n")

    # Crear entorno
    # use_lidar=True da observación simplificada de 12 valores
    env = gym.make(
        "FlappyBird-v0",
        render_mode=None,
        use_lidar=use_simple_obs
    )
    env = Monitor(env)

    # Configurar modelo según tipo de observación
    if use_simple_obs:
        # MLP para observación vectorial
        policy = "MlpPolicy"
        policy_kwargs = {"net_arch": [256, 256]}
    else:
        # CNN para imagen
        policy = "CnnPolicy"
        policy_kwargs = None

    if algorithm == "DQN":
        model = DQN(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0001,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            target_update_interval=1000,
            train_freq=4,
        )
    else:
        model = PPO(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
        )

    callback = FlappyCallback(verbose=1)

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True
    )

    # Guardar
    obs_type = "simple" if use_simple_obs else "rgb"
    save_path = f"flappy_{algorithm.lower()}_{obs_type}"
    model.save(save_path)
    print(f"\nModelo guardado: {save_path}.zip")
    print(f"Mejor score durante entrenamiento: {callback.best_score} tubos")

    env.close()
    return model, callback


def demo_flappy(model_path="flappy_dqn_simple", n_episodios=5, use_simple_obs=True):
    """
    Muestra el agente jugando Flappy Bird.
    """
    print(f"\nCargando modelo: {model_path}")

    # Detectar tipo de modelo
    if "dqn" in model_path.lower():
        model = DQN.load(model_path)
    else:
        model = PPO.load(model_path)

    env = gym.make(
        "FlappyBird-v0",
        render_mode="human",
        use_lidar=use_simple_obs
    )

    print(f"\nMostrando {n_episodios} partidas...")
    print("Acciones: 0=No hacer nada, 1=Saltar\n")

    scores = []
    for ep in range(n_episodios):
        obs, info = env.reset()
        score = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            score += max(0, reward)  # +1 por cada tubo
            done = terminated or truncated

        scores.append(int(score))
        print(f"Partida {ep+1}: {int(score)} tubos")

    print(f"\nPromedio: {np.mean(scores):.1f} tubos")
    print(f"Mejor: {max(scores)} tubos")

    env.close()


def comparar_observaciones(timesteps=100000):
    """
    Compara entrenamiento con observación simple vs RGB.
    """
    print("\n" + "="*60)
    print("  Comparación: Observación Simple vs RGB")
    print("="*60)

    resultados = {}

    # Simple (más rápido, pero menos información)
    print("\n--- Entrenando con observación SIMPLE (12D) ---")
    model_simple, cb_simple = entrenar_flappy(
        timesteps=timesteps,
        algorithm="DQN",
        use_simple_obs=True
    )
    resultados['Simple'] = {
        'scores': cb_simple.scores,
        'best': cb_simple.best_score
    }

    # RGB (más lento, pero ve todo)
    print("\n--- Entrenando con observación RGB ---")
    model_rgb, cb_rgb = entrenar_flappy(
        timesteps=timesteps,
        algorithm="DQN",
        use_simple_obs=False
    )
    resultados['RGB'] = {
        'scores': cb_rgb.scores,
        'best': cb_rgb.best_score
    }

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, data in resultados.items():
        scores = data['scores']
        if len(scores) > 10:
            smoothed = np.convolve(scores, np.ones(20)/20, mode='valid')
            ax.plot(smoothed, label=f"{name} (mejor: {data['best']})")

    ax.set_xlabel('Episodio')
    ax.set_ylabel('Score (tubos)')
    ax.set_title('Comparación de Observaciones en Flappy Bird')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("flappy_comparacion.png", dpi=150)
    plt.show()

    return resultados


def plot_training(callback, save_path="flappy_training.png"):
    """Grafica el progreso del entrenamiento."""
    if not callback.scores:
        print("No hay datos para graficar")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    scores = callback.scores
    window = min(100, len(scores) // 4) if len(scores) > 4 else 1

    # Scores por episodio
    axes[0].plot(scores, alpha=0.3, color='orange')
    if window > 1:
        smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(scores)), smoothed, 'r-', linewidth=2)
    axes[0].axhline(y=callback.best_score, color='green', linestyle='--',
                    label=f'Mejor: {callback.best_score}')
    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Tubos pasados')
    axes[0].set_title('Score por Episodio')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histograma de scores
    axes[1].hist(scores, bins=max(scores)+1 if scores else 10, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Tubos pasados')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Scores')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfica guardada: {save_path}")


def jugar_manual():
    """
    Permite jugar manualmente para entender el juego.
    """
    import pygame

    print("\n" + "="*60)
    print("  Flappy Bird - Modo Manual")
    print("="*60)
    print("\nControles:")
    print("  Espacio/Click: Saltar")
    print("  R: Reiniciar")
    print("  ESC: Salir\n")

    env = gym.make("FlappyBird-v0", render_mode="human")
    obs, info = env.reset()

    score = 0
    running = True
    clock = pygame.time.Clock()

    while running:
        action = 0  # No saltar por defecto

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    score = 0
                    print("Reiniciado")
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                action = 1

        obs, reward, terminated, truncated, info = env.step(action)
        score += max(0, reward)

        if terminated or truncated:
            print(f"Game Over! Score: {int(score)} tubos")
            obs, info = env.reset()
            score = 0

        clock.tick(30)

    env.close()


def comparar_algoritmos(timesteps=100000):
    """
    Variante D: Comparativa DQN vs PPO con misma observación.

    Compara los dos algoritmos principales usando la misma observación
    simple (12D LIDAR) para aislar el efecto del algoritmo.

    DQN (off-policy):
      - Usa replay buffer: aprende de experiencias pasadas
      - Más eficiente en datos (sample efficiency)
      - Puede ser inestable con Q-values sobreestimados

    PPO (on-policy):
      - Aprende solo de experiencias recientes
      - Más estable por el clipped surrogate objective
      - Menos eficiente en datos pero más robusto

    Ejecutar: python flappybird_dqn.py --compare-algorithms
    """
    print("\n" + "="*60)
    print("  Variante D: DQN vs PPO — misma observación simple")
    print("="*60)

    resultados = {}

    for algo in ["DQN", "PPO"]:
        print(f"\n--- Entrenando {algo} con observación simple ---")
        model, callback = entrenar_flappy(
            timesteps=timesteps,
            algorithm=algo,
            use_simple_obs=True
        )
        resultados[algo] = {
            "scores": callback.scores,
            "best": callback.best_score,
            "rewards": callback.episode_rewards
        }

    # Graficar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scores
    ax = axes[0]
    colors = {"DQN": "blue", "PPO": "orange"}
    for algo, data in resultados.items():
        scores = data["scores"]
        if len(scores) > 20:
            smoothed = np.convolve(scores, np.ones(20)/20, mode="valid")
            ax.plot(smoothed, color=colors[algo], label=f"{algo} (mejor: {data['best']})")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Score (tubos)")
    ax.set_title("DQN vs PPO — Scores en Flappy Bird")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recompensa acumulada
    ax2 = axes[1]
    for algo, data in resultados.items():
        rewards = data["rewards"]
        if len(rewards) > 20:
            smoothed = np.convolve(rewards, np.ones(20)/20, mode="valid")
            ax2.plot(smoothed, color=colors[algo], label=algo)
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Recompensa")
    ax2.set_title("Recompensa por episodio (suavizada)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Comparativa DQN vs PPO ({timesteps:,} steps)", fontsize=13)
    plt.tight_layout()
    plt.savefig("flappy_comparacion_algoritmos.png", dpi=150)
    plt.show()
    print("Gráfica guardada: flappy_comparacion_algoritmos.png")

    # Resumen
    print("\n" + "="*60)
    print("  RESULTADOS")
    print("="*60)
    for algo, data in resultados.items():
        scores = data["scores"]
        if scores:
            print(f"  {algo}: mejor={data['best']} tubos | media últimos 50ep={np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores):.1f}")

    print("\n  Interpretación:")
    print("  DQN: más sample-efficient (aprende con menos pasos)")
    print("  PPO: más estable, menos varianza en las curvas")

    return resultados


def comparar_todo(timesteps=80000):
    """
    Comparativa completa: DQN/PPO × Simple/RGB.

    Entrena las 4 combinaciones posibles:
      A: DQN + Simple  (más rápido de entrenar)
      B: PPO + Simple  (más estable)
      C: DQN + RGB     (más información, pero necesita CNN)
      D: PPO + RGB     (más lento, más información)

    Permite ver el impacto independiente de: algoritmo y observación.

    Ejecutar: python flappybird_dqn.py --compare-all
    """
    print("\n" + "="*60)
    print("  Comparativa COMPLETA: DQN/PPO × Simple/RGB")
    print("="*60)

    combinaciones = [
        ("DQN", True,  "A: DQN + Simple (obs. 12D)"),
        ("PPO", True,  "B: PPO + Simple (obs. 12D)"),
        ("DQN", False, "C: DQN + RGB    (imagen)"),
        ("PPO", False, "D: PPO + RGB    (imagen)"),
    ]

    resultados = {}
    for algo, simple, label in combinaciones:
        print(f"\n--- {label} ---")
        try:
            _, callback = entrenar_flappy(
                timesteps=timesteps,
                algorithm=algo,
                use_simple_obs=simple
            )
            resultados[label] = {
                "scores": callback.scores,
                "best": callback.best_score
            }
        except Exception as e:
            print(f"Error en {label}: {e}")
            resultados[label] = {"scores": [], "best": 0}

    # Graficar
    fig, ax = plt.subplots(figsize=(12, 6))
    colores = {
        "A: DQN + Simple (obs. 12D)": "blue",
        "B: PPO + Simple (obs. 12D)": "orange",
        "C: DQN + RGB    (imagen)":   "green",
        "D: PPO + RGB    (imagen)":   "red"
    }
    for label, data in resultados.items():
        scores = data["scores"]
        if len(scores) > 20:
            smoothed = np.convolve(scores, np.ones(20)/20, mode="valid")
            ax.plot(smoothed, color=colores.get(label, "gray"),
                    label=f"{label} (mejor: {data['best']})")

    ax.set_xlabel("Episodio")
    ax.set_ylabel("Score (tubos pasados)")
    ax.set_title(f"Comparativa DQN/PPO × Simple/RGB — Flappy Bird ({timesteps:,} steps)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("flappy_comparacion_completa.png", dpi=150)
    plt.show()
    print("Gráfica guardada: flappy_comparacion_completa.png")

    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flappy Bird con RL")
    parser.add_argument("--demo", action="store_true", help="Ver agente entrenado")
    parser.add_argument("--manual", action="store_true", help="Jugar manualmente")
    parser.add_argument("--compare", action="store_true",
                        help="Comparar Simple vs RGB (mismo algoritmo)")
    parser.add_argument("--compare-algorithms", action="store_true",
                        help="Variante D: Comparar DQN vs PPO (misma observación simple)")
    parser.add_argument("--compare-all", action="store_true",
                        help="Comparar las 4 combinaciones: DQN/PPO × Simple/RGB")
    parser.add_argument("--simple", action="store_true",
                        help="Usar observación simple (recomendado)")
    parser.add_argument("--algorithm", type=str, default="DQN",
                        choices=["DQN", "PPO"], help="Algoritmo")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Timesteps de entrenamiento")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodios para demo")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.manual:
        jugar_manual()
    elif args.demo:
        obs_type = "simple" if args.simple else "rgb"
        model_path = f"flappy_{args.algorithm.lower()}_{obs_type}"
        demo_flappy(model_path, args.episodes, args.simple)
    elif args.compare:
        comparar_observaciones(args.timesteps)
    elif args.compare_algorithms:
        comparar_algoritmos(args.timesteps)
    elif args.compare_all:
        comparar_todo(args.timesteps)
    else:
        model, callback = entrenar_flappy(
            timesteps=args.timesteps,
            algorithm=args.algorithm,
            use_simple_obs=args.simple or True
        )
        plot_training(callback)
        demo_flappy(n_episodios=3, use_simple_obs=args.simple or True)
