"""
Ejemplo Completo: Entrenar DQN en CartPole
==========================================
Este script muestra cómo entrenar un agente DQN para resolver CartPole-v1.

Uso:
    python ejemplo_cartpole_dqn.py

El agente aprende a equilibrar un poste sobre un carro.
Objetivo: Mantener el poste en pie durante 500 pasos.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from core.agentes import DQNAgent, entrenar_dqn
from core.utils import plot_learning_curve, evaluate_agent, MetricsTracker

# Configuración
ENV_NAME = "CartPole-v1"
N_EPISODIOS = 300
MAX_PASOS = 500

# Hiperparámetros del agente
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BUFFER_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 10
HIDDEN_SIZES = [128, 128]


def main():
    print("=" * 60)
    print("ENTRENAMIENTO DQN - CartPole-v1")
    print("=" * 60)

    # Crear entorno
    env = gym.make(ENV_NAME)

    print(f"\nEntorno: {ENV_NAME}")
    print(f"Espacio de observación: {env.observation_space}")
    print(f"Espacio de acciones: {env.action_space}")

    # Crear agente
    input_size = env.observation_space.shape[0]
    n_acciones = env.action_space.n

    agente = DQNAgent(
        input_size=input_size,
        n_acciones=n_acciones,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update=TARGET_UPDATE,
        hidden_sizes=HIDDEN_SIZES
    )

    print(f"\nAgente DQN creado:")
    print(f"  - Red: {HIDDEN_SIZES}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Gamma: {GAMMA}")
    print(f"  - Epsilon decay: {EPSILON_DECAY}")
    print(f"  - Device: {agente.device}")

    # Entrenar
    print(f"\n{'=' * 60}")
    print("ENTRENANDO...")
    print("=" * 60)

    recompensas, perdidas = entrenar_dqn(
        env, agente,
        n_episodios=N_EPISODIOS,
        max_pasos=MAX_PASOS,
        verbose=True
    )

    # Resultados
    print(f"\n{'=' * 60}")
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("=" * 60)

    print(f"Recompensa promedio (últimos 50 ep): {np.mean(recompensas[-50:]):.2f}")
    print(f"Recompensa máxima: {np.max(recompensas):.2f}")
    print(f"Epsilon final: {agente.epsilon:.4f}")

    # Evaluar sin exploración
    print(f"\n{'=' * 60}")
    print("EVALUACIÓN (sin exploración)")
    print("=" * 60)

    # Desactivar exploración
    agente.epsilon = 0
    stats = evaluate_agent(env, agente, n_episodes=20)

    print(f"Recompensa: {stats['recompensa_media']:.1f} ± {stats['recompensa_std']:.1f}")
    print(f"Rango: [{stats['recompensa_min']:.0f}, {stats['recompensa_max']:.0f}]")
    print(f"Pasos promedio: {stats['pasos_media']:.1f}")

    # Verificar si resolvió el entorno
    if stats['recompensa_media'] >= 475:
        print("\n🎉 ¡ENTORNO RESUELTO! (media >= 475)")
    else:
        print(f"\n⚠️ No resuelto aún. Objetivo: media >= 475")

    # Guardar modelo
    agente.guardar(str(Path(__file__).parent.parent / "modelos" / "cartpole_dqn.pth"))

    # Visualizar curvas de aprendizaje
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Curva de recompensas
    window = 20
    recompensas_suav = np.convolve(recompensas, np.ones(window)/window, mode='valid')

    axes[0].plot(recompensas, alpha=0.3, label='Por episodio')
    axes[0].plot(range(window-1, len(recompensas)), recompensas_suav,
                 'r-', linewidth=2, label=f'Media móvil ({window} ep)')
    axes[0].axhline(y=475, color='g', linestyle='--', label='Objetivo (475)')
    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_title('Curva de Aprendizaje')
    axes[0].legend()

    # Curva de pérdida
    axes[1].plot(perdidas, color='red', alpha=0.7)
    axes[1].set_xlabel('Episodio')
    axes[1].set_ylabel('Pérdida (Loss)')
    axes[1].set_title('Pérdida durante Entrenamiento')

    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent.parent / "modelos" / "cartpole_training.png"), dpi=150)
    plt.show()

    print("\n✅ Gráficos guardados en: cartpole_training.png")

    env.close()


if __name__ == "__main__":
    main()
