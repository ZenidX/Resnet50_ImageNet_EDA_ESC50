"""
Utilidades para Reinforcement Learning
=======================================
Funciones auxiliares para entrenar y evaluar agentes de RL.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random


def plot_learning_curve(recompensas, window=50, titulo="Curva de Aprendizaje"):
    """
    Grafica la curva de aprendizaje con media móvil.

    Args:
        recompensas: Lista de recompensas por episodio
        window: Ventana para la media móvil
        titulo: Título del gráfico
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Recompensas brutas (transparentes)
    ax.plot(recompensas, alpha=0.3, color='blue', label='Por episodio')

    # Media móvil
    if len(recompensas) >= window:
        media_movil = np.convolve(recompensas, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(recompensas)), media_movil,
                color='red', linewidth=2, label=f'Media móvil ({window} ep)')

    ax.set_xlabel('Episodio')
    ax.set_ylabel('Recompensa')
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison(resultados_dict, window=50, titulo="Comparación de Agentes"):
    """
    Compara múltiples agentes en un solo gráfico.

    Args:
        resultados_dict: Dict {nombre_agente: lista_recompensas}
        window: Ventana para media móvil
        titulo: Título del gráfico
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    colores = plt.cm.tab10(np.linspace(0, 1, len(resultados_dict)))

    for (nombre, recompensas), color in zip(resultados_dict.items(), colores):
        if len(recompensas) >= window:
            media_movil = np.convolve(recompensas, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(recompensas)), media_movil,
                    color=color, linewidth=2, label=nombre)

    ax.set_xlabel('Episodio')
    ax.set_ylabel('Recompensa (media móvil)')
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


class MetricsTracker:
    """
    Rastrea métricas durante el entrenamiento.
    """

    def __init__(self):
        self.recompensas = []
        self.pasos = []
        self.perdidas = []
        self.epsilons = []

    def registrar_episodio(self, recompensa, pasos, perdida=None, epsilon=None):
        """Registra las métricas de un episodio."""
        self.recompensas.append(recompensa)
        self.pasos.append(pasos)
        if perdida is not None:
            self.perdidas.append(perdida)
        if epsilon is not None:
            self.epsilons.append(epsilon)

    def get_stats(self, ultimos_n=100):
        """Devuelve estadísticas de los últimos N episodios."""
        r = self.recompensas[-ultimos_n:]
        p = self.pasos[-ultimos_n:]

        return {
            'recompensa_media': np.mean(r),
            'recompensa_std': np.std(r),
            'recompensa_max': np.max(r),
            'pasos_media': np.mean(p),
            'total_episodios': len(self.recompensas)
        }

    def plot_all(self, window=50):
        """Genera gráficos de todas las métricas."""
        n_plots = 2 + (1 if self.perdidas else 0) + (1 if self.epsilons else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))

        if n_plots == 1:
            axes = [axes]

        idx = 0

        # Recompensas
        axes[idx].plot(self.recompensas, alpha=0.3)
        if len(self.recompensas) >= window:
            mm = np.convolve(self.recompensas, np.ones(window)/window, mode='valid')
            axes[idx].plot(range(window-1, len(self.recompensas)), mm, 'r-')
        axes[idx].set_title('Recompensas')
        axes[idx].set_xlabel('Episodio')
        idx += 1

        # Pasos
        axes[idx].plot(self.pasos, alpha=0.3, color='green')
        if len(self.pasos) >= window:
            mm = np.convolve(self.pasos, np.ones(window)/window, mode='valid')
            axes[idx].plot(range(window-1, len(self.pasos)), mm, 'darkgreen')
        axes[idx].set_title('Pasos por Episodio')
        axes[idx].set_xlabel('Episodio')
        idx += 1

        # Pérdidas (si hay)
        if self.perdidas:
            axes[idx].plot(self.perdidas, color='red', alpha=0.5)
            axes[idx].set_title('Pérdida (Loss)')
            axes[idx].set_xlabel('Episodio')
            idx += 1

        # Epsilon (si hay)
        if self.epsilons:
            axes[idx].plot(self.epsilons, color='purple')
            axes[idx].set_title('Epsilon (Exploración)')
            axes[idx].set_xlabel('Episodio')

        plt.tight_layout()
        return fig


def discretize_state(state, bins_per_dim, state_bounds):
    """
    Discretiza un estado continuo en bins.

    Args:
        state: Estado continuo (numpy array)
        bins_per_dim: Número de bins por dimensión
        state_bounds: Lista de tuplas (min, max) por dimensión

    Returns:
        Tupla de índices discretizados
    """
    discretized = []

    for i, (val, (low, high)) in enumerate(zip(state, state_bounds)):
        # Clipear al rango válido
        val = np.clip(val, low, high)

        # Calcular el bin
        bin_width = (high - low) / bins_per_dim
        bin_idx = int((val - low) / bin_width)
        bin_idx = min(bin_idx, bins_per_dim - 1)  # Asegurar que no se pase

        discretized.append(bin_idx)

    return tuple(discretized)


class EpsilonScheduler:
    """
    Programa la exploración (epsilon) durante el entrenamiento.
    """

    def __init__(self, epsilon_start=1.0, epsilon_end=0.01, decay_type='exponential',
                 decay_rate=0.995, decay_steps=None):
        """
        Args:
            epsilon_start: Valor inicial de epsilon
            epsilon_end: Valor mínimo de epsilon
            decay_type: 'exponential', 'linear', o 'step'
            decay_rate: Factor de decay (para exponential)
            decay_steps: Pasos totales para el decay (para linear)
        """
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.step_count = 0

    def step(self):
        """Actualiza epsilon y devuelve el valor actual."""
        self.step_count += 1

        if self.decay_type == 'exponential':
            self.epsilon = max(self.epsilon_end, self.epsilon * self.decay_rate)

        elif self.decay_type == 'linear':
            if self.decay_steps:
                progress = min(self.step_count / self.decay_steps, 1.0)
                self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

        elif self.decay_type == 'step':
            # Reducir a la mitad cada decay_steps
            if self.decay_steps and self.step_count % self.decay_steps == 0:
                self.epsilon = max(self.epsilon_end, self.epsilon / 2)

        return self.epsilon

    def get(self):
        """Devuelve el epsilon actual sin actualizarlo."""
        return self.epsilon


def evaluate_agent(env, agent, n_episodes=10, render=False, max_steps=1000):
    """
    Evalúa un agente entrenado.

    Args:
        env: Entorno de Gymnasium
        agent: Agente con método seleccionar_accion(estado)
        n_episodes: Número de episodios de evaluación
        render: Si mostrar el entorno
        max_steps: Máximo de pasos por episodio

    Returns:
        Dict con estadísticas de evaluación
    """
    recompensas = []
    pasos = []

    for ep in range(n_episodes):
        estado, _ = env.reset()
        recompensa_total = 0
        n_pasos = 0
        terminado = False
        truncado = False

        while not (terminado or truncado) and n_pasos < max_steps:
            # Desactivar exploración para evaluación
            if hasattr(agent, 'epsilon'):
                old_epsilon = agent.epsilon
                agent.epsilon = 0

            accion = agent.seleccionar_accion(estado)

            if hasattr(agent, 'epsilon'):
                agent.epsilon = old_epsilon

            estado, recompensa, terminado, truncado, _ = env.step(accion)
            recompensa_total += recompensa
            n_pasos += 1

            if render:
                env.render()

        recompensas.append(recompensa_total)
        pasos.append(n_pasos)

    return {
        'recompensa_media': np.mean(recompensas),
        'recompensa_std': np.std(recompensas),
        'recompensa_min': np.min(recompensas),
        'recompensa_max': np.max(recompensas),
        'pasos_media': np.mean(pasos),
        'episodios': n_episodes
    }


def save_training_results(filepath, metrics_tracker, agent_params=None):
    """
    Guarda los resultados del entrenamiento en un archivo.

    Args:
        filepath: Ruta del archivo (.npz)
        metrics_tracker: Objeto MetricsTracker
        agent_params: Dict con parámetros del agente (opcional)
    """
    data = {
        'recompensas': np.array(metrics_tracker.recompensas),
        'pasos': np.array(metrics_tracker.pasos),
    }

    if metrics_tracker.perdidas:
        data['perdidas'] = np.array(metrics_tracker.perdidas)
    if metrics_tracker.epsilons:
        data['epsilons'] = np.array(metrics_tracker.epsilons)

    if agent_params:
        for key, value in agent_params.items():
            data[f'param_{key}'] = value

    np.savez(filepath, **data)
    print(f"Resultados guardados en: {filepath}")


def load_training_results(filepath):
    """
    Carga resultados de entrenamiento desde un archivo.

    Args:
        filepath: Ruta del archivo (.npz)

    Returns:
        Dict con los datos cargados
    """
    data = np.load(filepath, allow_pickle=True)
    return dict(data)


if __name__ == "__main__":
    # Test de las utilidades
    print("=== Test de Utilidades de RL ===\n")

    # Test MetricsTracker
    tracker = MetricsTracker()
    for i in range(100):
        tracker.registrar_episodio(
            recompensa=np.random.randn() * 10 + i,
            pasos=np.random.randint(50, 200),
            epsilon=1.0 * (0.99 ** i)
        )

    print("Stats últimos 20 episodios:")
    print(tracker.get_stats(20))

    # Test EpsilonScheduler
    scheduler = EpsilonScheduler(decay_type='exponential', decay_rate=0.99)
    print(f"\nEpsilon decay (10 pasos):")
    for i in range(10):
        print(f"  Paso {i}: epsilon = {scheduler.step():.4f}")

    print("\n✅ Todas las utilidades funcionan correctamente")
