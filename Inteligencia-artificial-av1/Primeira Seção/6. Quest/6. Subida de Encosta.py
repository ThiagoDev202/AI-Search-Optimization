# – Trabalho AV1 - Busca/Otimização Meta-heurística -


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função objetivo
def funcao_objetivo(x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

# Algoritmo de subida de encosta para maximização
def subida_encosta(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes, passo):
    x1 = np.random.uniform(dominio_x1[0], dominio_x1[1])
    x2 = np.random.uniform(dominio_x2[0], dominio_x2[1])
    valor_otimo = funcao_objetivo(x1, x2)

    for _ in range(total_iteracoes):
        candidato_x1 = np.clip(x1 + np.random.uniform(-passo, passo), dominio_x1[0], dominio_x1[1])
        candidato_x2 = np.clip(x2 + np.random.uniform(-passo, passo), dominio_x2[0], dominio_x2[1])
        valor_candidato = funcao_objetivo(candidato_x1, candidato_x2)

        if valor_candidato > valor_otimo:
            x1, x2, valor_otimo = candidato_x1, candidato_x2, valor_candidato

    return (x1, x2), valor_otimo

# Parâmetros do algoritmo
dominio_x1 = (-1, 1)
dominio_x2 = (-1, 1)
total_iteracoes = 1000
passo = 0.05  # Tamanho do passo para gerar novos pontos candidatos

# Execução do algoritmo
(ponto_otimo, valor_otimo) = subida_encosta(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes, passo)
print("Ponto ótimo:", ponto_otimo)
print("Valor ótimo:", valor_otimo)

# Criação do gráfico 3D
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.6)
ax.scatter(ponto_otimo[0], ponto_otimo[1], valor_otimo, color='r', s=100, label='Ponto Ótimo')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()
plt.tight_layout()
plt.show()

