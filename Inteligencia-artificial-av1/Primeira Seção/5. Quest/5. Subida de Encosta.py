import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funcao objetivo
def funcao_objetivo(x1, x2):
    return (x1 * np.cos(x1)) / 20 + np.exp(-((x1)**2 + (x2)**2)) + 0.01 * x1 * x2

# Algoritmo de subida de encosta para maximizacao
def subida_encosta(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes, passo):
    x1 = np.random.uniform(dominio_x1[0], dominio_x1[1])
    x2 = np.random.uniform(dominio_x2[0], dominio_x2[1])
    valor_otimo = funcao_objetivo(x1, x2)

    for _ in range(total_iteracoes):
        # Gerar novo ponto candidato
        candidato_x1 = x1 + np.random.uniform(-passo, passo)
        candidato_x2 = x2 + np.random.uniform(-passo, passo)
        candidato_x1 = max(min(candidato_x1, dominio_x1[1]), dominio_x1[0])
        candidato_x2 = max(min(candidato_x2, dominio_x2[1]), dominio_x2[0])
        valor_candidato = funcao_objetivo(candidato_x1, candidato_x2)

        # Se o novo ponto é melhor, mova para ele
        if valor_candidato > valor_otimo:
            x1, x2, valor_otimo = candidato_x1, candidato_x2, valor_candidato

    return (x1, x2), valor_otimo

# Parametros do algoritmo
dominio_x1 = (-10, 10)
dominio_x2 = (-10, 10)
total_iteracoes = 1000
passo = 0.1  # Tamanho do passo para gerar novos pontos candidatos

# Execucao do algoritmo
(ponto_otimo, valor_otimo) = subida_encosta(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes, passo)
print("Ponto ótimo:", ponto_otimo)
print("Valor ótimo:", valor_otimo)

# Criacao do grafico 3D
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.6)
ax.scatter(ponto_otimo[0], ponto_otimo[1], valor_otimo, color='r', s=100, label='Ponto Ótimo')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()
plt.show()
