# – Trabalho AV1 - Busca/Otimização Meta-heurística -


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a funcao objetivo conforme a questao
def funcao_objetivo(x1, x2):
    return -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))

# Implementa o algoritmo de Subida de Encosta para minimizacao
def subida_encosta(dominio_x1, dominio_x2, total_iteracoes, passo):
    x1 = np.random.uniform(*dominio_x1)
    x2 = np.random.uniform(*dominio_x2)
    valor_otimo = funcao_objetivo(x1, x2)

    for _ in range(total_iteracoes):
        # Gera um novo ponto candidato
        candidato_x1 = np.clip(x1 + np.random.uniform(-passo, passo), *dominio_x1)
        candidato_x2 = np.clip(x2 + np.random.uniform(-passo, passo), *dominio_x2)
        valor_candidato = funcao_objetivo(candidato_x1, candidato_x2)

        # Se o novo ponto e melhor (menor, pois estamos minimizando), move para ele
        if valor_candidato < valor_otimo:
            x1, x2, valor_otimo = candidato_x1, candidato_x2, valor_candidato

    return (x1, x2), valor_otimo

# Define os parametros do dominio e iteracoes
dominio_x1 = (-200, 200)
dominio_x2 = (-200, 200)
total_iteracoes = 1000
passo = 0.1  # Tamanho do passo para gerar novos pontos candidatos

# Executa o algoritmo de Subida de Encosta
(ponto_otimo, valor_otimo) = subida_encosta(dominio_x1, dominio_x2, total_iteracoes, passo)
print("Ponto otimo:", ponto_otimo)
print("Valor otimo:", valor_otimo)

# Cria o grafico 3D da funcao objetivo
x = np.linspace(*dominio_x1, 400)
y = np.linspace(*dominio_x2, 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

# Preparar para a plotagem do grafico 3D com a superficie e o ponto otimo encontrado
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.6)
ax.scatter(*ponto_otimo, valor_otimo, color='r', s=100, label='Ponto Otimo')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
ax.legend()
plt.tight_layout()

