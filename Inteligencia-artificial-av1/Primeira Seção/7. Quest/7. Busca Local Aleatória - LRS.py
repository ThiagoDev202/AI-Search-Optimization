# – Trabalho AV1 - Busca/Otimização Meta-heurística -


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a funcao objetivo conforme a questao
def funcao_objetivo(x1, x2):
    return -np.sin(x1) * (np.sin(x1**2 / np.pi)**10) - np.sin(x2) * (np.sin(2 * x2**2 / np.pi)**10)

# Implementa o algoritmo de Busca Local Aleatoria para maximizacao
def busca_local_aleatoria(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes):
    melhor_x1, melhor_x2 = None, None
    melhor_valor = -np.inf  # Para maximizacao, comecamos com o valor negativo infinito

    for _ in range(total_iteracoes):
        # Gera um novo ponto candidato aleatorio dentro do dominio
        x1_candidato = np.random.uniform(dominio_x1[0], dominio_x1[1])
        x2_candidato = np.random.uniform(dominio_x2[0], dominio_x2[1])
        
        # Avalia o ponto candidato
        valor_candidato = funcao_objetivo(x1_candidato, x2_candidato)
        
        # Atualiza a melhor solucao encontrada se o ponto candidato e melhor
        if valor_candidato > melhor_valor:
            melhor_x1, melhor_x2 = x1_candidato, x2_candidato
            melhor_valor = valor_candidato

    return (melhor_x1, melhor_x2), melhor_valor

# Define os parametros do dominio
dominio_x1 = (0, np.pi)
dominio_x2 = (0, np.pi)
total_iteracoes = 1000

# Executa o algoritmo de busca local aleatoria
(ponto_otimo, valor_otimo) = busca_local_aleatoria(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes)

# Gera os dados para o grafico
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

# Cria o grafico 3D da funcao objetivo com o ponto otimo encontrado
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.6)
ax.scatter(ponto_otimo[0], ponto_otimo[1], valor_otimo, color='r', s=100, label='Ponto Otimo')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
ax.legend()
plt.tight_layout()
plt.show()

