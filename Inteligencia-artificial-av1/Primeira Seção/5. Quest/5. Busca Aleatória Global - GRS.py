# – Trabalho AV1 - Busca/Otimizacao Meta-heuristica -


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funcao objetivo da questao.
def funcao_objetivo(x1, x2):
    return x1 * np.cos(x1) / 20 + np.exp(-((x1)**2 + (x2)**2)) + 0.01 * x1 * x2

# Algoritmo de Busca Aleatoria Global (Global Random Search - GRS) para maximizacao.
def busca_aleatoria_global(funcao_obj, dominio_x1, dominio_x2, iteracoes):
    melhor_ponto = None
    melhor_valor = -np.inf  # Para maximizacao, comecamos com infinito negativo.
    
    for _ in range(iteracoes):
        x1_candidato = np.random.uniform(dominio_x1[0], dominio_x1[1])
        x2_candidato = np.random.uniform(dominio_x2[0], dominio_x2[1])
        valor_candidato = funcao_obj(x1_candidato, x2_candidato)

        if valor_candidato > melhor_valor:
            melhor_valor = valor_candidato
            melhor_ponto = (x1_candidato, x2_candidato)

    return melhor_ponto, melhor_valor

# Definindo os parametros do dominio e numero de iteracoes.
dominio_x1 = (-10, 10)
dominio_x2 = (-10, 10)
iteracoes = 1000

# Executando o algoritmo de busca aleatoria global.
melhor_ponto, melhor_valor = busca_aleatoria_global(funcao_objetivo, dominio_x1, dominio_x2, iteracoes)

# Gerando o grafico da funcao objetivo e marcando o ponto otimo encontrado.
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', alpha=0.8)
ax.scatter(*melhor_ponto, melhor_valor, color='r', s=50, label='Melhor Solucao')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Valor da funcao objetivo')
ax.legend()
plt.tight_layout()
plt.show()

# Ponto otimo encontrado e o valor da funcao objetivo correspondente.
melhor_ponto, melhor_valor


