# – Trabalho AV1 - Busca/Otimização Meta-heurística -

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def funcao_objetivo(x1, x2):
    return (x1**2 - 10 * np.cos(2 * np.pi * x1)) + (x2**2 - 10 * np.cos(2 * np.pi * x2)) + 20


def hill_climbing(funcao_objetivo, dominio_x1, dominio_x2, max_iteracoes, passo):
    x1, x2 = np.random.uniform(dominio_x1[0], dominio_x1[1]), np.random.uniform(dominio_x2[0], dominio_x2[1])
    valor_atual = funcao_objetivo(x1, x2)

    for _ in range(max_iteracoes):
        mov_x1, mov_x2 = np.random.uniform(-passo, passo), np.random.uniform(-passo, passo)
        novo_x1, novo_x2 = x1 + mov_x1, x2 + mov_x2
        # Mantém dentro dos limites do domínio
        novo_x1 = max(min(novo_x1, dominio_x1[1]), dominio_x1[0])
        novo_x2 = max(min(novo_x2, dominio_x2[1]), dominio_x2[0])
        novo_valor = funcao_objetivo(novo_x1, novo_x2)

        if novo_valor < valor_atual:
            x1, x2, valor_atual = novo_x1, novo_x2, novo_valor

    return (x1, x2), valor_atual

dominio = (-5.12, 5.12)
max_iteracoes = 1000
passo = 0.1
rodadas = 100
solucoes = []

for _ in range(rodadas):
    solucao, valor = hill_climbing(funcao_objetivo, dominio, dominio, max_iteracoes, passo)
    solucoes.append(solucao)

moda_x1 = max(set(map(lambda x: x[0], solucoes)), key = lambda x: solucoes.count(x))
moda_x2 = max(set(map(lambda x: x[1], solucoes)), key = lambda x: solucoes.count(x))


x = np.linspace(dominio[0], dominio[1], 400)
y = np.linspace(dominio[0], dominio[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', alpha=0.7)
ax.scatter(moda_x1, moda_x2, funcao_objetivo(moda_x1, moda_x2), color='r', s=50, label='Moda das Soluções')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Z')
ax.legend()
plt.show()



