# – Trabalho AV1 - Busca/Otimização Meta-heurística -

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def funcao_objetivo(x1, x2):
    return (x1**2 - 10 * np.cos(2 * np.pi * x1)) + (x2**2 - 10 * np.cos(2 * np.pi * x2)) + 20

def busca_local_aleatoria(funcao_objetivo, dominio, total_iteracoes):
    melhor_x1, melhor_x2 = None, None
    melhor_valor = np.inf

    for _ in range(total_iteracoes):
        x1 = np.random.uniform(dominio[0], dominio[1])
        x2 = np.random.uniform(dominio[0], dominio[1])
        valor_atual = funcao_objetivo(x1, x2)

        if valor_atual < melhor_valor:
            melhor_x1, melhor_x2 = x1, x2
            melhor_valor = valor_atual

    return (melhor_x1, melhor_x2), melhor_valor

dominio = (-5.12, 5.12)
total_iteracoes = 1000
rodadas = 100
resultados = []

for _ in range(rodadas):
    resultado, valor = busca_local_aleatoria(funcao_objetivo, dominio, total_iteracoes)
    resultados.append((resultado, valor))

valores_x1 = [resultado[0][0] for resultado in resultados]
valores_x2 = [resultado[0][1] for resultado in resultados]

moda_x1 = max(set(valores_x1), key=valores_x1.count)
moda_x2 = max(set(valores_x2), key=valores_x2.count)

x = np.linspace(dominio[0], dominio[1], 400)
y = np.linspace(dominio[0], dominio[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none', alpha=0.8)
ax.scatter(moda_x1, moda_x2, funcao_objetivo(moda_x1, moda_x2), color='r', s=50, label='Moda')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Função Objetivo e Moda das Soluções')
ax.legend()
plt.show()


