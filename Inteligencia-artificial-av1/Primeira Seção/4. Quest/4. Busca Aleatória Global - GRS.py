# – Trabalho AV1 - Busca/Otimização Meta-heurística -

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def funcao_objetivo(x1, x2):
    return (x1**2 - 10 * np.cos(2 * np.pi * x1)) + (x2**2 - 10 * np.cos(2 * np.pi * x2)) + 20

def busca_local_aleatoria_min(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes):
    melhor_posicao = None
    melhor_valor = np.inf

    for _ in range(total_iteracoes):
        x1_candidato = np.random.uniform(dominio_x1[0], dominio_x1[1])
        x2_candidato = np.random.uniform(dominio_x2[0], dominio_x2[1])
        valor_candidato = funcao_objetivo(x1_candidato, x2_candidato)

        if valor_candidato < melhor_valor:
            melhor_posicao = (x1_candidato, x2_candidato)
            melhor_valor = valor_candidato

    return melhor_posicao, melhor_valor

dominio_x1 = (-5.12, 5.12)
dominio_x2 = (-5.12, 5.12)
total_iteracoes = 1000
rodadas = 100
melhores_posicoes = []

for _ in range(rodadas):
    posicao_otima, valor_otimo = busca_local_aleatoria_min(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes)
    melhores_posicoes.append(posicao_otima)

valores_x1 = [pos[0] for pos in melhores_posicoes]
valores_x2 = [pos[1] for pos in melhores_posicoes]

valores, contagens_x1 = np.unique(valores_x1, return_counts=True)
indice_moda_x1 = np.argmax(contagens_x1)
moda_x1 = valores[indice_moda_x1]

valores, contagens_x2 = np.unique(valores_x2, return_counts=True)
indice_moda_x2 = np.argmax(contagens_x2)
moda_x2 = valores[indice_moda_x2]

x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none', alpha=0.8)
ax.scatter(moda_x1, moda_x2, funcao_objetivo(moda_x1, moda_x2), color='r', s=50, label='Moda')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()
plt.show()


