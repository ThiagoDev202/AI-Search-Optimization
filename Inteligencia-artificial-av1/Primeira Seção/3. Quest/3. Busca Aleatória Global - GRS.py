# – Trabalho AV1 - Busca/Otimização Meta-heurística -

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funcao objetivo
def funcao_objetivo(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)

# Algoritmo de Busca Aleatoria Global de minimizacao
def busca_aleatoria_global(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes):
    melhor_ponto = None
    melhor_valor = np.inf

    for _ in range(total_iteracoes):
        x1_candidato = np.random.uniform(dominio_x1[0], dominio_x1[1])
        x2_candidato = np.random.uniform(dominio_x2[0], dominio_x2[1])
        valor_candidato = funcao_objetivo(x1_candidato, x2_candidato)

        if valor_candidato < melhor_valor:
            melhor_ponto = (x1_candidato, x2_candidato)
            melhor_valor = valor_candidato

    return melhor_ponto, melhor_valor

# Parametros da questao
dominio_x1 = (-8, 8)
dominio_x2 = (-8, 8)
total_iteracoes = 1000
rodadas = 100
resultados = []

# Execucao do algoritmo GRS para encontrar a solucao otima
for _ in range(rodadas):
    ponto_otimo, valor_otimo = busca_aleatoria_global(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes)
    resultados.append(ponto_otimo)

# Calculo da moda das solucoes otimas
def calcular_moda(valores):
    valores, contagens = np.unique(valores, return_counts=True)
    indice_moda = np.argmax(contagens)
    return valores[indice_moda]

moda_x1 = calcular_moda([s[0] for s in resultados])
moda_x2 = calcular_moda([s[1] for s in resultados])

# Grafico da funcao objetivo com a solucao mais frequente
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', alpha=0.7)
ax.scatter(moda_x1, moda_x2, funcao_objetivo(moda_x1, moda_x2), color='r', s=50, label='Moda das Soluções')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('F(x1, x2)')
ax.legend()
plt.tight_layout()
plt.show()


