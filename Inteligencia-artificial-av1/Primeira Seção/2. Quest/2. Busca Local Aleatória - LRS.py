# – Trabalho AV1 - Busca/Otimizacao Meta-heuristica -

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a nova funcao objetivo de maximizacao
def nova_funcao_objetivo(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 2.2)**2))

# Implementacao do algoritmo Local Random Search (LRS) de maximizacao
def busca_local_aleatoria_max(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes):
    melhor_posicao = None
    melhor_valor = -np.inf  # Inicia com infinito negativo para maximizacao

    # Loop principal do LRS
    for _ in range(total_iteracoes):
        x1_candidato = np.random.uniform(dominio_x1[0], dominio_x1[1])
        x2_candidato = np.random.uniform(dominio_x2[0], dominio_x2[1])
        valor_candidato = funcao_objetivo(x1_candidato, x2_candidato)

        if valor_candidato > melhor_valor:  # Mudança para condicao de maximizacao
            melhor_posicao = (x1_candidato, x2_candidato)
            melhor_valor = valor_candidato

    return melhor_posicao, melhor_valor

# Parametros ajustados para o novo dominio das variaveis
dominio_x1 = (-2, 4)
dominio_x2 = (-2, 5)
total_iteracoes = 1000
rodadas = 100
solucoes = []

# Execucao do LRS para maximizacao
for _ in range(rodadas):
    posicao_otima, valor_otimo = busca_local_aleatoria_max(nova_funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes)
    solucoes.append(posicao_otima)

# Calculo da moda para as solucoes otimas
def calcular_moda(valores):
    valores, contagens = np.unique(valores, return_counts=True)
    indice_moda = np.argmax(contagens)
    return valores[indice_moda]

moda_x1 = calcular_moda([s[0] for s in solucoes])
moda_x2 = calcular_moda([s[1] for s in solucoes])

print(moda_x1)
print(moda_x2)

# Criacao do grafico da funcao objetivo com a melhor solucao
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = nova_funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', alpha=0.7)
ax.scatter(moda_x1, moda_x2, nova_funcao_objetivo(moda_x1, moda_x2), color='r', s=50, label='Moda das Solucoes')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Funcao Objetivo')
ax.legend()
plt.tight_layout()
plt.show()


