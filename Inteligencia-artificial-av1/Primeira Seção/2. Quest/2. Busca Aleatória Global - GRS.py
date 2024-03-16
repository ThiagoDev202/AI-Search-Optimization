# – Trabalho AV1 - Busca/Otimização Meta-heurística -

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funcao objetivo
def funcao_objetivo(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 2.2)**2))

# GRS - Maximizacao
def busca_aleatoria_global(funcao_obj, dominio_x1, dominio_x2, iteracoes):
    melhor_ponto = None
    melhor_valor = -np.inf  # Iniciamos com infinito negativo para maximizacao
    
    for _ in range(iteracoes):
        # Gero ponto candidato aleatorio dentro do dominio
        x1, x2 = np.random.uniform(dominio_x1[0], dominio_x1[1]), np.random.uniform(dominio_x2[0], dominio_x2[1])
        # Avalio o ponto candidato
        valor_atual = funcao_obj(x1, x2)
        # Atualizo do melhor valor e ponto, se encontrado
        if valor_atual > melhor_valor:
            melhor_ponto = (x1, x2)
            melhor_valor = valor_atual

    return melhor_ponto, melhor_valor

# Parametros do grafico eixo x (-2 ~ 4) eixo y (-2 ~ 5)
dominio_x1 = (-2, 4)
dominio_x2 = (-2, 5)
iteracoes = 1000
rodadas = 100
resultados = []

# Execucao do GRS para a questao 2
for _ in range(rodadas):
    ponto_otimo, valor_otimo = busca_aleatoria_global(funcao_objetivo, dominio_x1, dominio_x2, iteracoes)
    resultados.append(ponto_otimo)

# Calculo da moda das solucoes otimas
def calcular_moda(valores):
    valores, contagens = np.unique(valores, return_counts=True)
    indice_moda = np.argmax(contagens)
    return valores[indice_moda]

valores_x1 = [pos[0] for pos in resultados]
valores_x2 = [pos[1] for pos in resultados]
moda_x1 = calcular_moda(valores_x1)
moda_x2 = calcular_moda(valores_x2)

# Impressao da moda das solucoes
print(moda_x1)
print(moda_x2)

# Geracao do grafico da funcao objetivo com a melhor solucao
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', alpha=0.8)
ax.scatter(moda_x1, moda_x2, funcao_objetivo(moda_x1, moda_x2), color='r', s=50, label='Moda das Soluções')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()
plt.tight_layout()
plt.show()


