# – Trabalho AV1 - Busca/Otimização Meta-heurística -

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funcao objetivo
def funcao_objetivo_q2(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 2.2)**2))

# GRS - Maximizacao
def busca_aleatoria_global_max(funcao_obj, dominio_x1, dominio_x2, iteracoes):
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
dominio_x1_q2 = (-2, 4)
dominio_x2_q2 = (-2, 5)
iteracoes_q2 = 1000
rodadas_q2 = 100
resultados_q2 = []

# Execucao do GRS para a questao 2
for _ in range(rodadas_q2):
    ponto_otimo, valor_otimo = busca_aleatoria_global_max(funcao_objetivo_q2, dominio_x1_q2, dominio_x2_q2, iteracoes_q2)
    resultados_q2.append(ponto_otimo)

# Calculo da moda das solucoes otimas
def calcular_moda_q2(valores):
    valores, contagens = np.unique(valores, return_counts=True)
    indice_moda = np.argmax(contagens)
    return valores[indice_moda]

valores_x1_q2 = [pos[0] for pos in resultados_q2]
valores_x2_q2 = [pos[1] for pos in resultados_q2]
moda_x1_q2 = calcular_moda_q2(valores_x1_q2)
moda_x2_q2 = calcular_moda_q2(valores_x2_q2)

# Impressao da moda das solucoes
print(moda_x1_q2)
print(moda_x2_q2)

# Geracao do grafico da funcao objetivo com a melhor solucao
x = np.linspace(dominio_x1_q2[0], dominio_x1_q2[1], 400)
y = np.linspace(dominio_x2_q2[0], dominio_x2_q2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo_q2(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', alpha=0.8)
ax.scatter(moda_x1_q2, moda_x2_q2, funcao_objetivo_q2(moda_x1_q2, moda_x2_q2), color='r', s=50, label='Moda das Soluções')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()
plt.tight_layout()
plt.show()


