import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a nova funcao objetivo para maximizacao 
def funcao_objetivo(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 2.2)**2))

# Estou procurando por valores de x1 e x2 que resultem no maior valor possivel da funcao objetivo
def busca_local_aleatoria_q2(funcao_obj, dominio_x1, dominio_x2, iteracoes):
    melhor_ponto = None
    melhor_valor = -np.inf  # Para maximizacao
    
    for _ in range(iteracoes):
        x1, x2 = np.random.uniform(dominio_x1[0], dominio_x1[1]), np.random.uniform(dominio_x2[0], dominio_x2[1])
        valor_atual = funcao_obj(x1, x2)
        if valor_atual > melhor_valor:
            melhor_ponto = (x1, x2)
            melhor_valor = valor_atual

    return melhor_ponto, melhor_valor

# Parametros do algoritmo para a funcao da 
dominio_x1 = (-2, 4)
dominio_x2 = (-2, 5)
iteracoes_q2 = 1000
rodadas_q2 = 100
resultados = []

# Executando o LRS para maximizacao
for _ in range(rodadas_q2):
    ponto_otimo, valor_otimo = busca_local_aleatoria_q2(funcao_objetivo, dominio_x1, dominio_x2, iteracoes_q2)
    resultados.append(ponto_otimo)

# Calculando a moda das posições ótimas encontradas
def calcular_moda_q2(valores):
    valores, contagens = np.unique(valores, return_counts=True)
    indice_moda = np.argmax(contagens)
    return valores[indice_moda]

valores_x1 = [pos[0] for pos in resultados]
valores_x2 = [pos[1] for pos in resultados]
moda_x1 = calcular_moda_q2(valores_x1)
moda_x2 = calcular_moda_q2(valores_x2)

# Gerando o gráfico da funcao objetivo com a melhor solucao
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

# Criando o grafico
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', alpha=0.8)
# Add a moda das solucoes no grafico, via funcao de x1 e x2
ax.scatter(moda_x1, moda_x2, funcao_objetivo(moda_x1, moda_x2), color='r', s=50, label='Moda das Soluções')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()
plt.tight_layout()
plt.show()
