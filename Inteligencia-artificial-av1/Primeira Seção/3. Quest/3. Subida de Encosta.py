# – Trabalho AV1 - Busca/Otimizacao Meta-heuristica -

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definicao da nova funcao objetivo
def funcao_objetivo(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.e

# Implementacao do algoritmo de subida de encosta para minimizacao
def subida_encosta_min(funcao_obj, dominio_x1, dominio_x2, iteracoes, epsilon):
    melhor_ponto = None
    melhor_valor = np.inf
    
    for _ in range(iteracoes):
        # Geracao de ponto candidato aleatorio dentro do dominio
        x1, x2 = np.random.uniform(dominio_x1[0], dominio_x1[1]), np.random.uniform(dominio_x2[0], dominio_x2[1])
        valor_atual = funcao_obj(x1, x2)
        if valor_atual < melhor_valor:
            melhor_ponto = (x1, x2)
            melhor_valor = valor_atual

        # Perturbacao do ponto atual para a busca local
        x1_novo, x2_novo = x1 + np.random.uniform(-epsilon, epsilon), x2 + np.random.uniform(-epsilon, epsilon)
        x1_novo, x2_novo = np.clip(x1_novo, dominio_x1[0], dominio_x1[1]), np.clip(x2_novo, dominio_x2[0], dominio_x2[1])
        valor_novo = funcao_obj(x1_novo, x2_novo)

        # Se um melhor ponto for encontrado, atualize o melhor ponto e valor
        if valor_novo < melhor_valor:
            melhor_ponto = (x1_novo, x2_novo)
            melhor_valor = valor_novo
        else:
            # Diminuir epsilon para refinar a busca
            epsilon *= 0.99

    return melhor_ponto, melhor_valor

# Parametros do algoritmo para a funcao da 
dominio_x1 = (-8, 8)
dominio_x2 = (-8, 8)
iteracoes = 1000
epsilon_inicial = 0.1  # Valor inicial para o tamanho do passo
rodadas = 100
resultados = []
        
# Execucao do algoritmo para a 
for _ in range(rodadas):
    ponto_otimo, valor_otimo = subida_encosta_min(funcao_objetivo, dominio_x1, dominio_x2, iteracoes, epsilon_inicial)
    resultados.append(ponto_otimo)

# Calculando a moda das solucoes otimas
def calcular_moda(valores):
    valores, contagens = np.unique(valores, return_counts=True)
    indice_moda = np.argmax(contagens)
    return valores[indice_moda]

valores_x1 = [pos[0] for pos in resultados]
valores_x2 = [pos[1] for pos in resultados]
moda_x1 = calcular_moda(valores_x1)
moda_x2 = calcular_moda(valores_x2)

# Gerando o grafico da funcao objetivo com a melhor solucao
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

# Criando o grafico
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', alpha=0.8)
# Add a moda das solucoes no grafico, via funcao de x1 e x2
ax.scatter(moda_x1, moda_x2, funcao_objetivo(moda_x1, moda_x2), color='r', s=50, label='Moda das Solucoes')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()
plt.tight_layout()
plt.show()



