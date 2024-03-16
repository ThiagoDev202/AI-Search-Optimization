# – Trabalho AV1 - Busca/Otimização Meta-heurística -

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definicao da funcao objetivo
def funcao_objetivo(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 2.2)**2))

# Algoritmo de subida de encosta para maximizacao
def subida_encosta(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes, epsilon):
    # Gera um ponto inicial aleatorio dentro do dominio
    x1, x2 = np.random.uniform(dominio_x1[0], dominio_x1[1]), np.random.uniform(dominio_x2[0], dominio_x2[1])
    valor_atual = funcao_objetivo(x1, x2)
    
    for _ in range(total_iteracoes):
        # Gera um novo ponto candidato proximo ao atual
        candidato_x1 = x1 + np.random.uniform(-epsilon, epsilon)
        candidato_x2 = x2 + np.random.uniform(-epsilon, epsilon)
        # Mantem os candidatos dentro dos limites do dominio
        candidato_x1 = np.clip(candidato_x1, dominio_x1[0], dominio_x1[1])
        candidato_x2 = np.clip(candidato_x2, dominio_x2[0], dominio_x2[1])
        valor_candidato = funcao_objetivo(candidato_x1, candidato_x2)
        
        # Se o valor do candidato for melhor
        if valor_candidato > valor_atual:
            # Atualizo o ponto otimo
            x1, x2, valor_atual = candidato_x1, candidato_x2, valor_candidato
        else:
            # Reduz o epsilon
            epsilon *= 0.99
    
    return (x1, x2), valor_atual

# Parametros da questao
dominio_x1 = (-2, 4)
dominio_x2 = (-2, 5)
total_iteracoes = 1000
epsilon = 0.1  # Valor inicial para o tamanho do passo

# Executa o algoritmo de subida de encosta
(posicao_otima, valor_otimo) = subida_encosta(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes, epsilon)

# Grafico da funcao objetivo
x = np.linspace(-2, 4, 100)
y = np.linspace(-2, 5, 100)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none', alpha=0.8)
ax.scatter(posicao_otima[0], posicao_otima[1], valor_otimo, color='r', s=50, label='Melhor Solucao')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()
plt.show()
