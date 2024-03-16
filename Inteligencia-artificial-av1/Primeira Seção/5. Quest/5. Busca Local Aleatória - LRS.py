# – Trabalho AV1 - Busca/Otimizacao Meta-heuristica -

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funcao objetivo da questao 5
def funcao_objetivo(x1, x2):
    return (x1 * np.cos(x1)) / 20 + np.exp(-((x1)**2 + (x2)**2)) + 0.01 * x1 * x2

# Algoritmo de Busca Local Aleatoria (LRS) para maximizacao
def busca_local_aleatoria(funcao_obj, dominio_x1, dominio_x2, iteracoes, sigma):
    # Inicializa a melhor solucao de forma aleatoria dentro do domínio
    melhor_x1 = np.random.uniform(dominio_x1[0], dominio_x1[1])
    melhor_x2 = np.random.uniform(dominio_x2[0], dominio_x2[1])
    melhor_valor = funcao_obj(melhor_x1, melhor_x2)

    for _ in range(iteracoes):
        # Gera um novo ponto candidato no entorno da melhor solucao atual
        candidato_x1 = np.random.normal(melhor_x1, sigma)
        candidato_x2 = np.random.normal(melhor_x2, sigma)
        # Garante que o candidato esta dentro do domínio
        candidato_x1 = np.clip(candidato_x1, dominio_x1[0], dominio_x1[1])
        candidato_x2 = np.clip(candidato_x2, dominio_x2[0], dominio_x2[1])
        valor_candidato = funcao_obj(candidato_x1, candidato_x2)
        
        # Se o novo candidato for melhor, atualiza a melhor solucao
        if valor_candidato > melhor_valor:
            melhor_x1, melhor_x2, melhor_valor = candidato_x1, candidato_x2, valor_candidato

    return (melhor_x1, melhor_x2), melhor_valor

# Parâmetros para o LRS
dominio_x1 = (-10, 10)
dominio_x2 = (-10, 10)
iteracoes = 1000
sigma = 0.1  # Desvio padrao para a geracao dos candidatos

# Executa o algoritmo de busca local aleatoria
(ponto_otimo, valor_otimo) = busca_local_aleatoria(funcao_objetivo, dominio_x1, dominio_x2, iteracoes, sigma)
print("Ponto otimo:", ponto_otimo)
print("Valor otimo:", valor_otimo)

# Criacao do grafico 3D
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.6)
ax.scatter(ponto_otimo[0], ponto_otimo[1], valor_otimo, color='r', s=100, label='Ponto otimo')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()
plt.show()






