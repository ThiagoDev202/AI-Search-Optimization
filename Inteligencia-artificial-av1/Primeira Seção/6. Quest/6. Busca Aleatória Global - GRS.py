# – Trabalho AV1 - Busca/Otimização Meta-heurística -


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funcao objetivo da questao 6
def funcao_objetivo(x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

# Algoritmo de Busca Aleatoria Global (GRS) para maximizacao
def busca_aleatoria_global(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes):
    melhor_x1, melhor_x2 = None, None
    melhor_valor = -np.inf  # Iniciamos com infinito negativo para maximizacao

    # Loop principal do GRS
    for _ in range(total_iteracoes):
        # Gera um ponto candidato aleatorio dentro do dominio
        x1_candidato = np.random.uniform(dominio_x1[0], dominio_x1[1])
        x2_candidato = np.random.uniform(dominio_x2[0], dominio_x2[1])
        
        # Avalia o ponto candidato
        valor_candidato = funcao_objetivo(x1_candidato, x2_candidato)
        
        # Se o valor do candidato for melhor que o atual, atualiza a melhor solucao
        if valor_candidato > melhor_valor:
            melhor_x1, melhor_x2 = x1_candidato, x2_candidato
            melhor_valor = valor_candidato

    return (melhor_x1, melhor_x2), melhor_valor

# Definindo os parametros do dominio
dominio_x1 = (-1, 1)
dominio_x2 = (-1, 1)
total_iteracoes = 1000

# Executando o algoritmo de busca aleatoria global
(ponto_otimo, valor_otimo) = busca_aleatoria_global(funcao_objetivo, dominio_x1, dominio_x2, total_iteracoes)

# Gerando os dados para o grafico
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

# Criando o grafico 3D da funcao objetivo com o ponto otimo encontrado
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.6)
ax.scatter(ponto_otimo[0], ponto_otimo[1], valor_otimo, color='r', s=100, label='Ponto Otimo')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
ax.legend()
plt.tight_layout()
plt.show()

