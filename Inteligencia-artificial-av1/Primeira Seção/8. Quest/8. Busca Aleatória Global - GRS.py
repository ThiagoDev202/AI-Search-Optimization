# – Trabalho AV1 - Busca/Otimizacao Meta-heuristica -


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funcao objetivo de acordo com a questao
def funcao_objetivo(x1, x2):
    return -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1/2 + 47))) - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))

# Algoritmo de Busca Aleatoria Global (GRS) para minimizacao
def busca_aleatoria_global(funcao_obj, dominio_x1, dominio_x2, iteracoes):
    melhor_x1, melhor_x2 = None, None
    melhor_valor = np.inf  # Iniciamos com infinito para minimizacao
    
    for _ in range(iteracoes):
        # Gero ponto candidato aleatorio dentro do dominio
        x1_candidato = np.random.uniform(dominio_x1[0], dominio_x1[1])
        x2_candidato = np.random.uniform(dominio_x2[0], dominio_x2[1])
        # Avalio o ponto candidato
        valor_atual = funcao_obj(x1_candidato, x2_candidato)
        # Atualizo o melhor valor e ponto, se encontrado
        if valor_atual < melhor_valor:
            melhor_x1, melhor_x2 = x1_candidato, x2_candidato
            melhor_valor = valor_atual

    return (melhor_x1, melhor_x2), melhor_valor

# Definindo os parametros do dominio e iteracoes
dominio_x1 = (-200, 200)
dominio_x2 = (-200, 200)
iteracoes = 1000

# Executando o algoritmo de busca aleatoria global
ponto_otimo, valor_otimo = busca_aleatoria_global(funcao_objetivo, dominio_x1, dominio_x2, iteracoes)
print("Ponto otimo:", ponto_otimo)
print("Valor otimo:", valor_otimo)

# Preparando os dados para o grafico 3D
x = np.linspace(dominio_x1[0], dominio_x1[1], 400)
y = np.linspace(dominio_x2[0], dominio_x2[1], 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

# Criando o grafico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.6)
ax.scatter(ponto_otimo[0], ponto_otimo[1], valor_otimo, color='r', s=100, label='Ponto otimo')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
ax.legend()
plt.tight_layout()
plt.show()





