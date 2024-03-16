# – Trabalho AV1 - Busca/Otimização Meta-heurística -


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funcao objetivo
def funcao_objetivo(x1, x2):
    # Insira aqui a expressao da sua funcao objetivo
    return -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1/2 + 47))) - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))

# Algoritmo de Busca Local Aleatoria (LRS) para maximizacao
def busca_local_aleatoria(funcao_obj, dominio_x1, dominio_x2, iteracoes, sigma):
    melhor_x1 = np.random.uniform(*dominio_x1)
    melhor_x2 = np.random.uniform(*dominio_x2)
    melhor_valor = funcao_obj(melhor_x1, melhor_x2)

    for _ in range(iteracoes):
        candidato_x1 = np.random.normal(melhor_x1, sigma)
        candidato_x2 = np.random.normal(melhor_x2, sigma)
        candidato_x1 = np.clip(candidato_x1, *dominio_x1)
        candidato_x2 = np.clip(candidato_x2, *dominio_x2)
        valor_candidato = funcao_obj(candidato_x1, candidato_x2)

        if valor_candidato > melhor_valor:
            melhor_x1, melhor_x2, melhor_valor = candidato_x1, candidato_x2, valor_candidato

    return (melhor_x1, melhor_x2), melhor_valor

# Parametros
dominio_x1 = (-200, 200)
dominio_x2 = (-200, 200)
iteracoes = 1000
sigma = 0.1  # Desvio padrao para a geracao dos candidatos

# Executa o LRS
(ponto_otimo, valor_otimo) = busca_local_aleatoria(funcao_objetivo, dominio_x1, dominio_x2, iteracoes, sigma)
print("Ponto otimo:", ponto_otimo)
print("Valor otimo:", valor_otimo)

# Preparando os dados para o grafico 3D
x = np.linspace(*dominio_x1, 400)
y = np.linspace(*dominio_x2, 400)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

# Criando o grafico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.6)
ax.scatter(*ponto_otimo, valor_otimo, color='r', s=100, label='Ponto Otimo')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
ax.legend()
plt.tight_layout()
plt.show()

