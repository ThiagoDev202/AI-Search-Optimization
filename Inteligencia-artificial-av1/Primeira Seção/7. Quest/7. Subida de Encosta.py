# – Trabalho AV1 - Busca/Otimização Meta-heurística -


# Importando as bibliotecas necessarias
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a funcao objetivo conforme a questao
def funcao_objetivo(x1, x2):
    # Definindo a funcao objetivo conforme o enunciado da questao
    return -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1/2 + 47))) - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))

# Implementando o algoritmo de Subida de Encosta
def subida_encosta(dominio_x1, dominio_x2, total_iteracoes, passo):
    # Inicializando x1 e x2 com valores aleatorios dentro dos limites do dominio
    x1 = np.random.uniform(*dominio_x1)
    x2 = np.random.uniform(*dominio_x2)
    # Calculando o valor otimo inicial
    valor_otimo = funcao_objetivo(x1, x2)

    # Loop de iteracoes
    for _ in range(total_iteracoes):
        # Gerando um novo ponto candidato dentro dos limites do dominio
        candidato_x1 = np.clip(x1 + np.random.uniform(-passo, passo), *dominio_x1)
        candidato_x2 = np.clip(x2 + np.random.uniform(-passo, passo), *dominio_x2)
        # Calculando o valor da funcao objetivo para o novo candidato
        valor_candidato = funcao_objetivo(candidato_x1, candidato_x2)

        # Se o valor do candidato for melhor, atualize o ponto otimo
        if valor_candidato < valor_otimo:
            x1, x2, valor_otimo = candidato_x1, candidato_x2, valor_candidato

    # Retornando o ponto otimo e o valor otimo
    return (x1, x2), valor_otimo

# Definindo os parametros do algoritmo
dominio_x1 = (-200, 200)
dominio_x2 = (-200, 200)
total_iteracoes = 1000
passo = 0.1  # Tamanho do passo para explorar a vizinhanca

# Executando o algoritmo de Subida de Encosta
(ponto_otimo, valor_otimo) = subida_encosta(dominio_x1, dominio_x2, total_iteracoes, passo)
print("Ponto otimo:", ponto_otimo)
print("Valor otimo:", valor_otimo)

# Criando os dados para o grafico
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



