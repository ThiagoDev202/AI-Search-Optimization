# – Trabalho AV1 - Busca/Otimização Meta-heurística -

# Funcao Objetivo: Minimizar f(x1, x2) = x^2 + x²^2
# Var: x1, x2 E [-100, 100]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Funcao objetivo
def funcao_objetivo(x1, x2):
    return x1**2 + x2**2

# Algoritmo de subida de encosta
def subida_encosta(funcao_objetivo, dominio, total_iteracoes, epsilon):
    # Gero um ponto inicial randomico
    x1, x2 = np.random.uniform(dominio[0], dominio[1], 2)
    # Calculo o valor da funcao_objetivo para o ponto_inicial
    valor_otimo = funcao_objetivo(x1, x2)
    
    for _ in range(total_iteracoes):
        # Gero um novo ponto_candidato_proximo ao ponto_atual
        candidato_x1 = x1 + np.random.uniform(-epsilon, epsilon)
        candidato_x2 = x2 + np.random.uniform(-epsilon, epsilon)
        # Garanto que o candidato esteja dentro dos limites pre-definidos pelo dominio
        candidato_x1 = np.clip(candidato_x1, dominio[0], dominio[1])
        candidato_x2 = np.clip(candidato_x2, dominio[0], dominio[1])
        # Calculo o valor da função objetivo para o candidato
        valor_candidato = funcao_objetivo(candidato_x1, candidato_x2)
        
        # Se o valor do candidato for melhor (ou seja, menor), atualizo o seu otimo
        if valor_candidato < valor_otimo:
            x1, x2 = candidato_x1, candidato_x2
            valor_otimo = valor_candidato
        else:
            # Caso nao haja melhora
            
            #reduzo o valor de epsilon
            epsilon *= 0.99

    return (x1, x2), valor_otimo

# Parametros pre definidos com base na problematica da questao
dominio = (-100, 100)
total_iteracoes = 1000
epsilon_inicial = 0.1  # Valor inicial para o tamanho do passo
rodadas = 100
solucoes_x1 = []
solucoes_x2 = []

# Executo o algoritmo de subida de encosta vartias vezes
for _ in range(rodadas):
    posicao_otima, valor_otimo = subida_encosta(funcao_objetivo, dominio, total_iteracoes, epsilon_inicial)
    solucoes_x1.append(posicao_otima[0])
    solucoes_x2.append(posicao_otima[1])

# Calcula a moda
def calcular_moda(lista_valores):
    valores, contagens = np.unique(lista_valores, return_counts=True)
    indice_moda = np.argmax(contagens)
    return valores[indice_moda]

moda_x1 = calcular_moda(solucoes_x1)
moda_x2 = calcular_moda(solucoes_x2)

# Imprimo a moda das solucoes
print("Moda das posições ótimas para x1:", moda_x1)
print("Moda das posições ótimas para x2:", moda_x2)

# Crio o grafico da funcao objetivo
x = np.linspace(-100, 100, 1000)
y = np.linspace(-100, 100, 1000)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=50, cstride=50, alpha=0.5, cmap='jet')

# Ploto a ulktima solucao otima encontrada
ax.scatter(solucoes_x1[-1], solucoes_x2[-1], valor_otimo, color='r', s=50, label='Melhor Solução')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Gráfico da Função Objetivo com a Melhor Solução')
ax.legend()
plt.show()






