# – Trabalho AV1 - Busca/Otimização Meta-heurística -

# Funcao Objetivo: Minimizar f(x1, x2) = x^2 + x²^2
# Var: x1, x2 E [-100, 100]

# Entendendo a diferença entre e ambos os Algoritimos estudados durante a execucao do trab av1 ->

# O GRS gera um novo ponto candidato completamente aleatorio dentro do dominio em cada iteracao.
# O mesmo nao possui nocao de "proximidade" ou "vizinhança" como no algoritimo 'LRS'. 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Func Obj
def funcao_objetivo(x1, x2):
    return x1**2 + x2**2

def busca_aleatoria_global(funcao_objetivo, dominio, total_iteracoes):
    melhor_posicao = None
    melhor_valor = np.inf

    for _ in range(total_iteracoes):
        # Gero um ponto candidato completamente aleatorio denteo do cominico
        x1_candidato = np.random.uniform(dominio[0], dominio[1])
        x2_candidato = np.random.uniform(dominio[0], dominio[1])
        # A partir da funcao obejetivo avalio a meu candidato
        valor_candidato = funcao_objetivo(x1_candidato, x2_candidato)

        # Se encontrar um candidato com valor melhor
        if valor_candidato < melhor_valor:
            #atualizo a melhor posicao e valor
            melhor_valor = valor_candidato
            melhor_posicao = (x1_candidato, x2_candidato)

    return melhor_posicao, melhor_valor


# Parametroi da questao
dominio = (-100, 100)
total_iteracoes = 1000
rodadas = 100
resultados = []

# realizo a busca eleatoria diversaa vezes
for _ in range(rodadas):
    posicao_otima, valor_otimo = busca_aleatoria_global(funcao_objetivo, dominio, total_iteracoes)
    resultados.append(posicao_otima)


# Calculo a moda das posicoes otima encontradas
def calc_moda(valores):
    valores, contagens = np.unique(valores, return_counts=True)
    indice_moda = np.argmax(contagens)
    return valores[indice_moda]


moda_x1 = calc_moda([pos[0] for pos in resultados])
moda_x2 = calc_moda([pos[1] for pos in resultados])

print("Moda das posicoes otimas para x1: ", moda_x1)
print("Moda das posicoes otimas para x1: ", moda_x2)

# graf func objetivo
x = np.linspace(dominio[0], dominio[1], 1000)
y = np.linspace(dominio[0], dominio[1], 1000)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none', alpha=0.8)
ax.scatter(moda_x1, moda_x2, funcao_objetivo(moda_x1, moda_x2), color='r', s=50, label='Moda das Solucoes')
ax.set_xlabel('x1')
ax.set_ylabel('y2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Funcao Objetivo e Moda das Solucoes')
ax.legend()
plt.show()