# – Trabalho AV1 - Busca/Otimização Meta-heurística -

# Funcao Objetivo: Minimizar f(x1, x2) = x^2 + x²^2
# Var: x1, x2 E [-100, 100]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Funcao objetivo
def funcao_objetivo(x1, x2):
    return x1**2 + x2**2

# Logica do algoritmo Local Random Search (LRS)
def busca_local_aleatoria(funcao_objetivo, dominio, total_iteracoes):
    # Inicializacao da melhor posicao e valor encontrado (Inicializo a variavel melhor_posicao com um valor 
    # extremamente alto para garantir que qualquer valor real encontrado pela funcao objetivo seja menor)
    melhor_posicao = None
    melhor_valor = np.inf
    
    # Loop principal do LRS
    for _ in range(total_iteracoes):
        # Gero um ponto candidato randomico dentro do dominio pré definido
        x1_candidato = np.random.uniform(dominio[0], dominio[1])
        x2_candidato = np.random.uniform(dominio[0], dominio[1])
        
        # Avalia o candidato pela funcao objetivo
        valor_candidato = funcao_objetivo(x1_candidato, x2_candidato)
        
        # Caso encontre um melhor candidato
        if valor_candidato < melhor_valor:
            # Atualizo a melhor posicao e valor
            melhor_posicao = (x1_candidato, x2_candidato)
            melhor_valor = valor_candidato

    return melhor_posicao, melhor_valor

# Parametros do algoritmo com base nos requisitos da questao
dominio = (-100, 100)  # Limites para as varriaveis x1 e x2
total_iteracoes = 1000 
rodadas = 100
# Lista (inicialmente vazia) no qual armazeno as melhores posições encontradas em cada rodada
resultados = []

# Executo o algoritmo LRS para o numero especificado de rodadas, que no caso sao 100 (De acordo com a questao/problematica)
for _ in range(rodadas):
    posicao_otima, valor_otimo = busca_local_aleatoria(funcao_objetivo, dominio, total_iteracoes)
    # Armazeno a melhor posicao de cada rodada
    resultados.append(posicao_otima)  


# def calc_moda(valores):
#     freq = {}
#     for val in valores:
#         if val in freq:    
#             freq[val] += 1
#         else:
#             freq[val] = 1
#     moda = max(freq, key=freq.get)
#     return moda
    

# Calc moda
def calc_moda(valores):
    valores, cont = np.unique(valores, return_counts=True)
    indice_moda = np.argmax(cont)
    return valores[indice_moda]

# Calculo a moda das posicoes otimas encontradas
# Moda = Posicao encontrada como otima nas rodadas
valores_x1 = [pos[0] for pos in resultados]
valores_x2 = [pos[1] for pos in resultados]
moda_x1 = calc_moda(valores_x1)
moda_x2 = calc_moda(valores_x2)

print("Moda das posicoes otimas para x1:", moda_x1)
print("Moda das posicoes otimas para x2:", moda_x2)

# Crio o grafico da funcao objetivo
x = np.linspace(dominio[0], dominio[1], 1000)
y = np.linspace(dominio[0], dominio[1], 1000)
X, Y = np.meshgrid(x, y)
Z = funcao_objetivo(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none', alpha=0.8)
ax.scatter(moda_x1, moda_x2, funcao_objetivo(moda_x1, moda_x2), color='r', s=50, label='Moda das Solucoes: ')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Funcao Objetivo e Moda das Solucoes: ')
ax.legend()
plt.show()
