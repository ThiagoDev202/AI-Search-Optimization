# – Trabalho AV1 - Busca/Otimizacao Meta-heurística -


import numpy as np
import matplotlib.pyplot as plt

# Funcao para calcular o numero de ataques entre rainhas
def calcular_ataques(estado):
    ataques = 0
    n = len(estado)
    for i in range(n):
        for j in range(i+1, n):
            # Rainhas na mesma linha ou em diagonais se atacam
            if estado[i] == estado[j] or abs(estado[i] - estado[j]) == j - i:
                ataques += 1
    return ataques

# Funcao objetivo f(x) = 28 - h(x)
def funcao_objetivo(estado):
    h = calcular_ataques(estado)
    return 28 - h

# Funcao de decaimento de temperatura
def decaimento_temperatura(temperatura_inicial, iteracao, k):
    return temperatura_inicial / (1 + k * iteracao)

# Funcao para perturbar a solucao
def perturbar_solucao(estado):
    n = len(estado)
    i, j = np.random.choice(n, 2, replace=False)
    estado[i], estado[j] = estado[j], estado[i]
    return estado

# Algoritmo de Tempera Simulada
def temperatura_simulada(temperatura_inicial, k, max_iteracoes):
    estado_atual = np.arange(8)
    np.random.shuffle(estado_atual)
    custo_atual = funcao_objetivo(estado_atual)
    melhor_estado = estado_atual.copy()
    melhor_custo = custo_atual
    
    custos = []  # Para plotar o grafico

    for iteracao in range(max_iteracoes):
        temperatura = decaimento_temperatura(temperatura_inicial, iteracao, k)
        novo_estado = perturbar_solucao(estado_atual.copy())
        custo_novo_estado = funcao_objetivo(novo_estado)
        delta_e = custo_novo_estado - custo_atual

        if delta_e > 0 or np.random.uniform() < np.exp(delta_e / temperatura):
            estado_atual = novo_estado
            custo_atual = custo_novo_estado
            if custo_novo_estado > melhor_custo:
                melhor_estado = novo_estado
                melhor_custo = custo_novo_estado
        
        custos.append(custo_atual)

        if melhor_custo == 28:  # Encontrou solucao sem ataques
            break

    return melhor_estado, melhor_custo, iteracao, custos

# Parametros da problematica
temperatura_inicial = 30
k = 0.1
max_iteracoes = 50000
solucoes_encontradas = set()
solucoes_desejadas = 92

# Executa o algoritmo enquanto nao encontrar todas as 92 solucaes unicas
while len(solucoes_encontradas) < solucoes_desejadas:
    estado, custo, iteracao, custos = temperatura_simulada(temperatura_inicial, k, max_iteracoes)
    solucoes_encontradas.add(tuple(estado))

# Avaliacao do custo computacional
iteracoes_totais = len(solucoes_encontradas) * max_iteracoes

# Plot do grafico de numero de ataques por iteracao
plt.figure(figsize=(10, 5))
plt.plot(custos)
plt.title('Evolucao do Numero de Ataques no Problema das 8 Rainhas')
plt.xlabel('Iteracaes')
plt.ylabel('Numero de ataques')
plt.show()

# Impressao do resultado
print(f"Solucaes unicas encontradas: {len(solucoes_encontradas)} / {solucoes_desejadas}")
print(f"Iteracaes totais realizadas: {iteracoes_totais}")




