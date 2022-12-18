import itertools
import random
import json
# First networkx library is imported
# along with matplotlib
import networkx as nx
import matplotlib.pyplot as plt

class product:
    def __init__(self,val,wei,index):
        self.wei = wei
        self.val = val
        self.index = index

class GraphVisualization:

    def __init__(self):
        self.visual = []

    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.show()

def dec2bin(a, lenght):
    res = ""
    while a:
        res += str(a % 2)
        a //= 2
    while len(res) < lenght:
        res = "0" + res
    return res

def knapsackDP(w, weights, values, n):
    k = [[0 for i in range(w + 1)] for i in range(n + 1)]
    res = [0 for i in range(n)]

    for i in range(n + 1):
        for j in range(w + 1):
            if i == 0 or j == 0:
                k[i][j] = 0
            elif weights[i-1] <= j:
                k[i][j] = max(k[i - 1][j], k[i - 1][j - weights[i - 1]] + values[i - 1])
            else:
                k[i][j] = k[i - 1][j]

    i = n
    j = w
    while i > 0 and j > 0:
        if k[i - 1][j] != k[i][j]:
            res[i - 1] = 1
            j -= weights[i - 1]
            i -= 1
        else:
            i -= 1

    return res

def greedyKnapsack(w, weights, values, n):
    arr=[]
    res=[0]*n
    for i in range(n):
        arr.append(product(values[i],weights[i],i))
    arr.sort(key=lambda x: (x.val / x.wei), reverse=True)
    weightSum=0
    for i in range(n):
        if arr[i].wei + weightSum < w:
            res[arr[i].index]=1
            weightSum += arr[i].wei
    return res
    """
    ratio = []
    index = []
    res = [0]*n
    for i in range(n):
        index.append(i)
        ratio.append(values[i]/weights[i])

    #insertion sort
    for i in range(1, n):
        j = i
        while j > 0 and ratio[j] >= ratio[j - 1]:
            ratio[j], ratio[j - 1] = ratio[j - 1], ratio[j]
            values[j], values[j - 1] = values[j - 1], values[j]
            weights[j], weights[j - 1] = weights[j - 1], weights[j]
            index[j], index[j - 1] = index[j - 1], index[j]
            j -= 1
    weightSum = 0
    i = 0
    print(values)
    print(weights)
    while i < n and weightSum + weights[i] < w:
        weightSum += weights[i]
        res[index[i]] = 1
        i += 1

    return res
    """
def knapsackBruteForce(w, weights, values,n):
    res = ""
    max = -1
    for i in range(2**n - 1):
        weight = 0
        value = 0
        for index, j in enumerate(dec2bin(i,n)):
            if j == '1':
                value += values[index]
                weight += weights[index]
        if weight <= w and value > max:
            res = dec2bin(i,n)
            max = value
    list = []
    list[:0] = res
    for i in range(len(list)):
        list[i] = int(list[i])
    return list

def bruteForceTSP(dist,v):
    n = len(dist); l = list(range(n))
    minDist = 1000000; goodPerm = [0]*n
    for p in list(itertools.permutations(l)):
        d = 0
        for i in range(n-1):
            d += dist[p[i]][p[i+1]]
        d += dist[p[-1]][p[0]]
        if d <= minDist:
            minDist = d
            goodPerm = p
    return list(goodPerm), minDist

def greedyTSP(dist,n):
    visited = [0]*n
    visited[0] = 1
    route = [0]*n
    i = 0
    d = 0
    counter = 0
    while counter < n - 1:
        MIN = 9999999
        for j in range(n):
            if dist[i][j] < MIN and visited[j] == 0:
                MIN = dist[i][j]
                route[counter] = j
        d += MIN
        visited[route[counter]] = 1
        i = route[counter]
        counter += 1
    d += dist[route[counter - 1]][0]
    route[counter] = 0
    return route,tourLength(route,dist)

def geneticAlgorithmTSP(distanceMatrix, populationSize, numGenerations):
    # Initialize the population with random tours
    population = [random.sample(range(len(distanceMatrix)), len(distanceMatrix)) for _ in range(populationSize)]

    # Evolve the population over the given number of generations
    for generation in range(numGenerations):
        # Calculate the fitness of each tour
        fitness = [(1 / tourLength(tour, distanceMatrix), tour) for tour in population]
        fitness.sort(reverse=True)

        # Select the top tours for reproduction
        num_elites = populationSize // 10
        elites = [tour for _, tour in fitness[:num_elites]]

        # Perform crossover and mutation on the rest of the population
        offspring = []
        while len(offspring) < populationSize - num_elites:
            parent1, parent2 = random.sample(elites, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            offspring.append(child)

        # Replace the current population with the new one
        population = elites + offspring

    # Return the best tour found
    return min(population, key=lambda x: tourLength(x, distanceMatrix)),tourLength(min(population, key=lambda x: tourLength(x, distanceMatrix)),distanceMatrix)

def crossover(tour1, tour2):
    # Perform crossover by combining sections of the two tours
    start = random.randint(0, len(tour1) - 1)
    end = random.randint(start, len(tour1))
    new_tour = tour1[start:end]
    for city in tour2:
        if city not in new_tour:
            new_tour.append(city)
    return new_tour

def mutate(tour):
    # Perform mutation by swapping two cities in the tour
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
    return tour

def tourLength(tour, distance_matrix):
    # Calculate the length of the tour using the distance matrix
    length = 0
    for i in range(len(tour)):
        length += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
    return length

def generateValues(n,valMin,valMax):
    values = [0]*n
    for i in range(n):
        values[i] = random.randint(valMin,valMax)
        #values[i] = round(random.SystemRandom().uniform(valMin, valMax), 2)
    return values

def generateWeights(n, weiMin, weiMax):
    weights = [0] * n
    for i in range(n):
        weights[i] = random.randint(weiMin,weiMax)
        #weights[i] = round(random.SystemRandom().uniform(weiMin, weiMax), 2)
    return weights

def generateProducts(n, labels, products):
    available = []
    for i in range(n):
        x = random.randint(0, 8)
        y = random.randint(0, 4)
        available.append([])
        available[i].append(products[x][y])
        available[i].append(labels[x])
    return available

def convertToProducts(res, available):
    temp = []
    for i in range(n):
        if res[i]:
            temp.append(available[i])
    return temp

def vertices(available,labels):
    v = []
    for i in range(len(available)):
        v.append(labels.index(available[i][1])+1)
    v = list(dict.fromkeys(v))
    v.sort()
    return v

def generateDistance(v, coordinates):
    dist = [[0]*len(v) for _ in range(len(v))]
    for i in range(len(v)):
        for j in range(len(v)):
            dist[i - 1][j - 1] = abs(coordinates[i - 1][0] - coordinates[j - 1][0]) + abs(coordinates[i - 1][1] - coordinates[j - 1][1])
    return dist

def dataFromJson(filename):
    file = open(filename)
    data = json.load(file)
    values = data['values']
    weights = data['weights']
    available = data['available']
    return values,weights,available

dataFromJson("data.json")
# Driver code
G = GraphVisualization()
G.addEdge(1, 2)
G.addEdge(1, 4)
G.addEdge(2, 3)
G.addEdge(2, 5)
G.addEdge(3, 6)
G.addEdge(4, 5)
G.addEdge(4, 7)
G.addEdge(5, 8)
G.addEdge(5, 6)
G.addEdge(6, 9)
G.addEdge(7, 8)
G.addEdge(8, 9)
G.visualize()
coordinates = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
labels = ['Piekarnia i cukiernia', 'Owoce i warzywa', 'Przemysłówka', 'Chemia', 'Mięsny', 'Napoje', 'Nabiał', 'Alkohole', 'Reszta']
products = [['Chleb jasny', 'Chleb razowy', 'Bułka kajzerka', 'Babka piaskowa', 'Muffin'],
            ['Ziemniaki', 'Papryka czerwona', 'Pomidor malinowy', 'Banan', 'Jabłko'],
            ['Worki na śmieci', 'Rękawice jednorazowe', 'Zestaw noży kuchennych', 'Czajnik', 'Papier toaletowy'],
            ['Kapsułki do zmywarki', 'Proszek do prania', 'Domestos', 'Cif', 'Płyn do mycia naczyń'],
            ['Filety drobiowe', 'Udo drobiowe', 'Mięso wieprzowe', 'Mięso wołowe', 'Baranina'],
            ['Sok jabłkowy', 'Woda', 'Coca cola', 'Sok pomarańczowy', 'Woda gazowana'],
            ['Śmietana', 'Jogurt naturalny', 'Mleko', 'Jogurt skyr', 'Mozarella'],
            ['Piwo', 'Wino', 'Cydr', 'Whiskey', 'Szampan'],
            ['Czekolada', 'Mąka', 'Ryż jaśminowy', 'Makaron', 'Tuńczyk w puszce']]

n = random.randint(5, 10)
w = random.randint(20, 20)
values = generateValues(n, 3, 10)
weights = generateWeights(n, 2, 5)
available = generateProducts(n, labels, products)
res = knapsackDP(w,weights,values,n)
packed = convertToProducts(res,available)
v = vertices(available,labels)
dist = generateDistance(v,coordinates)
print(w,n)
print(available)
print(values)
print(weights)
print(greedyKnapsack(w,weights,values,n))
print(v)
print(dist)
print(bruteForceTSP(dist,v))
print(greedyTSP(dist,len(dist)))
print(geneticAlgorithmTSP(dist,50,100))

