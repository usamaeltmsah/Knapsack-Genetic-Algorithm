import random



def get_knapsack_v_w(n):
    v_w = list()
    for j in range(int(n)):
        # Get Value and Weight in one line separated by space
        v_w.append(list(map(int, input().split())))

    return v_w


# Chromosomes Encoding
# Chromosome Length = n (Number of items)
def generate_chromosomes_genes(n):
    genes = []
    for i in range(int(n)):
        r = random.uniform(0, 1)
        if r <= 0.5:
            r = 0
        else:
            r = 1

        genes.append(r)

    return genes


# Generate Population of genes (elements)
def generate_population(m, n):
    pop = []
    for i in range(int(m)):
        pop.append(generate_chromosomes_genes(n))
    return pop


# Fitness Function (Which elements will be accepted and which will be rejected)
# V_m: is the 2D list which carry the knapsack,
# n: is the weights and values list size
def fitness(v_w, pop):
    weights = []
    values =  []

    for chromosome in pop:
        i = 0
        weight_of_knapsack = 0
        val_of_knapsack = 0

        for gene in chromosome:
            weight_of_knapsack += gene * v_w[i][0]
            val_of_knapsack += gene * v_w[i][1]
            i += 1

        weights.append(weight_of_knapsack)
        values.append(val_of_knapsack)

    return weights, values


# Select the chromosomes that can be fitted in the knapsack
# size: is Knapsack size
def evaluate_fitness(pop, weights, size):
    accepted = []
    for i in range(len(weights)):
        if weights[i] <= size:
            accepted.append(pop[i])
    return accepted


# This function will determine if there is feasible solutions without need to crossover and mutation
def feasible_solutions(pop, weights, size):
    feasible = []
    for i in range(len(weights)):
        if weights[i] == size:
            feasible.append(pop[i])
    return feasible
