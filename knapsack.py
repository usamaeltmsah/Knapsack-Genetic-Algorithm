import random


# Get Weights and benefits of knapsack
def get_knapsack_w_b(n):
    w_b = list()
    for j in range(int(n)):
        # Get Value and Weight in one line separated by space
        w_b.append(list(map(int, input().split())))

    return w_b


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
# m: is population size
def generate_population(m, n):
    pop = []
    for i in range(int(m)):
        r = generate_chromosomes_genes(n)
        # All population items should be different
        while r in pop:
            r = generate_chromosomes_genes(n)
        pop.append(r)
    return pop


# Fitness Function (Which elements will be accepted and which will be rejected)
# V_m: is the 2D list which carry the knapsack,
# n: is the weights and values list size
def fitness(v_w, pop):
    weights = []
    values = []

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


def roulette_wheel_calc(pop, weights):
    total_fitness = sum(weights)
    fitness_values = []
    for chromosome in pop:
        fit_val = chromosome / total_fitness
        if fit_val > 0:
            fitness_values.append(fit_val)

    return fitness_values


# This selection is implemented using Roulette Wheel
def selection(fitness_values, pop_size):
    if isinstance(fitness_values, list):
        # Sort Descending
        fitness_values.sort(reverse=True)
    i = 0
    # Make the fitness values size match the required population size
    # We will repeat the values have the highest probability
    # i % pop_size: Repeat again and again while the two sizes doesn't match
    while len(fitness_values) < pop_size:
        fitness_values.append(fitness_values[i % pop_size])
        i += 1

    return fitness_values


# We will need to select two chromosomes randomly to apply crossover on
# Single Point Crossover
def cross_over(c1, c2):
    os1 = []
    os2 = []
    siz = len(c1)
    # Generate random single point for crossover
    r = int(random.uniform(1, siz))
    for i in range(int(r)):
        os1.append(c1[i])
        os2.append(c2[i])

    for j in range(int(r), siz):
        os1.append(c2[j])
        os2.append(c1[j])

    return os1, os2


def get_pop_after_crossover(fitness_values, c1_ind, c2_ind, new_fitness):
    new_generation = []
    siz = len(fitness_values)
    for i in range(siz):
        if i != c1_ind and i != c2_ind:
            new_generation.append(fitness_values[i])
    new_generation.append(new_fitness)

    return new_generation


def flip_bit(bit):
    if bit == 0:
        bit = 1
    else:
        bit = 0
    return bit


# Make mutation on random bit of the chromosome
def mutation(chromosome):
    r = int(random.uniform(0, len(chromosome)))

    chromosome[r] = flip_bit(chromosome[r])

    return chromosome


c = int(input("Number of test cases: "))
n = int(input("Number of items: "))
size = int(input("Size of knapsack: "))

# List of Values and Weights
print("Values and Weights:-")
# get_knapsack_v_w(c, n)
l = get_knapsack_v_w(n)

x = generate_population(10, n)
print(x)
w_v = fitness(l, x)

print(w_v)

c = evaluate_fitness(x, w_v[1], size)

print(feasible_solutions(x, w_v[1], size))
