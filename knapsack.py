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
# If the gene is 1 that means that this item will be included in the knapsack
def generate_single_chromosome_genes(n_items):
    genes = []
    for i in range(int(n_items)):
        r = random.uniform(0, 1)
        if r <= 0.5:
            r = 1
        else:
            r = 0
        genes.append(r)

    return genes

# Generate Population of genes (elements)
def generate_population(pop_size, n_items):
    pop = []
    for i in range(int(pop_size)):
        chromosome = generate_single_chromosome_genes(n_items)
        # TODO: Check if it's a good chromosome
        #  (Size of knapsack should be greater than or equal this chromosome fitness)

        # All population items should be different
        while chromosome in pop:
            chromosome = generate_single_chromosome_genes(n_items)
        pop.append(chromosome)
    return pop


# Get the fitness of one single chromosome
def evaluate_single_chromosome(knapsack_w_b, chromosome):
    weight = 0
    benefit = 0
    i = 0
    for gene in chromosome:
        weight += gene * knapsack_w_b[i][0]
        benefit += gene * knapsack_w_b[i][1]
        i += 1

    return weight, benefit


# knapsack_w_b: is the 2D list which carry the knapsack,
# population: is the population of chromosomes
def evaluate_fitness(knapsack_w_b, population):
    weights = []
    benefits = []
    for chromosome in population:
        chromosome_fit = evaluate_single_chromosome(knapsack_w_b, chromosome)
        weights.append(chromosome_fit[0])
        benefits.append(chromosome_fit[1])

    return weights, benefits


# Filter Population Function (Which elements will be accepted and which will be rejected)
# Rejected chromosomes will be replaced by a new one
def filter_population(knapsack_w_b, population, fitness, knapsack_size):
    i = 0
    accepted_chromosomes = []
    for chromosome_fit in fitness[0]:
        if chromosome_fit <= knapsack_size:
            accepted_chromosomes.append(population[i])

        else:
            # Generate new chromosome while its fitness is greater than knapsack size
            c = generate_single_chromosome_genes(len(population[0]))
            j = 0
            # Try this just 5 times to avoid infinity loop
            while((evaluate_single_chromosome(knapsack_w_b, c)[0] > knapsack_size)) and j <= 5:
                c = generate_single_chromosome_genes(len(population[0]))
                j += 1
            accepted_chromosomes.append(c)
        i += 1
    return accepted_chromosomes


# This function will determine if there is feasible solutions without need to crossover and mutation
def feasible_solutions(pop, weights, size):
    feasible = []
    for i in range(len(weights)):
        if weights[i] == size:
            feasible.append(pop[i])
    return feasible


# Calculate the probability of every chromosome's weight
def roulette_wheel_calc(fitness_w_b):
    total_weight_fitness = sum(fitness_w_b[0])
    fitness_prop = []
    fit_val = 0
    for weight in fitness_w_b[0]:
        fit_val += weight / total_weight_fitness
        fitness_prop.append(fit_val)

    return fitness_prop


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
