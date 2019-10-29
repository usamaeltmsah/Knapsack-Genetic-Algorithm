import random


# Get Weights and benefits of knapsack
def get_knapsack_w_b(n):
    w_b = list()
    for j in range(int(n)):
        # Get Value and Weight in one line separated by space
        w_b.append(list(map(int, input().split())))

    return w_b


def read_input_file(file_path):
    lines = []
    inputs = []
    # "input_example.txt", "r"
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    c = int(lines[0])
    inputs.append(c)
    lines.pop(0)
    for i in range(c):
        knapsack_w_b = list()
        n = int(lines[0])
        inputs.append(n)
        lines.pop(0)
        size = int(lines[0])
        inputs.append(size)
        lines.pop(0)
        for i in range(int(n)):
            # Get Value and Weight in one line separated by space
            knapsack_w_b.append(list(map(int, lines[0].split())))
            lines.pop(0)
        inputs.append(knapsack_w_b)
    return (inputs)


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


# Calculate the probability of every chromosome's weight
def roulette_wheel_calc(fitness_w_b):
    # Weight fitness weight = fitness[0]
    total_weight_fitness = sum(fitness_w_b[0])
    fitness_prop = []
    fit_val = 0
    for weight in fitness_w_b[0]:
        fit_val += weight / total_weight_fitness
        fitness_prop.append(fit_val)

    return fitness_prop


# This selection is implemented using Roulette Wheel
# weights_prop: gets from implementation of roulette_wheel_calc() function
def get_index_to_select(fitness_w_b, pop_size):
    weights_prop = roulette_wheel_calc(fitness_w_b)
    r = random.uniform(0, 1)
    for i in range(pop_size):
        if r <= weights_prop[i]:
            # return the index of the selected chromosome
            return i


def apply_crossover(c1, c2):
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


def select_chromosome_for_crossover(population, fitness):
    pop_size = len(population)
    i = get_index_to_select(fitness, pop_size)
    c = population[i]
    population.pop(i)
    fitness[0].pop(i)

    return c


def cross_over(population, fitness):
    pop_size = len(population)
    new_generation = []
    # We will make crossover on half of the population
    # So we will apply this 1/4 times as we select 2 chromosomes every time
    limit = int(pop_size / 4)
    for i in range(limit):
        # Randomly select 2 chromosomes using roulette wheel to apply crossover on them
        # Pop the selected items from the population
        c1 = select_chromosome_for_crossover(population, fitness)
        c2 = select_chromosome_for_crossover(population, fitness)
        # Apply crossover to generate new generation
        os = apply_crossover(c1, c2)
        new_generation.append(os[0])
        new_generation.append(os[1])
    return new_generation


# append the new generation to the new ones
def get_new_population_after_crossover(half_pop, new_generation):
    return half_pop + new_generation


def flip_bit(bit):
    return int(not bit)


# Make mutation on random bit of the chromosome
def mutation(chromosome):
    # pm: fixed parameter in GA [0.001 → 0.1]
    # Probability that mutation will occur for a gene/bit in some chromosomes
    pm = 0.05
    chromosome_len = len(chromosome)

    # Check which bits should be flipped
    for i in range(chromosome_len):
        r = random.uniform(0, 1)
        if r <= pm:
            chromosome[i] = flip_bit(chromosome[i])

    return chromosome


def apply_mutation_on_population(population, knapsack_w_b, size):
    new_pop = []
    # TODO: If the fitness of the new chromosome better, replace ..
    for chromosome in population:
        old_ben = evaluate_single_chromosome(knapsack_w_b, chromosome)[1]
        new_chromosome = mutation(chromosome)
        new_fit = evaluate_single_chromosome(knapsack_w_b, new_chromosome)
        new_weight = new_fit[0]
        new_ben = new_fit[1]
        # If the new chromosome is better append it, otherwise append the new one
        if new_ben > old_ben and new_weight <= size:
            new_pop.append(new_chromosome)
        else:
            new_pop.append(chromosome)

    return new_pop


def sort_fitness_according_to_benefits(fitness_w_b):
    ben, weight = zip(*sorted(zip(fitness_w_b[1], fitness_w_b[0]), reverse=True))

    return list(weight), list(ben)


def sort_population_according_to_benefit(population, fitness_w_b):
    # Sort Benefits Descending
    # sorted_weight = [benefit for _, benefit in sorted(zip(fitness_w_b[1], fitness_w_b[0]))]
    ben, sorted_population = zip(*sorted(zip(fitness_w_b[1], population), reverse=True))

    return sorted_population


# This function takes the sorted population as parameter
def get_best_chromosome(filtered_pop, knapsack_w_b, knapsack_size):
    fitness_w_b = evaluate_fitness(knapsack_w_b, filtered_pop)
    sorted_fit = sort_fitness_according_to_benefits(fitness_w_b)
    fitness_w_b = sorted_fit

    best_chromosome = []
    # TODO: If the weight > size of knapsack, make its benefit = 0
    for i in range(len(fitness_w_b[1])):
        # If the weight > size of knapsack, make its benefit = 0
        if fitness_w_b[0][i] > knapsack_size:
            fitness_w_b[1][i] = 0
        else:
            best_chromosome = filtered_pop[i]
            break

    return best_chromosome


def apply_GA_On_Knapsack(knapsack_w_b, n_items, size):
    pop = generate_population(n_items*int(n_items/2), n_items)
    # print(pop)
    fitness_w_b = evaluate_fitness(knapsack_w_b, pop)
    # print(fitness_w_b)
    filtered_pop = filter_population(knapsack_w_b, pop, fitness_w_b, size)
    # print(filtered_pop)
    new_gen = cross_over(filtered_pop, fitness_w_b)
    # print(new_gen)
    pop_after_crossover = get_new_population_after_crossover(filtered_pop, new_gen)
    # print(pop_after_crossover)
    new_fit = evaluate_fitness(knapsack_w_b, pop_after_crossover)
    sorted_fit = sort_fitness_according_to_benefits(new_fit)
    # print(sorted_fit)
    mutated_pop = apply_mutation_on_population(pop_after_crossover, knapsack_w_b, size)
    sorted_pop = sort_population_according_to_benefit(mutated_pop, sorted_fit)
    # print(sorted_pop)
    best_chromosome = get_best_chromosome(sorted_pop, knapsack_w_b, size)
    # print(best_chromosome)
    best_chromosome_benefit = evaluate_single_chromosome(knapsack_w_b, best_chromosome)[1]
    # print(best_chromosome_benefit)

    # Make till 8 Generations
    for i in range(8):
        fitness_w_b = evaluate_fitness(knapsack_w_b, sorted_pop)
        filtered_pop = filter_population(knapsack_w_b, sorted_pop, fitness_w_b, size)
        new_gen = cross_over(filtered_pop, fitness_w_b)
        new_gen = get_new_population_after_crossover(filtered_pop, new_gen)
        new_gen = apply_mutation_on_population(new_gen, knapsack_w_b, size)
        new_fit = evaluate_fitness(knapsack_w_b, new_gen)
        sorted_pop = sort_population_according_to_benefit(new_gen, new_fit)
        new_best_chromosome = get_best_chromosome(sorted_pop, knapsack_w_b, size)
        new_best_chromosome_benefit = evaluate_single_chromosome(knapsack_w_b, new_best_chromosome)[1]
        if new_best_chromosome_benefit > best_chromosome_benefit:
            best_chromosome = new_best_chromosome
            # Append the best chromosome to the new generation
            sorted_pop = list(sorted_pop)
            sorted_pop[len(sorted_pop) - 1] = best_chromosome
            best_chromosome_benefit = new_best_chromosome_benefit
        # print(evaluate_fitness(knapsack_w_b, sorted_pop))
    return (best_chromosome_benefit)


# Get input from keyboard
# c = int(input("Number of test cases: "))
# n = int(input("Number of items: "))
# size = int(input("Size of knapsack: "))
# # List of Values and Weights
# print("Benefits and Weights:-")
# knapsack_w_b = get_knapsack_w_b(n)
# apply_GA_On_Knapsack(knapsack_w_b, n, size)

inputs = read_input_file("input_example.txt")
c = inputs[0]
inputs.pop(0)
for i in range(c):
    n = inputs[0]
    inputs.pop(0)
    size = inputs[0]
    inputs.pop(0)
    knapsack_w_b = inputs[0]
    inputs.pop(0)
    best = apply_GA_On_Knapsack(knapsack_w_b, n, size)
    print(f"Case: {i+1} {best}")
