# Knapsack-Genetic-Algorithm

Using Genetic Algorithm to solve the knapsack problem.

We want to **maximize** the benefits of the items to fit our knapsack.

### Steps
1. Generate Population of chromosomes.
2. Filter the populatin, so if there're any chromosomes that have fit is greater than the knapsack size so new chromosome will be generated.
3. Apply cross-over on half of the population.
4. Apply mutation on the whole population.
5. Again, evaluate the fitness.
6. Sort the population according to benefits.
7. Get the best chromosome.
