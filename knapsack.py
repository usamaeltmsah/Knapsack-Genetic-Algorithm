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
