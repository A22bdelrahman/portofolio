import random
import numpy as np

# Simple fitness function (e.g., optimize for a mock encryption strength)
def fitness(individual):
    return sum(individual)  # Mock: higher sum better

# Genetic Algorithm
population_size = 20
gene_length = 5
generations = 5

population = [[random.randint(0, 10) for _ in range(gene_length)] for _ in range(population_size)]

for gen in range(generations):
    population = sorted(population, key=fitness, reverse=True)
    next_gen = population[:10]  # Elitism
    for _ in range(10):
        parent1, parent2 = random.choices(population[:10], k=2)
        child = [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]
        if random.random() < 0.1:
            child[random.randint(0, gene_length-1)] += random.randint(-1, 1)
        next_gen.append(child)
    population = next_gen

best = population[0]
print(f"Best individual after {generations} gens: {best}, fitness: {fitness(best)}")