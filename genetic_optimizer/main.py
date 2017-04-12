import inspect

import numpy as np


class GeneticOptimizer:
    def __init__(self, function_to_be_optimized,
                 search_space_boundaries,
                 population_size=1000,
                 crossover_rate=0.3,
                 mutation_rate=0.3):

        self.func = function_to_be_optimized
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        search_space_length = np.squeeze(np.diff(search_space_boundaries))
        self.population = np.random.rand(self.population_size, *search_space_length.shape)
        self.population *= search_space_length
        
        self.population += search_space_boundaries[None, :, 0]

    def normalized_fitness(self):
        fitness_values = self.func(self.population)
        return fitness_values / fitness_values.sum()

    def crossover(self, normalized_fitness):
        sorted_args = np.argsort(normalized_fitness)
        cumsum = np.cumsum(normalized_fitness[sorted_args])

        trial = np.random.random_sample(normalized_fitness.shape)

        print(sorted_args)
        print(cumsum)
        print(crossover_len)

    def mutate(self):
        pass


def main():
    def testfun(x):
        return np.sum(x**2, axis=-1)

    go = GeneticOptimizer(testfun, np.array([[-5, 5], [-10, 10]]), population_size=10)
    fitness = go.normalized_fitness()
    go.crossover(fitness)


if __name__ == "__main__":
    main()
    
