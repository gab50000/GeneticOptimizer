import numpy as np
import matplotlib.pyplot as plt


class GeneticOptimizer:
    def __init__(self, function_to_be_optimized,
                 dimensionality,
                 search_space_boundaries,
                 population_size=1000,
                 crossover_rate=0.3,
                 mutation_rate=0.3,
                 mutation_factor=1.0):
        """
        Genetic Optimizer
        
        Optimizes a function R^n -> R using a stochastical parameter search.
        This involves crossover where new solutions are created from linear combinations
        of old ones, and mutation.
        
        Parameters
        ----------
        function_to_be_optimized: function
            Function to be optimized
        
        dimensionality: int
            Input-Dimensionality of the function to be optimized
            
        search_space_boundaries: array_like
            The space in which initial solutions will be placed
            
        population_size: int
            Number of simultaneous trial parameters
            
        crossover_rate: float
            Ratio of solutions that will be created as a linear combination of two parent solutions
            
        mutation_rate: float
            Ratio of solutions that will be mutated by adding a normal distributed random vector
        
        mutation_factor: float
            Determines by how much mutation will change an existing parameter
        """

        self.func = function_to_be_optimized
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.crossover_number = int(crossover_rate * population_size)
        self.mutation_rate = mutation_rate
        self.mutation_number = int(mutation_rate * population_size)
        self.mutation_factor = mutation_factor

        search_space_length = np.squeeze(np.diff(search_space_boundaries))
        self.population = np.random.rand(self.population_size, *search_space_length.shape)
        self.population *= search_space_length
        
        self.population += search_space_boundaries[None, :, 0]

    def normalized_fitness(self):
        fitness_values = self.func(self.population)
        fitness_values = np.where(fitness_values < 0, 1 - fitness_values, 1 / (1 + fitness_values))
        return fitness_values / fitness_values.sum()

    def crossover(self, normalized_fitness):
        # Sort ascending
        sorted_args = np.argsort(normalized_fitness)
        cumsum = np.cumsum(normalized_fitness[sorted_args])

        trial = np.random.uniform(0, cumsum[-1], size=self.crossover_number*2)
        parent1, parent2 = np.split(np.searchsorted(cumsum, trial), 2)

        # Generate offspring
        a, b = np.random.rand(2)
        self.population[sorted_args[:self.crossover_number]] = \
            (a * self.population[parent1] + b * self.population[parent2]) / (a + b)

    def mutate(self):
        pass


def main():
    def testfun(x):
        return np.sum(x**2, axis=-1)

    go = GeneticOptimizer(testfun, np.array([[-5, 5], [-10, 10]]), population_size=100)
    for i in range(1000):
        fitness = go.normalized_fitness()
        go.crossover(fitness)
        print(np.max(fitness))
        print(go.population[np.argmax(fitness)])
        if i % 10 == 0:
            plt.hist(fitness)
            plt.show()


if __name__ == "__main__":
    main()
    
