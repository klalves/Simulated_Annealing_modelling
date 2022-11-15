from math import exp
import random

class Simulated_Annealing:
    def __init__(self,
                 fitness_function,
                 neighbour_generator_function):

        print('Initializing Simulated_Annealing')

        self.fitness_function = fitness_function
        self.neighbour_generator_function = neighbour_generator_function

        self.current_sample = neighbour_generator_function()
        self.current_fitness = self.fitness_function(self.current_sample)
        self.best_sample = self.current_sample
        self.best_fitness = self.current_fitness

        self.temperature = 100
        self.k = 1

        self.log_iteration_count = []
        self.log_fitness = []
        self.log_best_fitness = []
        self.log_temperature = []
        self.log_sample = []
        self.log_best_sample = []

        self.iteration_count = 0

    def iterate(self):

        if(self.temperature >0):
            neighbour = self.neighbour_generator_function(self.current_sample)
            neighbour_fitness = self.fitness_function(neighbour)

            if(neighbour_fitness < self.best_fitness):
                self.best_sample = neighbour
                self.best_fitness = neighbour_fitness

            if(neighbour_fitness < self.current_fitness):
                self.current_sample = neighbour
                self.current_fitness = neighbour_fitness
            else:
                delta_fitness = neighbour_fitness - self.current_fitness
                acceptance_probabilty = exp(-(delta_fitness)/(self.temperature*self.k))

                if(random.random() < acceptance_probabilty):
                    self.current_sample = neighbour
                    self.current_fitness = neighbour_fitness
            
            self.temperature *= 0.994
        
        self.iteration_count += 1

        self.log_iteration_count.append(self.iteration_count)
        self.log_fitness.append(self.current_fitness)
        self.log_best_fitness.append(self.best_fitness)
        self.log_temperature.append(self.temperature)
        self.log_sample.append(self.current_sample)
        self.log_best_sample.append(self.best_sample)
