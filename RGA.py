import numpy as np
import math
import random
import matplotlib.pyplot as plt
import sys

plt.style.use('fivethirtyeight')


class RGA():
    def __init__(self, pop_shape, pc=0.8, pm=0.01, max_round=100, low=[0, 0], high=[0, 0], crossover=0):
        self.pop_shape = pop_shape
        self.pc = pc
        self.pm = pm
        self.max_round = max_round
        self.low = low
        self.high = high
        self.crossover = crossover

    def initialization(self):  # initialize first population
        # random float number [0, 1]
        pop = np.random.rand(self.pop_shape[0], self.pop_shape[1])
        return pop

    def fixChoromRange(self, number, m):
        match m:
            case 0:
                real = self.low[0] + (number * (self.high[0] - self.low[0]))
                return real
            case 1:
                real = self.low[1] + (number * (self.high[1] - self.low[1]))
                return real

    def chromosomeDecode(self, pop):
        l = 0
        gen = []
        for i in range(0, self.pop_shape[0]):
            for j in range(0, 2):
                l = pop[i][j]
                gen.append(self.fixChoromRange(l, j))
        return np.array(gen).reshape(pop.shape[0], 2)

    def crossOver(self, ind1, ind2):
        new_1, new_2 = 0, 0

        p_pc = np.random.random_sample(1)
        if p_pc < self.pc:  # doing crossover
            match self.crossover:
                case 0:
                    landa1 = np.random.random_sample(1)
                    landa2 = 1 - landa1
                case 1:
                    landa1 = random.uniform(-1, 1)
                    landa2 = random.uniform(-1, 1)
            new_1 = (landa1 * ind1) + (landa2 * ind2)
            new_2 = (landa2 * ind1) + (landa1 * ind2)
        else:  # Transfer without crossover
            new_1 = ind1
            new_2 = ind2
        return new_1, new_2

    def normFixRange(self, pop):
        for i in range(0, self.pop_shape[0]):
            for j in range(0, 2):
                if j == 0:
                    if pop[i][j] < self.low[0] or pop[i][j] > self.high[0]:
                        pop[i][j] = pop[i][j] / self.high[0]
                        if pop[i][j] < 0:
                            pop[i][j] = sys.float_info.epsilon
                        elif pop[i][j] > 1:
                            pop[i][j] = 1 - sys.float_info.epsilon
                        pop[i][j] = self.low[0] + \
                            (pop[i][j] * (self.high[0] - self.low[0]))
                elif j == 1:
                    if pop[i][j] < self.low[1] or pop[i][j] > self.high[1]:
                        pop[i][j] = pop[i][j] / self.high[1]
                        if pop[i][j] < 0:
                            pop[i][j] = sys.float_info.epsilon
                        elif pop[i][j] > 1:
                            pop[i][j] = 1 - sys.float_info.epsilon
                        pop[i][j] = self.low[1] + \
                            (pop[i][j] * (self.high[1] - self.low[1]))
        return pop

    def mutation(self, pop):
        # Calculate the number of bits that must mutation
        num_mut = math.ceil(self.pm * pop.shape[0] * pop.shape[1])
        for m in range(0, num_mut):
            i = np.random.randint(0, pop.shape[0])
            j = np.random.randint(0, pop.shape[1])
            pop[i][j] = pop[i][j] + (np.random.normal(0, 0.5, 1))
        fixed_pop = self.normFixRange(pop)
        return fixed_pop

    def fitnessFunc(self, real_val):
        fitness_val = 21.5 + \
            real_val[0]*np.sin(4*np.pi*real_val[0]) + \
            real_val[1]*np.sin(20*np.pi*real_val[1])
        # fitness_val = (1+np.cos(2*np.pi*real_val[0]*real_val[1])) * np.exp(-(abs(real_val[0])+abs(real_val[1]))/2)
        return fitness_val

    def roulette_wheel_selection(self, population):
        chooses_ind = []
        population_fitness = sum([self.fitnessFunc(population[i])
                                 for i in range(0, population.shape[0])])
        chromosome_fitness = [self.fitnessFunc(population[i])
                              for i in range(0, population.shape[0])]
        chromosome_probabilities = [
            chromosome_fitness[i]/population_fitness for i in range(0, len(chromosome_fitness))]
        for i in range(0, population.shape[0]):
            chooses_ind.append(np.random.choice([i for i in range(
                0, len(chromosome_probabilities))], p=chromosome_probabilities))  # Chromosome selection based on their probability of selection
        return chooses_ind  # return selected individuals

    def selectInd(self, chooses_ind, pop):  # Perform crossover on the selected population
        new_pop = []
        for i in range(0, len(chooses_ind), 2):
            a, b = self.crossOver(
                pop[chooses_ind[i]], pop[chooses_ind[i+1]])
            new_pop.append(a)
            new_pop.append(b)
        npa = np.asarray(new_pop, dtype=np.float16)

        fixed_pop = self.normFixRange(npa)
        return fixed_pop

    def bestResult(self, population):  # calculate best fitness, avg fitness
        population_fitness = [self.fitnessFunc(
            population[i]) for i in range(0, population.shape[0])]
        population_best_fitness = max(population_fitness)
        agents_index = np.argmax(population_fitness)
        agents = population[agents_index]
        avg_population_fitness = sum(
            population_fitness) / len(population_fitness)
        return population_best_fitness, avg_population_fitness, population_fitness, agents

    def run(self):
        avg_population_fitness = []
        population_best_fitness = []
        population_fitness = []
        agents = []
        # crossover = 0 => convex ; crossover = 1 => linear
        rga = RGA((100, 2), low=[-3, 4.1], high=[12.1, 5.8], crossover=0)
        # rga = RGA((100, 2),low=[-4, -1.5], high=[2, 1], crossover=0)
        pop = rga.initialization()
        decoded = rga.chromosomeDecode(pop)
        pop = decoded
        for i in range(0, self.max_round):
            b_f, p_f, p, a = rga.bestResult(pop)
            avg_population_fitness.append(p_f)
            population_best_fitness.append(b_f)
            population_fitness.append(p)
            agents.append(a)
            selected = rga.roulette_wheel_selection(pop)
            npop = rga.selectInd(selected, pop)
            new_pop = rga.mutation(npop)
            pop = new_pop
        return population_best_fitness, avg_population_fitness, population_fitness, agents

    def plot(self, population_best_fitness, avg_population_fitness, population_fitness, agents):
        fig, ax = plt.subplots()
        ax.plot(avg_population_fitness, linewidth=2.0, label="avg_fitness")
        ax.plot(population_best_fitness, linewidth=2.0, label="best_fitness")
        plt.legend(loc="lower right")
        print(f"best solution: {max(population_best_fitness)}")
        print(f"best solution: {agents[np.argmax(population_best_fitness)]}")
        plt.show()
