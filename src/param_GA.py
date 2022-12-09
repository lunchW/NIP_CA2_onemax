import numpy as np
import _main_function_

def generate_population(size):

    #chromosome will be [pop_size,t_size,m_rate,m_size,cross_rate
    #pop_size between 5 and 20
    #t_size between 2 and 10
    #m_rate between 0 and 1
    #m_size between 0 and 15
    #crossover rate between 0 and 1

    return [ [np.random.randint(5,20), 2*np.random.randint(1,5), np.random.rand(),np.random.randint(0,15),np.random.rand()] for i in range(size) ]

def fitness(individual):
    #fitness of an individual is 1000-mean_num_iterations. 

    print("evaluating individual, ",individual)
    iterations = _main_function_.run_ga(individual[0],individual[1],individual[2],individual[3],individual[4])
    print("iterations generated")
    #Taking absolute value so we don't end up with negatives ranking higher than the rest
    mean = abs(1000 - np.mean(iterations))
    std = np.std(iterations)

    return (mean, std)
    
def assign_ranks(population):
    #Takes a population as a list of bits and returns a list of tuples (bitstring,rank)

    population_fitness = []
    for candidate in population:
        fit = fitness(candidate)
        population_fitness.append( (candidate,fit[0],fit[1]) )

    #sorting by the mean ascending
    population_fitness.sort(key=lambda a:a[1])

    ranked_pop = [] 
    rank = 1

    while True:

        #indexes of the candidates we'll assign this rank to
        pareto_front= []
        compc = population_fitness[0]
        pareto_front.append(0)
        ranked_pop.append( (rank,compc[0],compc[1],compc[2]) )

        #finding candidates also in this pareto front
        for i in range(1,len(population_fitness)):

            candidate = population_fitness[i]
            if candidate[2] < compc[2]:
                compc = candidate
                pareto_front.append( i )
                ranked_pop.append( (rank,candidate[0],candidate[1],candidate[2]) )

        #these have been added to 
        pareto_front.sort(reverse=True)
        for i in pareto_front:
            population_fitness.pop(i)

        if len(population_fitness) != 0:
            #Now calculate the next pareto front
            rank += 1
            compc = population_fitness[0]
        else:
            break
    
    return ranked_pop

def tournament():
    pass

def crossover():
    pass

def mutate():
    pass

def replacement():
    pass

def get_population_max_fitness(population_list):
    return max(fitness(individual) for individual in population_list)

def main():

    population = generate_population(5)
    print(assign_ranks(population))
    pass




if __name__ == "__main__":

    main()