import numpy as np
import _main_function_

from collections import Counter

def generate_population(size):

    """
    A chromosome will be [pop_size,t_size,m_rate,m_size,cross_rate
    pop_size between 5 and 20
    t_size between 2 and 10
    m_rate between 0 and 1
    m_size between 0 and 15
    crossover rate between 0 and 1
    """

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

def tournament(ranked_population,t_size=2):
    #selecting parents based on tournament selection with no replacement

    if t_size >= len(ranked_population):
        print("tournament size greater than or equal to population size")
        return 0

    #creating the probabilities of being selected
    ranks = list(np.array(ranked_population,dtype=object)[:,0])
    c = Counter(ranks)
    ranks = list(set(ranks))
    rank_sum = sum(ranks)
    rank_prob = { ranks[i] : ranks[i-1]/rank_sum  for i in range(len(ranks)) }

    #There are a lot of subltities here because candidates can share ranks and therefore probabilities
    probabilities = [ rank_prob[ranked_population[i][0]]/c[ranked_population[i][0]] for i in range(len(ranked_population))]

    parents = []

    for r in range(2):
        tourn = []
        pool = ranked_population.copy()

        #picks random index based on rank probabilities and pops it, adding it to the tournament
        to_pop = np.random.choice(np.arange(len(pool)),size=t_size, replace=False, p=probabilities)
        for index in to_pop:
            tourn.append( pool.pop(index) )
        
        #Pick a candidate to compare the others to from what is left in the pool
        compc = pool[np.random.randint(len(pool))]

        final = []

        for candidate in tourn:
            if candidate[0] > compc[0]:
                final.append(candidate)

        if len(final) > 1:
            #split the tie randomly between winners
            parents.append(final[np.random.randint(len(final))])
        elif len(final) == 1:
            #We have only one winner
            parents.append(final[0])
        else:
            #We have no winners, so take randomly from those in the tournament
            parents.append(tourn[np.random.randint(len(tourn))])                

    return parents

def crossover(parent_1,parent_2,c_rate):

    """ 
    Takes in two parents (rank, chromosome, mean, std).
    If random number (betwen 0 and 1) is less than c_rate, cross over the parent chromosomes at a random location
    return two children chromosomes (i.e. just lists, without mean, std or rank)
    """
    pass

def mutate(child_1,child_2,m_rate,m_size):

    """
    Takes in two children chromosomes.
    do this m_rate number of times:
        if random number (0<r<1) less than m_rate:
            Pick gene at random.
            mutate the gene with 
                (if it is population, it needs to stay an int) 

    """

    pass

def replacement():

    

    pass

def get_population_max_fitness(population_list):
    return max(fitness(individual) for individual in population_list)

def main():

    iterations = 100
    #population size
    p_size = 15
    #tournament size
    t_size=2
    #crossover rate
    c_rate = 0.8
    #mutation rate
    m_rate = 0.1
    #mutation size < 5
    m_size = 2

    for i in range(iterations):

        population = generate_population(p_size)
        ranked_population = assign_ranks(population)
        parents = tournament(ranked_population,t_size=t_size)

        child_1 , child_2 = crossover(parents[0],parents[0],c_rate)
        child_1 , child_2 = mutate(child_1,child_2,m_rate,m_size)

        population = replacement(population, child_1,child_2)




if __name__ == "__main__":

    main()