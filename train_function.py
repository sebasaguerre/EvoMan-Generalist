import pandas as pd

# imports other libs
import numpy as np
import os
import random
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm

# os.chdir('/Users/christophlaute/Evolutionary/Group-Assignment-Evolutionary-Computing/EvoMan-Generalist')

def adaptive_ensemble_mutation(population, p_mutation, nr_gen, p_gaussian):
    """mutation operator that applies gaussian mutation and cauchy distribution adaptively"""
    #initialize variables
    gaussian_rows = int(p_gaussian*population.shape[0])
    gaussian_indx = np.random.choice(population.shape[0],gaussian_rows,replace=False)
    gaussian_pop = population[gaussian_indx]
    cauchy_pop = np.array([individual for individual in population.tolist() if individual not in gaussian_pop.tolist()])
    success_counter=0

    

    #mutation per operator_subgroup
    for index, g_ind in  enumerate(gaussian_pop):
        for gene in range(len(g_ind)):
            if p_mutation > random.uniform(0,1):
                gaussian_pop[index][gene]+=np.random.normal(0,1) 
    for index, c_ind in enumerate(cauchy_pop):
        for gene in range(len(c_ind)):
            if p_mutation > random.uniform(0,1):                     #still have to debug
                cauchy_pop[index][gene]+=1+(np.arctan(np.random.standard_cauchy())/np.pi)

    
    #form new population and evaluate top percent
    
    if cauchy_pop.size > 0:
        population = np.concatenate((gaussian_pop, cauchy_pop))
    else:
        population = gaussian_pop
    if nr_gen > 10:
        
        fitness_list = get_fitness(population)
        fitness_pop=fitness_list[-1]
        sorted_fitness_ind = sorted(range(len(fitness_pop)), key=lambda x:fitness_pop[x], reverse=True)
        elites = population[sorted_fitness_ind[:int(0.2*population.shape[0])]]
        for element in elites:
            if element in gaussian_pop:
                success_counter+=1
        p_gaussian = success_counter/len(elites)

    
    # p_gaussian=0.3
    return population, p_gaussian

def get_max_range(population):
    max_range =- 1
    for index,individual in enumerate(population):
        for other in population[index+1:]:
            range = abs(np.linalg.norm(individual-other))
            if range > max_range:
                max_range = range
    return max_range


#fitness sharing
def shared_fitness_function(population, fitness_values, current_gen, max_gen):
    """function that penalizes fitness if individual is too similar"""
    k = 5 / max_gen
    s_fit_values=[]
    max_range = get_max_range(population)
    initial_threshold = max_range * 0.2
    final_threshold = max_range * 0.03
    threshold = final_threshold + (initial_threshold-final_threshold)*np.exp(-k*current_gen)
    
   #loop over individual in population and the possible pairs
    for value in range(population.shape[0]):
        s_fit_pen =  0
        for other in range(population.shape[0]):
            if other != value:
                
                #calculates euclidean distance between individual and candidate for niche
                euc = np.linalg.norm(population[value]-population[other])
                if euc < threshold:
                    
                    #sums penalisation
                    s_fit_pen += (1-(euc / threshold))
        
        #calculates new value
        if s_fit_pen > 0:
            s_fit_value = fitness_values[value] / s_fit_pen
            s_fit_values.append(s_fit_value)
        else:
            s_fit_values.append(fitness_values[value])
    return s_fit_values

def get_diversity(population):
    """calculates how diverse each individual is"""
    mean_ind = np.mean(population,axis=0)
    diversity_list = []
    for individual in population:
        diversity = abs(np.linalg.norm(individual-mean_ind))
        diversity_list.append(diversity)
    return diversity_list

def two_archives_survival(parents, children,fit_parents,fit_mutated, p_fitness, current_gen, max_gen):
    """function that performs survivor selection proportional to p_fitness based on fitness and otherwise based on diversity"""
    
    survivors = []
    ind = 0
    final_fit = []
    
    # Concatenate parents and children into one array, calculate fitness and diversity
    total_pop = np.concatenate((parents, children), axis = 0)                              #AXIS = 0
    diversity_p = get_diversity(total_pop)
    total_pop_fit = np.concatenate((fit_parents,fit_mutated),axis =0)                      #CAMBIAR
    total_shared = np.array(shared_fitness_function(total_pop, total_pop_fit, current_gen, max_gen))  # Shared fitness values
    
    # Sort fitnesses and diversity and return indices
    fitness_sorted = np.argsort(total_shared[:, -1])[::-1]
    diversity_sorted = np.argsort(diversity_p)[::-1]
    
    # Loop that chooses survivors proportionally to p_fitness from fitness or else from diversity
    while len(survivors) < parents.shape[0]:
        if np.random.uniform(0, 1) < p_fitness and total_pop[fitness_sorted[ind]].tolist() not in [s.tolist() for s in survivors]:
            survivors.append(total_pop[fitness_sorted[ind]])
            final_fit.append(total_pop_fit[fitness_sorted[ind]])  
        elif total_pop[diversity_sorted[ind]].tolist() not in [s.tolist() for s in survivors]:
            survivors.append(total_pop[diversity_sorted[ind]])
            final_fit.append(total_pop_fit[diversity_sorted[ind]])  
        else:
            if np.random.uniform(0, 1) < 0.5:
                survivors.append(total_pop[fitness_sorted[ind]])
                final_fit.append(total_pop_fit[fitness_sorted[ind]])  
            else:
                survivors.append(total_pop[diversity_sorted[ind]])   
                final_fit.append(total_pop_fit[diversity_sorted[ind]])  
        ind += 1
    
    return np.array(survivors), np.array(final_fit)


def my_levy(u, c = 1.0, mu = 0.0):
    return mu + c / (2.0 * (norm.ppf(1.0 - u))**2)

def swag_mutation(population, fit_values, m_rate, fitness_threshold=None, **kwargs):
    rng = np.random.default_rng()
    mutated_population = []

    for i in range(len(population)):
       # print(f"fit_values_shape = {fit_values.shape}")
        individual = population[i]
        mutated_individual = individual
        # if fitness is below threshold, mutate all genes
        if fit_values[i][-1] < fitness_threshold:
            for j in range(len(individual)):
                if rng.random() < m_rate:
                    u = rng.random()
                    sign = np.random.choice([-1, 1])
                    mutated_individual[j] = sign*my_levy(u)
        # if fitness is above threshold, mutate only one gene
        elif fit_values[i][-1] >= fitness_threshold or fitness_threshold == None:
            j = random.randint(0, len(individual)-1)
            u = rng.random()
            sign = np.random.choice([-1, 1])
            mutated_individual[j] = sign*my_levy(u)

        # append mutated individual to the mutated population
        mutated_population.append(mutated_individual)

    return np.array(mutated_population)


def evo_strategies(pop_size, m_rate, max_gen, mutation_operator, survival_selection, enemies, local = True, **kwargs):
    """Evolution strategy algorithm made for testing different mutation operators"""
    enemies = enemies 
    if kwargs.get("p_gaussian"): p_gaussian = kwargs.get("p_gaussian")

    # initialize population
    population = pop_init(pop_size)
    fit_values = get_fitness(population)

    # get best individuals
    idx = np.argmax(x[-1] for x in fit_values)
    best_ind = [population[idx]]
    best_fit = [fit_values[idx][-1]]

    # start evos trategies
    for gen in range(max_gen):

        # mutate population based on operator
        if mutation_operator == adaptive_ensemble_mutation:
           mutated, p_gaussian = adaptive_ensemble_mutation(population, m_rate, gen + 1, p_gaussian=p_gaussian)
        elif mutation_operator == swag_mutation:    
            mutated = swag_mutation(population, fit_values, m_rate, **kwargs)
        elif mutation_operator == multi_adaptive_mutation:
            mutated = multi_adaptive_mutation(population, fit_values, m_rate, gen, max_gen)
        elif mutation_operator == cauchy_polynomial_mutation:
            mutated = cauchy_polynomial_mutation(population, fit_values, m_rate, max_gen, gen, 
                                                 generation_threshold=generation_threshold, scale=scale, polynomial_eta=polynomial_eta)

        # evaluate mutation
        fit_mutated = get_fitness(mutated)

        
        if local:
            # memetic algorithm happening every even round  
            if gen % 6 == 0:
                # memetic algorithm -> life improvement of individuals 
                improved_mutated, fit_imporved = pop_simulated_annealing(mutated, fit_mutated, 100, 1, 0.95)
                # survival selection 
                population, fit_values =survival_selection(population, improved_mutated,fit_values,fit_imporved,0.8,gen, max_gen)
            else:
                population, fit_values =survival_selection(population, mutated,fit_values,fit_mutated, 0.8, gen, max_gen)
        else:
            population, fit_values =survival_selection(population, mutated,fit_values,fit_mutated, 0.8, gen, max_gen)
            
        # record best solution 
        idx = np.argmax(x[-1] for x in fit_values)
        gen_best = population[idx]
        gen_best_fitness = fit_values[idx][-1]
        # check if gen best is highest 
        if gen_best_fitness > best_fit[-1]:
            best_ind.append(gen_best)
            best_fit.append(gen_best_fitness)
        else:
            best_ind.append(best_ind[-1])
            best_fit.append(best_fit[-1])

        # termination condition
        if best_fit[-1] > 95:
            break
    
    return best_ind, best_fit 

#' example
#                                    df = train(evo_strategies, swag_mut, set_1
#                                               popsize =  n,
#                                               m_rate = 0.5,
#                                                max_gen = 100,
#                                               mutation_operator = swag_mutation,
#                                               survival_selection = two_archive_survival,
#                                               enemies = enemies,
#                                               fitness_threshold = 0.9                 
                                                

def train(ea, file_name, enemy_set, **kwargs): 
    """
    ParAarameters:
        EA = main algorithm
        file_name = mutation operator (short)
        enemy_set = set1 or set 2
        kwargs = all other arguments for current EA

    Train a EA for n(runs) times on a number of Enemies and outputs:
    
        best_ind_per_enemy -> list of lists, with 10 best ind per enemy
        fitness_evolv_per enemy -> 10 track records of the evolution of fitness 
            scores per enemy
        mean_fit_evolv_per_enemy -> 10 track records of mean fitness per enemy
        sd_evolv_per_enemy -> 10 track record of sd fitness per enemy 
    """
    # Setting up the Enviorment for training EVOMAN
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'debugging ai'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # collect the data 
    best_ind = []
    fit_evolv = []
    mean_evolv = []
    sd_evolv = []

    print("Trainning has Initiated\n")
    for run in range(10):

        best, fitness, fit_mean, fit_sd = ea(**kwargs)

        # append resuls of ea with enemy_i to blobal results
        best_ind.append(best_ind[-1])
        fit_evolv.append(fitness)
        mean_evolv.append()
        sd_evolv.append(fit_sd) 
    
    # Create data structure for DataFrame
    data = {
        "Individuals": best_ind,
        "Evolved Fitness" : fit_evolv,
        "Mean Fitness" : mean_evolv,
        "Sd fitness" : sd_evolv
        }
    
    # Create DataFrame
    df = pd.DataFrame(data)

    # Save DataFrame
    df.to_csv(file_name + "_" +enemy_set + ".csv", index = False)
    
    return df

df = train(evo_strategies, "swag_mutation", "set_1",
            popsize =  101,
            m_rate = 0.5,
            max_gen = 100,
            mutation_operator = swag_mutation,
            survival_selection = two_archives_survival,
            enemies = [4,6,7],
            fitness_threshold = 0.9)