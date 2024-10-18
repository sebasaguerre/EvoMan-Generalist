import pandas as pd

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