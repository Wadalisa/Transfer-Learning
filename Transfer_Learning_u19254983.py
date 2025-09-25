import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import builtins
import copy #for the deepcopy function in crossover and mutation
import statistics

#Loading dataset
dataset = pd.read_csv("227_cpu_small.tsv", sep='\t')

#normalize data using the Min-Max Normalization method, defining this for later use
def normalize(dataset):
    """Normalize DataFrame columns to [0,1] range"""
    # Exclude non-numeric columns if needed
    numeric_cols = dataset.select_dtypes(include=['number']).columns
    normalized_dataset = dataset.copy()
    
    for col in numeric_cols:
        min_val = dataset[col].min()
        max_val = dataset[col].max()
        normalized_dataset[col] = (dataset[col] - min_val) / (max_val - min_val)
    
    return normalized_dataset

#normalizing dataset
dataset = normalize(dataset)
#dropping duplicates
dataset = dataset.drop_duplicates()
#dropping NaN or inf values
dataset = dataset.dropna()
print(dataset.shape)
print(dataset.head(1))

#getting the number of features
features = [col for col in dataset.columns if col != 'target']
#splitting into X and Y variables
X_train = dataset.drop('target',axis=1)
y_target_train = dataset['target']
#retrieve the indices
index_train = X_train.index

class Node():
    def __init__(self,value = None,left = None,right = None,arity = None):
        self.value = value #value of the node
        self.left = left #left child
        self.right = right #right child
        self.arity = arity #number of values a node can/must carry
    def __str__(self):
        """Returns a string representation of the node and its children."""
        if self.arity == 0:  # Terminal node
            return str(self.value)
        elif self.arity == 1:  # Unary function (e.g., 'sin')
            return f"({self.value} {self.left})"
        else:  # Binary function (e.g., '+', '-')
            return f"({self.left} {self.value} {self.right})"
#growing the tree using the growth method
def grow_tree(depth = 0):
    if depth == max_depth :
        node = random.choice(terminal)
        node = Node(value = node, arity=0)
        return node
    else:
        node = random.choice(function)
        if node in ["+","-","/","*"]:
            arity1 = 2
        else: arity1 = 1
        node = Node(value = node, arity = arity1)
        if arity1 >= 1:
            node.left = grow_tree(depth + 1)
        if arity1 == 2:
            node.right = grow_tree(depth + 1)
        return node


#initializing the population
def init_pop(pop_size):
    pop = [grow_tree() for _ in range(pop_size)]
    return pop
    
def evaluate(node, row, constant):
    if node.arity == 0:  # Terminal node
        if node.value == 'c':
            return constant
        elif node.value.startswith('x'):
            try:
                feat_idx = int(node.value[1:]) - 1
                return row[feat_idx]
            except:
                return 0.0
        else:
            try:
                return float(node.value)
            except:
                return 0.0
    
    left_val = evaluate(node.left, row, constant)
    
    if node.arity == 1:  # Unary operator
        if node.value == 'sqrt':
            return math.sqrt(abs(left_val))
        elif node.value == 'sin':
            return math.sin(left_val)
        elif node.value == 'cos':
            return math.cos(left_val)
        elif node.value == 'log':
                return math.log(abs(left_val)) if left_val > 1e-10 else 0.0
        else:
            return left_val
    else:  # Binary operator
        right_val = evaluate(node.right, row, constant)
        if node.value == '+':
            return left_val + right_val
        elif node.value == '-':
            return left_val - right_val
        elif node.value == '*':
            return left_val * right_val
        elif node.value == '/':
            try:
                return left_val / right_val
            except:
                return 1.0
        else:
            return left_val

#fitness function
def fitness_fun(y_pred, y_target, index, tolerance, equation):
    n = len(y_target) #the number of values in the dataset
    summation = 0 #storing the  the summation
    i = 0
    for ind in index: #calculation of the summation
        diff = y_pred[i] - y_target[ind]
        absolute = abs(diff)
        summation = summation + absolute
        i+=1
    MAE = summation / n
    #calculating the hits criterion
    hits = 0
    for ypred, yhat in zip(y_pred,y_target):
        if abs(ypred - yhat) <= tolerance * abs(ypred):
            hits += 1
    hits_rate = hits / len(y_target)
    MAE = MAE 
    return MAE

def tournament(population, fitness_scores, tournament_size, num_ind,):
    #selecting values of the tournament score
    combined = [[x, y] for x, y in zip(population, fitness_scores)]
    for i in range(num_ind):
        selection = random.sample(combined , tournament_size)
        #select the best individual which is the smallest MSE
        if selection[i][1] > selection[i+1][1]:
            indx = combined.index(selection[i+1])
            selected_parent = population[indx]
        else:
            indx = combined.index(selection[i])
            selected_parent = population[indx]    
    return selected_parent

def crossover(parent1, parent2, crossover_prob):
    if random.random() > crossover_prob:
        return parent1, parent2
    
    # Implement subtree crossover
    def get_random_subtree(node):
        if node.arity == 0 or random.random() < 0.3:
            return node
        elif node.arity == 1 or random.random() < 0.5:
            return get_random_subtree(node.left)
        else:
            return get_random_subtree(node.right)
            
    def replace_subtree(main_tree, subtree, to_replace):
        if main_tree == to_replace:
            return subtree
        if main_tree.arity >= 1:
            main_tree.left = replace_subtree(main_tree.left, subtree, to_replace)
        if main_tree.arity == 2:
            main_tree.right = replace_subtree(main_tree.right, subtree, to_replace)
        return main_tree
    
    subtree1 = get_random_subtree(parent1)
    subtree2 = get_random_subtree(parent2)
    
    child1 = replace_subtree(copy.deepcopy(parent1), copy.deepcopy(subtree2), subtree1)
    child2 = replace_subtree(copy.deepcopy(parent2), copy.deepcopy(subtree1), subtree2)
    
    return child1, child2

def mutation(parent, probability,max_depth):
    if random.random() > probability:
        return parent
    
    # Make a copy of the parent to modify
    mutant = copy.deepcopy(parent)
    
    # Select a random node to mutate
    def get_random_node(node, depth=0):
        if node.arity == 0 or random.random() < 1.0/(max_depth-depth+1):
            return node
        elif node.arity == 1 or random.random() < 0.5:
            return get_random_node(node.left, depth+1)
        else:
            return get_random_node(node.right, depth+1)
    
    node_to_mutate = get_random_node(mutant)
    
    # Mutate the selected node
    if node_to_mutate.arity == 0:  # Terminal node
        node_to_mutate.value = random.choice(terminal)
    else:  # Function node
        node_to_mutate.value = random.choice(function)
        # If arity changed, we may need to grow new subtrees
        if node_to_mutate.value in ["+", "-", "*", "/","pow"]:
            node_to_mutate.arity = 2
            if node_to_mutate.right is None:
                node_to_mutate.right = grow_tree(depth=max_depth-1)
        else:  # Unary function
            node_to_mutate.arity = 1
            if node_to_mutate.left is None:
                node_to_mutate.left = grow_tree(depth=max_depth-1)
    
    return mutant

## Initializing Parameters
population_size = 100
max_depth = 3
constant = 0.5
function = ["*","+","-","/","sqrt","sin","cos","log"]
num_features = len(dataset.columns)-1 #number of columns
terminal = [f"x{i}" for i in range(1, num_features+1)] + ["c"]  # for n number of x features x1, x2,..., xn, c
generations = 10
tolerance = 0.05
regulation = 0.05
elitism_rate = 0.05
mutation_percentage = 0.35
crossover_percentage = 0.6

## Generating Source GP
seed = 0
random.seed(seed)
#having a first population
first_population = init_pop(population_size)
#to store the new population at every generation
curr_pop = first_population 
#collect all the fitness scores for each generation mainly for the histogram
gen_fitness_score=[]
#clean out the array for the new population
new_pop = []
transfer_best = []
for gen in range(generations):
    # to store the fitness score of each individual in the population for the selection process
    fit_score = []
    #to store the predicted Y value so i can calculate the fitness function of the generation
    y_pred = []
    #to combine the population with its fitness scores
    combined = []
    #Evaluate the fitness scores of each
    for i in range(len(curr_pop)):
        equation = curr_pop[i]
        #get predicted Y value for the training datasets
        temp_ypred = [] #temporary list for this equation
        for indx in index_train:
            y_predict = evaluate(equation,X_train,constant= constant) 
            temp_ypred.append(y_predict)
        fitness_score= fitness_fun(temp_ypred, y_target_train, index_train, tolerance,equation)
        fit_score.append(fitness_score)
         #for reproduction/ elitism
    sorted = builtins.sorted #had to overwrite wasn't working on its own
    combined = [[x, y] for x, y in zip(curr_pop, fit_score)]
    #sorting
    num_individuals = round(len(curr_pop) * elitism_rate)
    sorted_pop= sorted(combined, key=lambda x: x[1])
    best_individuals = []
    for i in range(num_individuals):
        best_individuals.append(sorted_pop[i][0])
    new_pop = copy.deepcopy(best_individuals)
    transfer_best = copy.deepcopy(best_individuals)
    #troubleshooting Nan or infinite trees
    for f in range(len(fit_score)):
        if math.isnan(fit_score[f]) or math.isinf(fit_score[f]):
            #assigning them the worst fitness
            fit_score[f] = 4.0 
    while len(new_pop) < len(curr_pop) and len(new_pop) <= population_size:
        #Selection using tournament selection
        parent1 = tournament(curr_pop, fit_score, tournament_size = 2 , num_ind = 1)
        parent2 = tournament(curr_pop, fit_score, tournament_size = 2, num_ind = 1)
        #getting fitness for parents
        y_p1 = []
        y_p2 = []
        for indx in index_train:
            y_pred_p1 = evaluate(parent1,X_train,constant= constant)
            y_pred_p2 = evaluate(parent2,X_train,constant= constant)
            y_p1.append(y_pred_p1)
            y_p2.append(y_pred_p2)
        parent1_fit= fitness_fun(y_p1, y_target_train, index_train, tolerance,equation)
        parent2_fit = fitness_fun(y_p2, y_target_train, index_train, tolerance,equation)
        #Create a crossover with the rate of 60%
        children = crossover(parent1,parent2,crossover_percentage)
        child1 = children[0]
        child2 = children[1]
        #making the the crossover strict
        y_child1 = []
        y_child2 = []
        for indx in index_train:
            y_pred_c1 = evaluate(child1,X_train,constant= constant)
            y_pred_c2 = evaluate(child2,X_train,constant= constant)
            y_child1.append(y_pred_c1)
            y_child2.append(y_pred_c2)
        child1_fit = fitness_fun(y_child1, y_target_train, index_train, tolerance,equation)
        child2_fit = fitness_fun(y_child2, y_target_train, index_train, tolerance,equation)
        if child1_fit > parent1_fit and child1_fit > 0:#and child1_fit > parent2_fit:
            children = crossover(parent1,parent2,crossover_percentage)
            child1 = children[0]
            child2 = child2
        elif child2_fit > parent2_fit and child2_fit > 0: #and child2_fit > parent1_fit:
            children = crossover(parent1,parent2,crossover_percentage)
            child1 = child1
            child2 = children[1]
        else:
            child1 = child1
            child2 = child2
        #Create mutation with a mutation rate of 40%
        child1 = mutation(child1,mutation_percentage,max_depth)
        child2 = mutation(child2, mutation_percentage,max_depth)
        mut_c1_y = []
        mut_c2_y = []
        #creating a strict mutation
        for indx in index_train:
            y_pred_mc1 = evaluate(child1,X_train,constant= constant)
            y_pred_mc2 = evaluate(child2,X_train,constant= constant)
            mut_c1_y.append(y_pred_mc1)
            mut_c2_y.append(y_pred_mc2)
        child1_mfit = fitness_fun(mut_c1_y, y_target_train, index_train, tolerance,equation)
        child2_mfit = fitness_fun(mut_c2_y, y_target_train, index_train, tolerance,equation)
        if child1_mfit > parent1_fit and child1_fit > 0:#and child1_fit > parent2_fit:
            children = crossover(parent1,parent2,crossover_percentage)
            child1 = children[0]
            child2 = child2
        elif child2_mfit > parent2_fit and child2_fit > 0: #and child2_fit > parent1_fit:
            children = crossover(parent1,parent2,crossover_percentage)
            child1 = child1
            child2 = children[1]
        else:
            child1 = child1
            child2 = child2
        new_pop.append(child1)
        new_pop.append(child2)
    curr_pop = new_pop
print("-----------------------RUN----------COMPLETE-------------------------")

## Generating Source GP Regressor
print("----------------------------------------------------------------------------")
print("-------------Best Predicted Equation----------------------------------------")
# trying to find the best from the last generation to check the best equation
final_pop_fits = []
for i in range(len(curr_pop)):
    equation = curr_pop[i]
    #get predicted Y value for the training datasets
    ypred = [] #temporary list for this equation
    for indx in index_train:
        y_predict = evaluate(equation,X_train,constant= constant) 
        ypred.append(y_predict)
    fitness_score = fitness_fun(ypred, y_target_train, index_train, tolerance,equation)
    final_pop_fits.append(fitness_score)
#getting equation from the last population
#checks for the smallest fitness score
best_fitted = min(final_pop_fits) 
#gets the index of that fitness score
best_index = fit_score.index(best_fitted)
#find the equation
best_equation = curr_pop[best_index]
print("MAE: ", best_fitted)
print(best_equation)

## Transfer Learning
#loading second dataset
dataset2 = pd.read_csv("197_cpu_act.tsv", sep = "\t")
#normalizing dataset
dataset2 = normalize(dataset2)
#dropping duplicates
dataset2 = dataset2.drop_duplicates()
#dropping NaN or inf values
dataset2 = dataset2.dropna()
print(dataset2.shape)
print(dataset2.head(1))

#splitting data
#getting the number of features
features = [col for col in dataset.columns if col != 'target']
#splitting into X and Y variables
X_test = dataset2.drop('target',axis=1)
y_target_test = dataset2['target']
#retrieve the indices
index_test = X_test.index

##Transfer Learning
#getting common features from source dataset and target dataset
common_features = list(set(dataset.columns) & set(dataset2.columns))
print("Common features:", common_features)
target_popsize = 50
#calculating how much half of the last population
transfer_rate= 0.2 #using 20% of the last population
function = ["*","+","-","/","sqrt","sin","cos"]
num_features = len(dataset2.columns)-1 #number of columns
terminal = [f"x{i}" for i in range(1, num_features+1)] + ["c"]  # for n number of x features x1, x2,..., xn, c
t_generations = 5

#sort list of the source populations last generation
#Combine trees with their fitness scores
combined = list(zip(curr_pop, final_pop_fits))
#Sort by fitness (ascending for MAE, descending if using accuracy)
source_pop_sorted = sorted(combined, key=lambda x: x[1])  # x[1] is the fitness
#Extract just the sorted trees (without fitness values) and select top individuals for transfer (e.g., top 50%)
# Calculate transfer count safely
transfer_count = int(len(curr_pop) * transfer_rate)
if transfer_count <= 0:
    transfer_count = 1  # Ensure we transfer at least 1
#Select top performers
source_pop_trees = [tree for tree, fit in source_pop_sorted[:transfer_count]]
# Generate remaining population
remaining_count = max(target_popsize - transfer_count, 0)
new_individuals = init_pop(remaining_count) if remaining_count > 0 else []
# Combine populations
target_pop = source_pop_trees + new_individuals

##Generating Target GP
#to store the new population at every generation
curr_pop2 = target_pop
#collect all the fitness scores for each generation mainly for the histogram
gen_fitness_score=[]
#clean out the array for the new population
new_pop = []
transfer_best = []
for gen in range(t_generations):
    # to store the fitness score of each individual in the population for the selection process
    fit_score = []
    #to store the predicted Y value so i can calculate the fitness function of the generation
    y_pred = []
    #to combine the population with its fitness scores
    combined = []
    #Evaluate the fitness scores of each
    for i in range(len(curr_pop2)):
        equation = curr_pop2[i]
        #get predicted Y value for the training datasets
        temp_ypred = [] #temporary list for this equation
        for indx in index_test:
            y_predict = evaluate(equation,X_test,constant= constant)
            temp_ypred.append(y_predict)
        fitness_score = fitness_fun(temp_ypred, y_target_test, index_test, tolerance,equation)
        fit_score.append(fitness_score)
         #for reproduction/ elitism
    sorted = builtins.sorted #had to overwrite wasn't working on its own
    combined = [[x, y] for x, y in zip(curr_pop, fit_score)]
    #sorting
    num_individuals = round(len(curr_pop2) * elitism_rate)
    sorted_pop= sorted(combined, key=lambda x: x[1])
    best_individuals = []
    for i in range(num_individuals):
        best_individuals.append(sorted_pop[i][0])
    new_pop = copy.deepcopy(best_individuals)
    transfer_best = copy.deepcopy(best_individuals)
    #troubleshooting Nan or infinite trees
    for f in range(len(fit_score)):
        if math.isnan(fit_score[f]) or math.isinf(fit_score[f]):
            #assigning them the worst fitness
            fit_score[f] = 4.0
    while len(new_pop) < len(curr_pop2) and len(new_pop) <= population_size:
        #Selection using tournament selection
        parent1 = tournament(curr_pop2, fit_score, tournament_size = 2 , num_ind = 1)
        parent2 = tournament(curr_pop2, fit_score, tournament_size = 2, num_ind = 1)
        #getting fitness for parents
        y_p1 = []
        y_p2 = []
        for indx in index_test:
            y_pred_p1 = evaluate(parent1,X_test,constant= constant)
            y_pred_p2 = evaluate(parent2,X_test,constant= constant)
            y_p1.append(y_pred_p1)
            y_p2.append(y_pred_p2)
        parent1_fit = fitness_fun(y_p1, y_target_test, index_test, tolerance,equation)
        parent2_fit = fitness_fun(y_p2, y_target_test, index_test, tolerance,equation)
        #Create a crossover with the rate of 60%
        children = crossover(parent1,parent2,crossover_percentage)
        child1 = children[0]
        child2 = children[1]
        #making the the crossover strict
        y_child1 = []
        y_child2 = []
        for indx in index_test:
            y_pred_c1 = evaluate(child1,X_test,constant= constant)
            y_pred_c2 = evaluate(child2,X_test,constant= constant)
            y_child1.append(y_pred_c1)
            y_child2.append(y_pred_c2)
        child1_fit = fitness_fun(y_child1, y_target_test, index_test, tolerance,equation)
        child2_fit = fitness_fun(y_child2, y_target_test, index_test, tolerance,equation)
        if child1_fit > parent1_fit and child1_fit > 0:#and child1_fit > parent2_fit:
            children = crossover(parent1,parent2,crossover_percentage)
            child1 = children[0]
            child2 = child2
        elif child2_fit > parent2_fit and child2_fit > 0: #and child2_fit > parent1_fit:
            children = crossover(parent1,parent2,crossover_percentage)
            child1 = child1
            child2 = children[1]
        else:
            child1 = child1
            child2 = child2
        #Create mutation with a mutation rate of 40%
        child1 = mutation(child1,mutation_percentage,max_depth)
        child2 = mutation(child2, mutation_percentage,max_depth)
        mut_c1_y = []
        mut_c2_y = []
        #creating a strict mutation
        for indx in index_test:
            y_pred_mc1 = evaluate(child1,X_test,constant= constant)
            y_pred_mc2 = evaluate(child2,X_test,constant= constant)
            mut_c1_y.append(y_pred_mc1)
            mut_c2_y.append(y_pred_mc2)
        child1_mfit = fitness_fun(mut_c1_y, y_target_test, index_test, tolerance,equation)
        child2_mfit = fitness_fun(mut_c2_y, y_target_test, index_test, tolerance,equation)
        if child1_mfit > parent1_fit and child1_fit > 0:#and child1_fit > parent2_fit:
            children = crossover(parent1,parent2,crossover_percentage)
            child1 = children[0]
            child2 = child2
        elif child2_mfit > parent2_fit and child2_fit > 0: #and child2_fit > parent1_fit:
            children = crossover(parent1,parent2,crossover_percentage)
            child1 = child1
            child2 = children[1]
        else:
            child1 = child1
            child2 = child2
        new_pop.append(child1)
        new_pop.append(child2)
    curr_pop2 = new_pop
print("-----------------------RUN----------COMPLETE-------------------------")

print("----------------------------------------------------------------------------")
print("-------------Best Predicted Equation----------------------------------------")
# trying to find the best from the last generation to check the best equation
final_pop_fits = []
for i in range(len(curr_pop2)):
    equation = curr_pop2[i]
    #get predicted Y value for the training datasets
    ypred = [] #temporary list for this equation
    for indx in index_test:
        y_predict = evaluate(equation,X_test,constant= constant)
        ypred.append(y_predict)
    fitness_score = fitness_fun(ypred, y_target_test, index_test, tolerance,equation)
    final_pop_fits.append(fitness_score)
#getting equation from the last population
#checks for the smallest fitness score
best_fitted = min(final_pop_fits)
#gets the index of that fitness score
best_index = fit_score.index(best_fitted)
#find the equation
best_equation = curr_pop2[best_index]
print("MAE: ", best_fitted)
print(best_equation)