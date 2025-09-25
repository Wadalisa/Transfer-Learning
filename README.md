# Transfer-Learning

## Introduction

Training a Machine Learning model, is an intense process and is quite time-consuming. Training these models demand more computing power, large data before the production of these models. With the use of the Transfer Learning technique, a model can be pre-trained on a certain task and later adapted and refined for a new, related task.

## Dataset Statistics

### Dataset for Source GP 
Title of Data Set: 227_cpu_small

Variables : 13 

Observations : 8192 

Attribute Characteristics: Continuous 

Missing/NaN Values: The dataset does contain zero values. 

Duplicate rows : None

### Dataset for Target GP

Title of Data Set: 197_cpu_act

Variables : 20 

Observations : 8192 

Attribute Characteristics: Continuous 

Missing/NaN Values: The dataset does contain zero values.

Duplicate rows : None

The datasets are pre-proccessed by dropping duplicates and normalized. Duplicates are 'dropped' to reduce model bias. The data normalization method used is the Min-Max normalization, due to fields such as freemem where the data is large and runqsz where the data are small. Normalize assist in such cases so that the model doesn't regard a certain feature to be more important than the other just because it has a larger value.

## Representation 

The regressor is represented as an expression tree. Each individual of the tree is made up of the root node that is randomly selected from the functional set and the leaf nodes that are randomly selected from either the terminal set or functional set, to create expressions. An expression tree is known for its flexibility.Expression trees make it easier for the information at nodes to be accessed and manipulated. In the expression tree, a node could either be an operator or an operand.

The initial population was generated using the growth method. The growth method generates a tree with an initial depth of n. In this report, the initial depth of the tree is 0, to start generating simple trees.

## Fitness Function

A fitness function is a function used to calculate and assign a non-negative value, to evaluate how best suited an individual of the population. The use of the mean absolute error(MAE) is due to it never returning a negative value, meeting the requirement for a fitness function which a non-neagtive number.

In order to calculate a fitness function ,the MAE is used to evaluate each equation -individual of the population.

## Selection Method

A selection method randomly selects two or more individuals in the population as parents for the next generation. In this report, tournament selection is the selection method of choice. For k individuals, a random number of individuals are selected, for n, number of parents. The individuals with the smallest fitness score is selected for reproduction. Tournament selection is preferable as it is simple and has a complexity of O(n), which makes it easier to compute. One parent is returned in the selection method.

## Genetic Operators

Genetic operators assist with creating 'children' in the population from the parents. 
In this report, Crossover and Mutation are used. The combination of these genetic operators allows for both the exploitation and exploration of the search space. Crossover will be use for \textit{exploitation} in the search space meaning that cross over will combine two 'good' attributes from the parent trees as per fitness. whereas Mutation will be used for \textit{exploration} to introduce variety and new parts of the search space.

### Crossover Operator

The crossover operator will swap out sub-trees of the parents. It locates a random point in the sub-trees of the parent trees then the selected sub-trees will be "crossed over" to the other parent tree. \\ This method is called subtree crossover. The rate is used to determine how much crossover is applied to the population.

### Mutation Operator

The mutation operator locates a random point in the parent tree then replacing the node or leaf at that tree with a random node from the search space. This is called Point Mutation. The rate is used to control how much of the population will get mutated.

## Transfer Learning

As per the introduction, the source GP is used to pre-train the model with the help of the source dataset. During the training of the model a population of size n is produced at the end of each generation then the last population of the source GP are then transferred to the target GP. The transfer rate used determines how much of the source population will make the initial population of the target GP and the rest of the population is generated using the growth method. The population is then evolved using the genetic operators over $g$ generations. The fitness is the recieved afterwards.

## Termination Criteria

The termination criteria of the GP's is the number of generations.

## Parameters

The parameters used in this paper are, after parameter tuning :

### Population Size

#### Source GP: 50

A moderate population size is chosen, to create diversity, yet not too diverse to ignore important features. due to the large dataset, it will be easier to train the model

#### Target GP: 25

A small population size is chosen as to increase chances of selecting the trained population, yet include some novelty.

#### Crossover Rate: 60%

Crossover is applied against the individuals, if the random float is less than the crossover rate, crossover should occur, if crossover doesn't occur the individual should be added as is to the new population.

#### Mutation Rate: 35%

Mutation is applied against the individuals, if the random float is less than the mutation rate, mutation should occur, if mutation doesn't occur the individual should be added as it is. The slightly high mutation rate is due to the fact that only a point in the tree gets mutated. To increase novelty and reduce converging in the local optimum.

#### Elistism Rate: 5%

To control the growth of the trees generated, a max tree depth is used. After multiple simulations, the tree depth of 3, grew trees that were manageable and weren't too large to create noise.

#### Maximum tree depth: 3

To control the growth and complexity of the trees generated, a max tree depth is used. After multiple simulations, the tree depth of 3, grew trees that were manageable and weren't too large to create noise.

#### Initial Population Generation Method: Growth Method

#### Tournament Size: 2

When more trees are selected in the tournament selection, high probability of selecting "good parents", however the size is not so high to that the chance of picking the same tree is reduced.

#### Functional set: {+,-,/,*,sqrt,cos,sin,log

Basic mathematical operands are used including uriary operators such as cos, sin, log functions these operands are stored as strings in the array. This helps in the creation of an expression tree. Error handling is used to protect division by zero and taking the square-root of a negative number. In the case where the division by zero occurs, 1 is returned, in order to avoid program crashing and for the continuation of the GP. In the case of the square of a negative number, the absolute value of that number is taken therefore, allowing the square-root operand to be evaluated. with a non-negative input.


#### Terminal set: f(xi) for i in range(1, num_features+1) + ["c"] 

The terminal set is for all features in the dataset. The terminal set is depends on the features in each of the datasets used in the GP at the time. The values of x differ in each GP. For the source GP the terminal set will be x1,x2...x12 and constant, c. For the target GP the terminal set will be x1,x2...x19 and constant, c.
