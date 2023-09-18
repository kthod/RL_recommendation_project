


import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt 
from scipy.optimize import minimize, Bounds, linprog
# Environment parameters

global K  # Number of content items
global u_min   # Threshold for content relevance
global C  # Number of cached items
#C = 2
# User model parameters
global N # Number of recommended items
global q   # Probability of ending the viewing session
global alpha   # Probability of selecting a recommended item
#tradeoff_factor=0.4
# Generate random relevance values
global U 
global Cost    



def normalize_matrix_R(R):
    row_sums = R.sum(axis=1)

    # New matrix is old matrix divided by row sums
    # Use np.newaxis to match the dimensions for broadcasting
    R = N*R / row_sums[:, np.newaxis]
    return R

# R = normalize_matrix_R(U)
# print(R.sum(axis=1))

def all_recommendations_are_relevant(recommendations,s):
    """
    function to check whether everey recommended state in the racommendation batch is 
    relevant to s

    arguments:
    recommendations (tuple of ints): recommendation batch for state s
    s (int): current state

    returns:
    True: if all recommendation are relevant to s
    False: otherwise
    """
    for u in recommendations:
        if U[s][int(u)]<u_min:
            return False
    return True


def func(x,Q):
        return -np.dot(x , Q )/N

def maximize_model(Q,s):
    new_P = np.zeros((K, K), dtype=np.float64) #create a Q value array
    # # Print the result
    # print(f"Minimized function value: {result.fun}")
    # print(f"Argmin: {result.x}")

    
    C=-1/N*Q
    # Equality constraint for summation: sum(x) = N
    A_eq = [[1] * K]
    b_eq = [N]

    # Adding the specific constraint x_i = 0
    # You can directly specify this in bounds, no need to add in A_eq

    # Bounds for each variable: 0 <= x_j <= 1 and x_i = 0
    bounds = [(0, 1) if j != s else (0, 0) for j in range(K)]

    # Now use linprog
    result = linprog(C, bounds=bounds, A_eq=A_eq, b_eq=b_eq, method='highs')
    new_P = result.x
    max = -result.fun
    return new_P,max

def maximize_model2(P,Q,s):
    new_P = np.zeros((K, K), dtype=np.float64) #create a Q value array
    # # Print the result
    # print(f"Minimized function value: {result.fun}")
    # print(f"Argmin: {result.x}")

    
        
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - N},  # Sum of elements should be N
                    {'type': 'eq', 'fun': lambda x: x[s]}]
    # Define the bounds (0 <= v(i) <= 1 for all i)
    bounds = Bounds([0]*K, [1]*K)

    # Initial guess
    x0 = P  # Initial guess should respect the constraints

    # Call the minimize() function with the 'SLSQP' method, bounds and constraints
    result = minimize(func, args = (Q), x0=x0, method='SLSQP', bounds=bounds, constraints=constraints)
    new_P = result.x
    max = -result.fun
    return new_P,max


def slateQ(gamma, epsilon,learning_rate):
    """
    Function to perform Q learning algorithm.

    arguments:
    gamma (float): discounting factor
    epsilon (float): epslilon greedy probability
    learning_rate (float): learning rate

    returns:
    Q (matrix K x num_of_actions):  martix of state action value function calculated using bellman equation
    """
    Q = np.zeros((K,K)) 
    P = normalize_matrix_R(U) 
    #prev_Q = np.zeros((K,K))
    t = 0
    while True:
        s = np.random.randint(K) #random initial state
        while True:
            if np.random.uniform() < epsilon:  # Explore if e(t) times
                
                numbers = list(range(K))
                numbers.remove(s)

                # Sample two distinct integers from the list
                #a = tuple(random.sample(numbers, N))
                i = random.sample(numbers,1)[0] #choose random action
                    
            else:  # Exploit 1-e(t) times
                #a= tuple(np.argpartition(-np.array(P[s]), N)[:N])
                i = np.argmax(Q[s]) #choose greedily the action with highest Q value
            # a = action_table[s][a_idx]  


            if U[s][i]>u_min:
            
                
                s_prime = i  # Pick a random item from relevant recommended items
                reward = (1 - 2*Cost[i])
            else:
                s_prime = np.random.randint(K)  # Pick a random item
                reward = -1
            #reward = (1 - 2*Cost[i])
            if np.random.uniform() < q: #if user opt to terminate session
                target = reward
                Q[s][i] = Q[s][i] + learning_rate * ( target - Q[s][i] )
                break
            else:
                P[s_prime],maxval = maximize_model(Q[s_prime],s_prime)
                target = reward + gamma*maxval
            Q[s][i] = Q[s][i] + learning_rate * ( target - Q[s][i] )
            
            s = s_prime
        t+=1
        epsilon = (t+1)**(-1/3)*(K*math.log(t+1))**(1/3)
        #epsilon = 0.1
        #learning_rate = learning_rate*(1/t)**(1/2)
        
        #if (np.max(np.abs(prev_Q - Q)) < delta and t>1000*K) or 
        if t > 60*K: #check if the new V estimate is close enough to the previous one;
            break # if yes, finish loop
        #prev_Q = Q.copy()
    
    return P



# pi_Q_learning  =  np.zeros((K, N), dtype=np.int16)

# Q = Q_learning(1-q,1,0.01)

# pi_Q_learning = np.argpartition(Q, -N, axis=1)[:, -N:]


def simulate_session(policy, max_steps=1000):
    """
    Simulate a viewing session following a given policy

    arguments:
    policy to be simulated

    returns:
    total cost of the session
    
    """
    s = np.random.randint(K)  # random initial
    cost_total = Cost[s]  
    for _ in range(max_steps):
        if np.random.uniform() < q:  # The user decides to quit
            break

        if (all_recommendations_are_relevant(policy[s],s)):
            
            if np.random.uniform() < alpha:  # If all recommended items are relevant
                s_prime = int(np.random.choice(policy[s]))  # Pick a random item from relevant recommended items
            else:  # If at least one recommended item is not relevant
                s_prime = np.random.randint(K)  # Pick a random item
        else:
            s_prime = np.random.randint(K)  # Pick a random item
        
        s=s_prime
        cost_total += Cost[s]  # Add the cost of the picked item
    return cost_total

avg_cost = []

def Q_learning(gamma, epsilon,learning_rate):
    """
    Function to perform Q learning algorithm.

    arguments:
    gamma (float): discounting factor
    epsilon (float): epslilon greedy probability
    learning_rate (float): learning rate

    returns:
    Q (matrix K x num_of_actions):  martix of state action value function calculated using bellman equation
    """
    Q = np.zeros((K,num_of_actions)) 
    prev_Q = np.zeros((K,num_of_actions))
    t = 0
    while True:
        s = np.random.randint(K) #random initial state
        while True:
            if np.random.uniform() < epsilon:  # Explore if e(t) times
                
                a_idx = np.random.randint(num_of_actions) #choose random action
                    
            else:  # Exploit 1-e(t) times
                
                a_idx = np.argmax(Q[s]) #choose greedily the action with highest Q value
            a = action_table[s][a_idx]  

            if (all_recommendations_are_relevant(a,s)):
            
                if np.random.uniform() < alpha:  # If all recommended items are relevant
                    s_prime = int(np.random.choice(a))  # Pick a random item from relevant recommended items
                else:  # If at least one recommended item is not relevant
                    s_prime = np.random.randint(K)  # Pick a random item
            else:
                s_prime = np.random.randint(K)  # Pick a random item
        
            if np.random.uniform() < q: #if user opt to terminate session
                target = (1/2 - Cost[s_prime])
                Q[s][a_idx] = prev_Q[s][a_idx] + learning_rate * ( target - prev_Q[s][a_idx] )
                break
            else:
                target = (1/2 - Cost[s_prime]) - Cost[s_prime] + gamma*np.max(prev_Q[s_prime])
            Q[s][a_idx] = prev_Q[s][a_idx] + learning_rate * ( target - prev_Q[s][a_idx] )
            
            s = s_prime
        t+=1
        epsilon = (t+1)**(-1/3)*(num_of_actions*math.log(t+1))**(1/3)
        #epsilon = 0.1
        #learning_rate = learning_rate*(1/t)**(1/2)
        
        #if (np.max(np.abs(prev_Q - Q)) < delta and t>1000*K) or 
        if t > 2000*K: #check if the new V estimate is close enough to the previous one;
            break # if yes, finish loop
        prev_Q = Q.copy()
    print(t)
    return Q






def simulation(policy):
    """
    function to run multiple sessions
    """
    total_cost = 0
    num_of_episodes=50000
    for _ in range(num_of_episodes):
        total_cost  += simulate_session(policy)

    print(total_cost/num_of_episodes)
    return total_cost/num_of_episodes
# print(U)
# print(Cost)
# #print(P_opt1)
# print(pi_Q_learning)

# print("average cost for Policy iteration:")
# #simulation(P_opt1)
# #simulation(P_opt2)
# print("average cost for Q Learning:")
# simulation(pi_Q_learning)
slateQ_avgCost =[]
QL_avgCost =[]
slateQ_time = []
QL_time = []
N_vec=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for k in N_vec:
    print(k)
    K = 100  # Number of content items
    u_min = k  # Threshold for content relevance
    C = int(0.2 * K)  # Number of cached items
    #C = 2
    # User model parameters
    N = 5 # Number of recommended items
    q = 0.2  # Probability of ending the viewing session
    alpha = 0.8  # Probability of selecting a recommended item
    #tradeoff_factor=0.4
    # Generate random relevance values
    U = np.random.rand(K, K)
    np.fill_diagonal(U, 0)  # Set diagonal elements to 0
    # U = np.array([[0., 0.09709217, 0.95697935, 0.76421269, 0.79379138],
    #               [0.85679266, 0., 0.73115609, 0.97025111, 0.00706508],
    #               [0.38327773, 0.27582305, 0., 0.40938946, 0.70918518],
    #               [0.27415892, 0.89691232, 0.47103534, 0., 0.97776446],
    #               [0.06699551, 0.96500574, 0.00547615, 0.74654658, 0.]])
    # U = np.array([[0.0, 0.8, 0.3, 0.6],
    #               [0.8, 0.0, 0.7, 0.2],
    #               [0.3, 0.1, 0.0, 0.2],
    #               [0.6, 0.4, 0.2, 0.0]])
    
    #vector to denote the cost of each state. 1 for non-cached, 0 for cached
    Cost = [1]*(K-C) +[0]*C  
    # action_set = []
    # for i in range(K):
    #     for j in range(i+1,K):
    #         a = (i, j)
    #         action_set.append(a)

    # num_of_actions = len(action_set)

    # action_table = [[] for _ in range(K)]

    # for i in range(K):
    #     for a in action_set:
    #         if i not in a:
    #             action_table[i].append(a)
    # num_of_actions = len(action_table[0]) 
    #Cost = [1,0,1,0]
    random.shuffle(Cost)
    start_time = time.time()
    P = slateQ(1-q,1,0.01)
    end_time = time.time()
    pi_Q_learning = np.argpartition(P, -N, axis=1)[:, -N:]
    slateQ_avgCost.append(simulation(pi_Q_learning))

    slateQ_time.append(end_time - start_time)

    # start_time = time.time()
    # Q = Q_learning(1-q,1,0.01)
    # end_time = time.time()

    # for s in range(K):
    # #Q[s][s] = float('-inf')
    #     action = np.argmax(Q[s])
    #     pi_Q_learning[s] = action_table[s][action]
    
    # QL_avgCost.append(simulation(pi_Q_learning))
    # QL_time.append(end_time - start_time)
# Create a figure and axis
print(avg_cost)
plt.plot(N_vec, slateQ_avgCost, label='slateQ')
#plt.plot(K_vec, QL_avgCost, label='tabular QL')
# Title and labels
plt.title("Average cost per session for slateQ")
plt.xlabel("u_min")
plt.ylabel("average cost per session")
#plt.legend()
# Show the plot
plt.show()

plt.plot(N_vec, slateQ_time, label='slateQ')
#plt.plot(K_vec, QL_time, label='tabular QL')
# Title and labels
plt.title("Elapsed time for slateQ and Qlearning")
plt.xlabel("u_min")
plt.ylabel("elapsed time in sec")
#plt.legend()
plt.show()