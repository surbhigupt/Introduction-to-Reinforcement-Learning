import warnings 
warnings.filterwarnings("ignore")

import numpy as np
from itertools import cycle
import random 

random.seed(321)
np.random.seed(321)

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    '''
    Evaluate a policy using the state value function
    pi : ndarray = Policy to be evaluated
    P : dict = Transition probabilites from one state to next (for all valid actions)
    '''
    prev_V = np.zeros(len(P), dtype=np.float64)

    while True:
        V = np.zeros(len(P), dtype=np.float64)

        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))

        if np.max(np.abs(prev_V - V)) < theta:
            break

        prev_V = V.copy()

    return V

def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return new_pi

def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]

    while True:
        old_pi = {s:pi(s) for s in range(len(P))}

        V = policy_evaluation(pi, P, gamma=0.99)
        pi = policy_improvement(V, P, gamma=0.99)

        if old_pi == {s:pi(s) for s in range(len(P))}:
            break
    
    return V, pi

def value_iteration(P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)

    while True:
        V = np.zeros(len(P), dtype=np.float64)
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * prev_V[next_state] * (not done))
        
        V = np.max(Q, axis=1)
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()

    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return V, pi