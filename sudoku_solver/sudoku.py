
# coding: utf-8

## Solving Sudoku with Python

# In[1]:

import pandas as pd
import numpy as np


# In[95]:

#puzzles 

p1="""000003600
000760009
000092780
201050803
056030140
403070506
062480000
800015000
007900000"""

p2="""093850000
500073000
007400000
409010208
062040710
701030406
000005300
000120009
000096870"""

#hard puzzle
p3="""000304900
040065000
006800000
430000108
090000020
201000035
000006300
000210060
008507000"""


# In[93]:

#convert strings to a matrix
def to_matrix(p):
    return np.array([np.array([int(i) for i in r]) for r in p.split('\n')])


# In[94]:

# helper functions to pull associated rows, columns and boxes for a given cell.
get_row = lambda p,x,y: (p[x]).flatten()
get_col = lambda p,x,y: (p[:,y]).flatten()
get_box = lambda p,x,y: (p[3*(x/3):3*(x/3)+3,3*(y/3):3*(y/3)+3]).flatten()

get_rect = [get_row, get_col, get_box]


# In[82]:

def get_possibilities(p):
    """For each entry in the grid, list out all of the things it could be"""
    full = set(range(1,10))
    possibiliteis = np.array([[set() for x in range(10)] for x in range(10)])
    for c in range(0,9):
        for r in range(0,9):
            candidates = set(full)
            if p[r,c] == 0:   
                for get_func in get_rect:
                    [candidates.remove(x) for x in set(get_func(p,r,c)) if x in candidates]
                possibiliteis[r,c] = candidates
            else:
                possibiliteis[r,c] = set([p[r,c]])
    return possibiliteis

def solve(px):
    """Attempt to solve the sudoku puzzle using 2 stratigies"""
    
    p = px.copy()
    is_update = True
    while is_update:
        is_update = False
        possibiliteis = get_possibilities(p)
        
        # if there is only one possible thing a grid item could be, then we have our answer!
        for c in range(0,9):
            for r in range(0,9):
                if p[r,c] == 0:   
                    if len(possibiliteis[r,c]) == 1:
                        p[r,c] = list(possibiliteis[r,c])[0]
                        is_update = True
                        
        # This is sort of the opposite of the above
        # If this is the only cell in the row, col, box than can be a number - then make it so
        for c in range(0,9):
            for r in range(0,9):
                if p[r,c] == 0:    
                    for get_func in get_rect:
                        #for each possibility, see if this is the only cell that can satisfy it
                        func_poss = [s for s in get_func(possibiliteis,r,c)]
                        func_poss.remove(possibiliteis[r,c])
                        poss = set(range(1,10))

                        for s in func_poss:
                            for e in s:
                                if e in poss:
                                    poss.remove(e)
                        if len(poss) == 1:
                            p[r,c] = poss.pop()
                            is_update = True

    return p


# In[60]:

def is_correct(p):
    """Are all of the rows, cols, and boxes completed (1-9 entries)"""
    full = set(range(1,10))
    for c in range(0,9):
        for r in range(0,9): 
            for get_func in get_rect:
                if len(set(get_func(p,r,c)).intersection(full)) != 9:
                    return False
    return True


def is_complete(p):
    """Are all of the rows, cols, and boxes non zero"""
    for c in range(0,9):
        for r in range(0,9): 
            if p[r,c] == 0:
                return False                
    return True


####### Let's try to solve some puzzles

# In[76]:

p1_solution = solve(to_matrix(p1))
print(p1_solution)
print(is_correct(p1_solution))
print(is_complete(p1_solution))


# In[79]:

p2_solution = solve(to_matrix(p2))
print(p2_solution)
print(is_correct(p2_solution))
print(is_complete(p2_solution))


# In[80]:

p3_solution = solve(to_matrix(p3))
print(p3_solution)
print(is_correct(p3_solution))
print(is_complete(p3_solution))


####### p1 and p2 worked, the third did not. This means that we are going to have to create condiditonal boards

# In[91]:

def solve_recursive(p):
    solution = solve(p)
    if is_complete(solution):
        if is_correct(solution):
            return solution
        else:
            return None
    
    
    # find the first cell with multiple possibilities
    # make a guess!
    possibiliteis = get_possibilities(solution)
    for c in range(0,9):
        for r in range(0,9):
            if len(possibiliteis[r,c]) > 1:
                for poss in possibiliteis[r,c]:
                    solution[r,c] = poss
                    rs = solve_recursive(solution)
                    # 
                    if rs != None:
                        return rs
    return None


# In[92]:

p3_solution = solve_recursive(to_matrix(p3))
print(p3_solution)
print(is_correct(p3_solution))
print(is_complete(p3_solution))


####### Huzzah!
