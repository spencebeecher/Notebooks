import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

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

def create_matrix():
    return np.zeros(dtype=int, shape=(9,9))

def to_matrix(p):
    return np.array([np.array([int(i) for i in r]) for r in p.split('\n')])

# helper functions to pull associated rows, columns and boxes for a given cell.
get_row = lambda p,x,y: (p[x]).flatten()
get_col = lambda p,x,y: (p[:,y]).flatten()

def get_box(p, x, y):
    row_low = 3*int(x/3)
    row_high = 3*int(x/3)+3
    col_low = 3*int(y/3)
    col_high = 3*int(y/3)+3
    
    return (p[row_low:row_high,col_low:col_high]).flatten()

get_rect = [get_row, get_col, get_box]

def enumerate_rows(p):
    result = []
    for i in range(0, 9):
        result.append(p[i].flatten())
    return result

def enumerate_columns(p):
    result = []
    for i in range(0, 9):
        result.append(p[:,i].flatten())
    return result

def enumerate_boxes(p):
    result = []
    for x in range(0, 9, 3):
        for y in range(0, 9, 3):
            result.append(get_box(p,x,y))
    return result

def get_possibilities(p):
    """For each entry in the grid, list out all of the things it could be"""
    full = set(range(1,10))
    possibiliteis = np.array([[set() for x in range(1, 10)] for x in range(1, 10)])
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

def solve_step(p):
    """Attempt to solve the sudoku puzzle using 2 stratigies"""
    
    solution = create_matrix()

    is_update = False
    possibiliteis = get_possibilities(p)

    # if there is only one possible thing a grid item could be, then we have our answer!
    for c in range(0,9):
        for r in range(0,9):
            if p[r,c] == 0:   
                if len(possibiliteis[r,c]) == 1:
                    solution[r,c] = list(possibiliteis[r,c])[0]
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
                        solution[r,c] = poss.pop()
                        is_update = True
    if is_update:
        return solution
    else:
        return None

class Count(object):
    def __init__(self):
        self.c=0
    def increment(self):
        self.c += 1
    def get(self):
        return self.c

def solve(px, solution_list=None, step=0, count=None):
    if count is None:
        count = Count()
    count.increment()
    
    if step > 100 or count.get() > 1000:
        # sometimes we get unlucky and istead of waiting for
        # the very hard puzzle to complete we shortcircuit 
        # this is due to us getting unlucky with a recursion step
        raise ValueError('oops - unlucky branch in solver')

            
    p = px.copy()
    
    solution = solve_step(p)

    if solution is not None:
        if solution_list is None:
            solution_list = []

    
    while solution is not None:
        p += solution
        solution_list.append(p)
        solution = solve_step(p)
        
    if is_complete(p):
        if is_correct(p):
            return p, solution_list
        else:
            return None, None
    
    # find the first cell with multiple possibilities
    # make a guess!
    possibiliteis = get_possibilities(p)

    r_range = np.arange(9)
    c_range = np.arange(9)

    np.random.shuffle(r_range)
    np.random.shuffle(c_range)

    is_done = True
    for row in r_range:
        for col in c_range:
            poss_list = np.array(list(possibiliteis[row, col]))
            if len(poss_list) > 1:
                is_done = True
                break
        if is_done:
            break

    np.random.shuffle(poss_list)

    if len(poss_list) <= 1:
        return None, None

    for poss in poss_list:
        p[row, col] = poss

        if solution_list is not None:
            rs, slns = solve(p.copy(), solution_list + [p.copy()], step+1, count)
        else:
            rs, slns = solve(p.copy(), solution_list, step+1, count)

        if rs is not None:
            if is_complete(rs):
                if is_correct(rs):
                    return rs, slns
        p[row, col] = 0
                               
    return None, None

def solve_puzzle(px):
    result, steps = solve(px)
    res_steps = []
    curr = steps[0]
    res_steps.append(curr)
    
    # the solver has a bug in results generation
    # where it will return multiple identical board states (part of recursion)
    # filter these out
    for s in steps[1:]:
        if np.sum(s==curr) == 81:
            continue
        else:
            curr = s
            res_steps.append(curr)
    return res_steps

# for a puzzle, remove all spots at random, 
#check if the puzzle can find that piece via solve step
# if so recurse

def unsolve_puzzle(p):
    px = p.copy()
    
    r_range = np.arange(9)
    c_range = np.arange(9)

    np.random.shuffle(r_range)
    np.random.shuffle(c_range)
    
    unsolved_list = []
    
    for row in r_range:
        for col in c_range:
            
            v = px[row, col]
            px[row, col] = 0
            if not solve_step(px)[row, col] == v:
                px[row, col] = v
    
    return px
        