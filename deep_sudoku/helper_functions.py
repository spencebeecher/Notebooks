import numpy as np
import torch
import torch.cuda
import torch.nn
from torch.autograd import Variable
import time
import torch.nn.functional as F

# helper functions converting between a sudoku board and a 1 hot encoded version
# 9x9 matrix where each cell can take 1-9 values becomes a 1 dimensional array of size 9x9x9
def matrix_to_one_hot(m):
    '''Make a 1-hot encoded version of an input sudoku matrix'''
    ret = np.zeros((9,9*9), dtype=int)
    
    mtx = m.flatten()
    
    for v in range(9):
        
        index = mtx==(v+1)
        
        ret[v][index] = 1
        
    return ret.reshape(9*9*9)


def one_hot_to_matrix(oh):
    '''Make convert a one hot encoded sudoku matrix into a sudoku 
    matrix'''
    ret = np.zeros(81, dtype=int)
    
    mtx = oh.reshape((9, 9*9))
    
    for v in range(9):   
        ret[mtx[v]==1] = v+1
    
    return ret.reshape(9,9)
    
    
# load puzzles from disk
def get_x_y(f, batches=1):
    '''Read cached sudoku puzzles/answers from disk. Convert them to 1 hot encoded versions'''
    
    if f.closed:
        return [] ,[]
    
    retx = []
    rety = []
    for _ in range(batches):
        line = f.readline()
        
        if f.closed:
            break
        
        xx, yy = line.strip().split(',')

        xx = np.array([int(v) for v in xx]).reshape((9,9))
        yy = np.array([int(v) for v in yy]).reshape((9,9))

        retx.append(xx)
        rety.append(yy)
        
        
    x = list([matrix_to_one_hot(v) for v in retx])

    y = list([matrix_to_one_hot(v) for v in rety])
    
    return x, y


def predict(puz, model, is_only_blanks=True):
    model.eval()
    predictions = model(Variable(torch.cuda.FloatTensor([puz])))
    predictions = predictions[0]
    
    predictions = F.softmax(predictions.reshape((9, 81)), dim=0)
    predictions = predictions.reshape(9*81)
    
    predictions = np.array(predictions.tolist())
    min_pred = predictions.min() - 1
    
    if is_only_blanks:
        for i, e in enumerate(puz != 0):
            if e:
                predictions[i] = min_pred
                curr = i
                curr += 81
                while curr % 729 != i:
                    predictions[curr % 729] = min_pred
                    curr += 81  
        
    return predictions
    
def predict_best(puz, model):    
    predictions = predict(puz, model)
    
    ret = np.zeros(81*9, dtype=int)
        
    ret[np.argmax(predictions)] = 1  
    
    return ret


def predict_puzzle(puzzle, model):
    
    num_zeros = np.sum(puzzle == 0)
    r = matrix_to_one_hot(puzzle)
    for _ in range(num_zeros):
        pred = predict_best(r, model)
        
        r = pred + r
        
    r = one_hot_to_matrix(r)
    return r


def accuracy(answer, puzzle, prediction):
    num_zeros = np.sum(puzzle == 0)
    return ((prediction == answer).sum() - (81 - num_zeros)), num_zeros


def try_complete_sudoku(in_file_name, out_file_name, model, limit=10):
    i = 0
    with open(in_file_name, 'r') as in_file:
        with open(out_file_name, 'w') as out_file:
            for line in in_file:
                i+=1
                
                if i > limit:
                    return
                
                quiz, solution = line.strip().split(",")
                
                puz = []
                sln = []
                
                puz.append([int(c) for c in quiz])
                sln.append([int(c) for c in solution])
                
                puz = np.array(puz).reshape((-1, 9, 9))
                sln = np.array(sln).reshape((-1, 9, 9))
                
                
                p = predict_puzzle(puz, model)
                
                new_puz = sln.copy()
                new_puz[sln!=p] = 0
                
                pstring = lambda x: ''.join([str(x) for x in x.reshape(-1)])
                
                if np.sum(sln!=p) > 0:
                    out_file.write('{},{}\n'.format(pstring(new_puz), pstring(sln)))
                
def eval_and_score_puzzle(model, file_name):
    kaggle_puz = []
    kaggle_sln = []
    for i, line in enumerate(open(file_name, 'r').read().splitlines()):
        try:
            quiz, solution = line.split(",")
            kaggle_puz.append([int(c) for c in quiz])
            kaggle_sln.append([int(c) for c in solution])
        except:
            pass


    kaggle_puz = np.array(kaggle_puz).reshape((-1, 9, 9))
    kaggle_sln = np.array(kaggle_sln).reshape((-1, 9, 9))

    scores = []
    for puz, sln in zip(kaggle_puz, kaggle_sln):
        p = predict_puzzle(puz, model)
        score = accuracy(sln, puz, p)
        scores.append(score)

    #print(scores)
    #print(np.mean(scores))
    #print('')
    return scores
                
def get_tensors(x, y):
    #tensor_x = Variable(torch.Tensor(x))
    #tensor_y = Variable(torch.Tensor(y), requires_grad=False)   
    #return tensor_x, tensor_y    
    x = torch.cuda.FloatTensor(x)
    tensor_x = Variable(x)
    y = torch.cuda.FloatTensor(y)
    tensor_y = Variable(y, requires_grad=False)
    
    
    target = x.reshape(-1, 9, 81).sum(1)==0
    target = target.repeat(1, 9).reshape(-1, 9, 81).reshape(-1, 9*81)
    
    return tensor_x, tensor_y, target

def run_training(file_name, model, num_examples, epochs, batch_size, learning_rate = 0.0001,
                 restart_learning_rate = True, use_targets=True, grow_batch_size=False):
    t0 = time.time()
    
    
    m_name = str(model) + f' epochs:{epochs} batch_size:{batch_size} ' \
        + f'use_targets:{use_targets} learning_rate:{learning_rate} grow_batch_size:{grow_batch_size}'

    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #learning_rate = 0.5
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    loss_fn = torch.nn.MultiLabelSoftMarginLoss(reduce=False)
    
    loss_results = []
    time_results = []
    
    def get_file_tensors(f, is_soft_loss, batch_size):
        x, y = get_x_y(f, batches=batch_size)
        return get_tensors(x, y)

    for epoch in range(epochs):
        
        if restart_learning_rate:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        with open(file_name, 'r') as f:
            test_x, test_y, test_target = get_file_tensors(f, True, 400)

            for t in range(int(num_examples/batch_size)):

                if t % 500 == 1:
                    model.eval()
                    # Forward pass: compute predicted y by passing x to the model.
                    y_pred = model(test_x)

                    # Compute and print loss.
                    loss = loss_fn(y_pred * test_target.type(torch.cuda.FloatTensor), 
                                   test_y * test_target.type(torch.cuda.FloatTensor))
                    #loss = loss * test_target.type(torch.cuda.FloatTensor)
                    #print('test error\t', t, '\t', loss.data[0].tolist())
                    loss_results.append(loss.data.sum().tolist())
                    time_results.append(int(time.time() - t0))


                else:
                    model.train()
                    # Before the backward pass, use the optimizer object to zero all of the
                    # gradients for the variables it will update (which are the learnable
                    # weights of the model). This is because by default, gradients are
                    # accumulated in buffers( i.e, not overwritten) whenever .backward()
                    # is called. Checkout docs of torch.autograd.backward for more details.
                    optimizer.zero_grad()
                    
                    bs = batch_size * (epoch+1) if grow_batch_size else batch_size
                    
                    x, y, target = get_file_tensors(f, True, bs)
                    
                    if len(x) == 0:
                        break
                    
                    # Forward pass: compute predicted y by passing x to the model.
                    y_pred = model(x)

                    # Compute and print loss.
                    if use_targets:
                        loss = loss_fn(y_pred * target.type(torch.cuda.FloatTensor), 
                                       y * target.type(torch.cuda.FloatTensor))
                        loss = loss.sum() / target.type(torch.cuda.FloatTensor).sum()
                    else:
                        loss = loss_fn(y_pred, y)
                        loss = loss.sum()
                    
                  
                    # Backward pass: compute gradient of the loss with respect to model
                    # parameters
                    loss.backward()

                    # Calling the step function on an Optimizer makes an update to its
                    # parameters
                    optimizer.step()
    
    return (m_name, loss_results, time_results)