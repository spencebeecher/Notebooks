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
    
    
total_params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

def countZeroWeights(model):
    zeros = 0
    for param in model.parameters():
        if param is not None:
            zeros += torch.sum((param == 0).int()).data[0]
    return zeros


def predict(puz, model, is_only_blanks=True):
    model.eval()
    predictions = model(Variable(torch.cuda.FloatTensor([puz])))
    predictions = predictions[0]
    
    predictions = F.softmax(predictions.reshape((9, 81)), dim=0)
    predictions = predictions.reshape(9*81)
    
    predictions = np.array(predictions.tolist())
    min_pred = 0
    
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

def predict_puzzle_until_wrong(puzzle, answer, model):
    
    num_zeros = np.sum(puzzle == 0)
    r = matrix_to_one_hot(puzzle)
    a = matrix_to_one_hot(answer)
    for _ in range(num_zeros):
        pred = predict_best(r, model)
        
        if np.sum(a & pred) == 1:

            r = pred + r
        else:
            r = one_hot_to_matrix(r)
            p = one_hot_to_matrix(pred)            
            return r - p, r
        
    r = one_hot_to_matrix(r)
    return r

def predict_cell(x,y, puz, model):
    preds = predict(matrix_to_one_hot(puz), model)
    preds = preds.reshape((9, 9, 9))
    ret = []
    for i in range(9):
        ret.append([i+1, preds[i,y,x]])
    return ret

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def eval_and_score_puzzle(model, file_name, solver):
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
        p = solver(puz, model)
        score = accuracy(sln, puz, p)
        scores.append(score)

    
    return scores

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
        
        sp = line.strip().split(',')
        if len(sp) != 2:
            f.close()
            break
        
        xx, yy = sp[0], sp[1]

        xx = np.array([int(v) for v in xx]).reshape((9,9))
        yy = np.array([int(v) for v in yy]).reshape((9,9))

        retx.append(xx)
        rety.append(yy)
        
        
    x = list([matrix_to_one_hot(v) for v in retx])

    y = list([matrix_to_one_hot(v) for v in rety])
    
    return x, y

def get_tensors(x, y):
    #tensor_x = Variable(torch.Tensor(x))
    #tensor_y = Variable(torch.Tensor(y), requires_grad=False)   
    #return tensor_x, tensor_y    
    x = torch.cuda.FloatTensor(x)
    tensor_x = Variable(x)
    y = torch.cuda.FloatTensor(y)
    tensor_y = Variable(y, requires_grad=False)
    
    return tensor_x, tensor_y

def get_file_tensors(f, batch_size):
    x, y = get_x_y(f, batches=batch_size)
    return get_tensors(x, y)

def run_training(file_name, model, num_examples, epochs, batch_size, learning_rate = 0.0001,
                 restart_learning_rate = True, use_targets=True, grow_batch_size=False,
                 weight_decay=0.0):
    t0 = time.time()
    
    
    m_name = str(model) + f' epochs:{epochs} batch_size:{batch_size} ' \
        + f'use_targets:{use_targets} learning_rate:{learning_rate} grow_batch_size:{grow_batch_size}'

    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    #learning_rate = 0.5
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    loss_fn = torch.nn.MultiLabelSoftMarginLoss(reduce=False)
    
    loss_results = []
    time_results = []
    
    

    for epoch in range(epochs):
        
        if restart_learning_rate:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        with open(file_name, 'r') as f:
            test_x, test_y, test_target = get_file_tensors(f, 400)

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

class FileTrainingData:
    def __init__(self, file_name, offset, batch_size):
        self.file_name = file_name
        self.offset = offset
        self.batch_size = batch_size
        self.file = open(self.file_name, 'r')
        
    def __iter__(self):
        if self.file:
            self.file.close()
        self.file = open(self.file_name, 'r')
        _ = get_file_tensors(self.file, self.offset)
        return self

    def __next__(self):
        x, y = get_file_tensors(self.file, self.batch_size)

        if len(x) == 0:
            self.file.close()
            self.file = None
            raise StopIteration
        
        return (x,y)

    def __str__(self):
        return f'FileTrainingData {str(self.file_name)} {str(self.offset)} {str(self.batch_size)}'

    
class CachedFileTrainingData:
    def __init__(self, file_name, offset, batch_size):
        self.file_name = file_name
        self.offset = offset
        self.batch_size = batch_size
        self.i = 0
        self.data = []
        
        with open(self.file_name, 'r') as file:
            _ = get_x_y(file, self.offset)
            
            while True:
                x, y = get_x_y(file, self.batch_size)
                
                if len(x) == 0:
                    break
                    
                self.data.append((x,y))
        
            
        
        
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i == len(self.data):
            raise StopIteration
        x, y = self.data[self.i]
        self.i += 1

        return get_tensors(x,y)
    
   

   

    def __str__(self):
        return f'CachedFileTrainingData {str(self.file_name)} {str(self.offset)} {str(self.batch_size)}'
        

import random                
class ZeroTrainingData:
    def __init__(self, training_data, percent_random):
        self.training_data = training_data
        self.percent_random = percent_random
        
     
    def __iter__(self):
        
        for x, y in self.training_data:
            if random.random() < self.percent_random:
                x[x!=0] = 0
            yield x, y         

    def __str__(self):
        return f'ZeroTrainingData {str(self.percent_random)} {str(self.training_data)}' 

                
class Trainer:
    def __init__(self, model, training_data, test_data, loss_fn, optimizer):
        self.model = model
        self.training_data = training_data
        self.test_data = test_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.loss_results = []
        self.time_results = []
        self.num_epochs = 0
    
    def __str__(self):
        return (f'{str(self.model)}\n\tepochs:{self.num_epochs}\n\t{str(self.training_data)}\n\t'
                f'{str(self.loss_fn)}\n\t{str(self.optimizer)}')
    

            
        
        
    def train_step(self):
        self.num_epochs = self.num_epochs + 1
        
        
        
        t0 = time.time()
        
        if len(self.time_results) > 0:
            min_time = self.time_results[-1]
        else:
            min_time = 0


        loss_results = []
        time_results = []

        test_x, test_y = self.test_data

        i = 0
        for x, y in self.training_data:
            i = i + 1
            
            if i % 500 == 1:
                self.model.eval()
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = self.model(test_x)

                # Compute and print loss.
                loss = self.loss_fn(y_pred, test_y)
                
                loss_results.append(loss.data.sum().tolist())
                time_results.append(int(time.time() - t0 + min_time))

            self.model.train()
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = self.model(x)

            # Compute and print loss.

            loss = self.loss_fn(y_pred, y)
            loss = loss.sum()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            
        self.loss_results.extend(loss_results)
        self.time_results.extend(time_results)
        