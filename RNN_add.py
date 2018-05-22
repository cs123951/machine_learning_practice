import csv
import numpy as np
import itertools
import nltk
from preprocessing import getSentenceData


class Tanh:
    def forward(self, x):
        return np.tanh(x)
    def backward(self, x, diff):
        output = np.tanh(x)
        return (1.0 - np.square(output)) * diff
    
class sigmoid:
    def forward(self, x):
        return 1.0/(1.0+np.exp(-x))
    def backward(self, x, diff):
        output = self.forward(x)
        return (1.0-output)*output*diff
    
class softmax:
    def predict(self, x):
        return np.exp(x)/np.sum(np.exp(x))
    
    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[0,y])
    
    def diff(self, x, y):
        probs = self.predict(x)
        probs[0,y] -= 1.0
        return probs


class MultiplyGate:
    def forward(self, x, w):
        return np.dot(x, w)
    
    def backward(self, x, w, dz):
        dw = np.dot(x.T, dz)
        dx = np.dot(dz, w.T)
        return dw, dx
    
class AddGate:
    def forward(self, x1, x2):
        return x1 + x2
    
    def backward(self, x1, x2, dz):
        dx1 = dz
        dx2 = dz
        return dx1, dx2


class RNNLayer:
    def forward(self, x, prev_s, U, W, V):
        self.mulu = mulGate.forward(x, U)
        self.mulw = mulGate.forward(prev_s, W)
        self.adduw = addGate.forward(self.mulu, self.mulw)
        self.state = activation.forward(self.adduw)
        self.mulv = mulGate.forward(self.state, V)
    
    def backward(self, x, prev_s, U, W, V, dmulv):       
        self.forward(x, prev_s, U, W, V)
        dV, dVx = mulGate.backward(self.state, V, dmulv)
        dadd = activation.backward(self.adduw, dVx)
        dmulu, dmulw = addGate.backward(self.mulu, self.mulw, dadd)
        dU, dUx = mulGate.backward(x, U, dmulu)
        dW, dWx = mulGate.backward(prev_s, W, dmulw)
        return dU, dW, dV


class RNN:
    def __init__(self, input_dim, hidden_nodes, output_dim, lr = 0.001, bptt_truncate = 4):
        self.input_dim = input_dim
        self.hidden_nodes = hidden_nodes
        self.output_dim = output_dim
        self.U = np.random.random([input_dim, hidden_nodes])*0.01
        self.W = np.random.random([hidden_nodes, hidden_nodes])*0.01
        self.V = np.random.random([hidden_nodes, output_dim])*0.01
        self.lr = lr
        self.bptt_truncate = bptt_truncate


    def forward(self, x):
        # the length of input sequence
        self.time_steps = x.shape[1]
        layers = []
        prev_s = np.zeros([1, self.hidden_nodes])
        for t in range(self.time_steps):
            layer = RNNLayer()
            input_vec = x[:,t]
            input_vec = np.reshape(input_vec, [1,x.shape[0]])
            layer.forward(input_vec, prev_s, self.U, self.W, self.V)
            prev_s = layer.state
            layers.append(layer)
        return layers
    
    def backward(self, x, y):
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)
        layers = self.forward(x)
        for t in range(self.time_steps):
            dmulv = output.diff(layers[t].mulv, y[t])
            input_vec = x[:,t]
            input_vec = np.reshape(input_vec, [1, x.shape[0]])
            prev_s = np.zeros([1,self.hidden_nodes])
            dU_t, dW_t, dV_t = layers[t].backward(input_vec, prev_s, self.U, self.W, self.V, dmulv)
            for i in range(t-1,max(-1, t-self.bptt_truncate-1),-1):
                input_vec = x[:, i]
                input_vec = np.reshape(input_vec, [1, x.shape[0]])
                prev_s_i = np.zeros([1,self.hidden_nodes]) if i == 0 else layers[i-1].state
                dU_i, dW_i, dV_i = layers[i].backward(input_vec, prev_s_i, self.U, self.W, self.V, dmulv)
                dU_t += dU_i
                dW_t += dW_i
                dV_t += dV_i
            dU += dU_t
            dW += dW_t
            dV += dV_t
        return dU, dW, dV
    
    def sgd_optimizer(self, x, y, lr):
        dU, dW, dV = self.backward(x,y)
        self.U -= lr*dU
        self.W -= lr*dW
        self.V -= lr*dV
    
    def caculate_loss(self, x, y):
        loss = 0.0
        for example in range(len(y)):
            single_loss = 0.0
            layers = self.forward(x[example])
            for j,layer in enumerate(layers):                
                single_loss += output.loss(layer.mulv, y[example][j])
            loss += (single_loss/len(layers))
        return loss/len(y)
            
    
    def train(self, x, y, lr=0.005, nepoch=100, evaluate_loss_after=5):     
        for epoch in range(nepoch):
            if epoch % evaluate_loss_after == 0:
                loss = self.caculate_loss(x,y)
                print("Epoch=%d   Loss=%f" % (epoch, loss))
            for i in range(len(y)):
                self.sgd_optimizer(x[i], y[i], lr) # x[i], y[i] is a list

    def predict(self, x):
        output = softmax()
        layers = self.forward(x)
        predict_y = [np.argmax(output.predict(layer.mulv)) for layer in layers]
        return predict_y


mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()
output = softmax()
input_dim = 2
hidden_dim = 16
output_dim = 2


model = RNN(input_dim, hidden_dim, output_dim)

def generate_data(binary_dim, largest_number, int2binary):
    a = np.random.randint(largest_number/2)
    b = np.random.randint(largest_number/2)
    c = a + b
    return a,b,c,int2binary[a], int2binary[b], int2binary[c]

int2binary = {}
binary_dim = 8
largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)

for i in range(largest_number):
    int2binary[i] = binary[i]

X_train = []
y_train = []
for i in range(1000):
    _, _, _, a, b, c = generate_data(binary_dim, largest_number, int2binary)
    x = np.stack((a, b))
    y = np.array(c)
    X_train.append(x[:,::-1]) # because we caculate from right to left
    y_train.append(y[::-1])

losses = model.train(X_train, y_train, lr=0.005, nepoch=10, evaluate_loss_after=1)

# test
# the input and predict result should be reversed
inta, intb, intc, a, b, c = generate_data(binary_dim, largest_number, int2binary)
x = np.stack((a, b))
y = np.array(c)
print("input:")
print(str(inta)+' + '+ str(intb) + ' = ', str(intc))
print(a, b, c)
print("predict: ")
predict_y = model.predict(x[:,::-1])
print(predict_y[::-1])
inty = 0
index = 0
for i in predict_y:
    inty += pow(2,index)*i
    index += 1
print(inty)


