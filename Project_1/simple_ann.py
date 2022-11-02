import numpy as np
from numpy import savetxt
import random
import time
import os
import csv

# Simple ANN with 2 layers was created. The hidden layer has 8 neurons and the output layer has one neuron.
# The activation function for both layers is sigmoid function.
# The ANN is trained on randomly generated dataset.
# Main goal was to implement simple own neural network rather than using fameworks (Keras, PyTorch etc.) to
# better understand the topic.
# Another improvments could be in trying to shorten learning time (accepted training took about 15 minutes
# to train on 10000 data for 1000 epoch). The neural network is designed only for two layers, by improving 
# another layers could be added to create more complex neural network.


class Layer:
    def __init__(self, input_n, neuron_n, training_rate,is_output = False):
        """Layer of simple binary classificator.

        Parameters:
        input_n: int, inputs number
        neuron_n: int, number of neurons
        training_rate: float
        is_output: bool
        """

        self.weights = 0.10*np.random.randn(input_n, neuron_n)
        self.biases = np.zeros([1,neuron_n])
        self.inputs = np.zeros([input_n,neuron_n])
        self.delta_weights = np.zeros([input_n, neuron_n])
        self.delta_biases = None
        self.delta_w = 0
        self.delta_b = 0
        self.output_preactivation = 0
        self.output_postactivation = 0
        self.is_output = is_output
        self.output_l = None
        self.training_rate = training_rate
        self.train_data = []
        self.loss_last = 0

    def forward(self, inputs):
        """Method for forward pass through layer.

        Parameters:
        inputs: numpy array
        """
        self.inputs = inputs
        self.output_preactivation = np.dot(inputs, self.weights) + self.biases
        self.output_postactivation = self.activation_sigmoid(self.output_preactivation)
    
    def backpropagation(self, targets = None):
        """Method for backpropagation 
        """
        if self.is_output:
            self.delta = (targets-self.output_postactivation)*self.output_postactivation*(1-self.output_postactivation)
            self.delta_weights = self.inputs*(self.training_rate*self.delta)
            self.delta_weights = np.transpose(self.delta_weights)
            self.delta_biases = self.training_rate*self.delta
        else:
            self.delta = self.output_l.delta*self.output_l.weights
            self.delta = sum(self.delta)
            self.delta = self.delta*self.output_postactivation*(1-self.output_postactivation)
            for j in range(self.delta_weights.shape[0]):
                for i in range(self.delta_weights.shape[1]):
                    self.delta_weights[j,i] = self.training_rate*self.delta[0,i]*self.inputs[j]
            self.delta_biases = self.training_rate*self.delta
    
    def update_deltas(self):
        """Method that updates updates weights and biases after backpropagation.
        """
        self.weights += self.delta_weights
        self.biases += self.delta_biases

    def activation_sigmoid(self, x):
        """Sigmoid activation function.

        Parameters:
        x: numpy array

        returns inputs after activation function
        """
        return 1 / (1 + np.exp(-x))
    
    def derivative_sigmoid(self, x):
        """Method for derivative of sigmoid function. (not used)

        Parameters:
        x: numpy array
        """
        return self.activation_sigmoid(x)*(1-self.activation_sigmoid(x))

    def connect_layer(self, output_l):
        """Method, that connects layer to the output layer.

        Parameters:
        output_l: class Layer()
        """
        self.output_l = output_l

class ANN:
    def __init__(self, training_rate):
        """Simple ANN consisting of 1 hidden layer with 8 neurons and 1 output layer with 1 neuron that classifies if inputs are inside or outside
        of the ellipse specified by equotation: 0.4444444*(xa+2)**2 + 2.3668639*(xb-3)**2 = 1

        Parameters:
        training_rate: float
        """
        self.hidden = Layer(2,8,training_rate)
        self.output = Layer(8,1,training_rate,True)
        self.hidden.connect_layer(self.output)

    def validate(self, valid_count, hidden, output):
        """Validates ANN training.
        
        Parameters:
        valid_count: int
        hidden: class Layer()
        output: class Layer()
        
        returns list of lists [validation_ok, validation_ok_targets, validation_ok_results, validation_nok, validation_nok_targets, validation_nok_results]
        """
        i = 0
        j = 0
        #k = 0
        validation_ok = np.zeros([valid_count,2])
        validation_nok = np.zeros([valid_count,2])
        validation_ok_targets = np.zeros([valid_count,1])
        validation_nok_targets = np.zeros([valid_count,1])

        while i<valid_count or j<valid_count:
            xa = random.uniform(-4,2)
            xb = random.uniform(2,5)
            z_new = 0.4444444*(xa+2)**2 + 2.3668639*(xb-3)**2
            if z_new < 1 and i<valid_count:
                validation_ok[i,0] = xa
                validation_ok[i,1] = xb
                validation_ok_targets[i,0] = 1
                i += 1
            elif z_new >= 1 and j<valid_count:
                validation_nok[j,0] = xa
                validation_nok[j,1] = xb
                validation_nok_targets[j,0] = 0
                j += 1
        validation_ok_results = []
        validation_nok_results = []
        accuracy_ct = 0
        for valid in range(validation_ok.shape[0]):
            hidden.forward(validation_ok[valid])
            output.forward(hidden.output_postactivation)
            z = 0.4444444*(validation_ok[valid,0]+2)**2 + 2.3668639*(validation_ok[valid,1]-3)**2
            validation_ok_results.append([output.output_postactivation, z])
            if output.output_postactivation >= 0.5:
                accuracy_ct += 1

        for valid in range(validation_nok.shape[0]):
            hidden.forward(validation_nok[valid])
            output.forward(hidden.output_postactivation)
            z = 0.4444444*(validation_nok[valid,0]+2)**2 + 2.3668639*(validation_nok[valid,1]-3)**2
            validation_nok_results.append([output.output_postactivation, z])
            if output.output_postactivation < 0.5:
                accuracy_ct += 1
        
        accuracy = accuracy_ct/(valid_count*2)
        print("accuracy: %.4f" % (accuracy))

        return [validation_ok, validation_ok_targets, validation_ok_results, validation_nok, validation_nok_targets, validation_nok_results]

    def shuffle_training_set(self, inputs, targets):
        """Method that shuffles values of training set.

        Parameters:
        inputs: numpy array
        targets: numpy array

        returns list [inputs,targets], shuffled training set
        """
        training_samples = inputs.shape[0]

        shuffled = []
        not_shuffled = []
        for i in range(training_samples):
            not_shuffled.append(i)
        while len(not_shuffled) > 1:
            source_i = int(np.random.randint(len(not_shuffled)))
            source = not_shuffled[source_i]
            while source in shuffled:
                source_i = int(np.random.randint(len(not_shuffled)))
                source = not_shuffled[source_i]
            not_shuffled.remove(source)   
            dest_i = int(np.random.randint(len(not_shuffled)))
            dest = not_shuffled[dest_i]
            while dest in shuffled:
                dest_i = int(np.random.randint(len(not_shuffled)))
                dest = not_shuffled[dest_i]
            not_shuffled.remove(dest)
            shuffled.append(source)
            shuffled.append(dest)

            dest_copy = [inputs[dest,0], inputs[dest,1], targets[dest,0]]
            inputs[dest,0]=inputs[source,0]
            inputs[dest,1]=inputs[source,1]
            targets[dest,0]=targets[source,0]
            inputs[source,0]=dest_copy[0]
            inputs[source,1]=dest_copy[1]
            targets[source,0]=dest_copy[2]
        return [inputs,targets]

    def generate_train_data(self, count):
        """Method that randomly generates count x Inside training data and count x outside Training data. Returns list

        Parameters:
        count: int

        returns list shuffled [[np.array,np.array],np.array]
        """
        test_inputs = np.zeros([count*2,2])
        targets = np.zeros([count*2,1])
        i = 0
        j = 0
        k = 0
        while i<count or j<count:
            xa = random.uniform(-4,2)
            xb = random.uniform(2,5)
            z_new = 0.4444444*(xa+2)**2 + 2.3668639*(xb-3)**2
            if z_new < 1 and i<count:
                test_inputs[k,0] = xa
                test_inputs[k,1] = xb
                targets[k,0] = 1
                i += 1
                k += 1
            elif z_new >= 1 and j<count:
                test_inputs[k,0] = xa
                test_inputs[k,1] = xb
                targets[k,0] = 0
                j += 1
                k += 1
        shuffled = self.shuffle_training_set(test_inputs, targets)
        return shuffled
    
    def train(self, train_data_count, epoch_count, validate_count, train_data=False):
        """Main method for training. The ANN is trained for number of epoch with provided training data or randomly generated training data. The training data
        are then validated.

        Parameters:
        train_data_count: int
        epoch_count: int
        validate_count: int
        train_data: list
        """
        self.hidden = Layer(2,8,0.05)
        self.output = Layer(8,1,0.05,True)
        self.hidden.connect_layer(self.output)
        if train_data_count == 0:
            train_data_count = 1000
        if epoch_count == 0:
            epoch_count = 1000

        if not train_data:
            self.train_data = self.generate_train_data(train_data_count)
        
        test_inputs = self.train_data[0]
        targets = self.train_data[1]

        loss = np.zeros([epoch_count,1])
        for epoch in range(epoch_count):
            error = np.zeros([test_inputs.shape[0],1])
            for i in range(test_inputs.shape[0]):
                self.hidden.forward(test_inputs[i])
                self.output.forward(self.hidden.output_postactivation)
                self.output.backpropagation(targets[i,0])
                self.hidden.backpropagation()
                self.output.update_deltas()
                self.hidden.update_deltas()
                error[i,0] = (self.output.output_postactivation-targets[i,0])**2
            loss[epoch,0] = np.mean(error)

            print("epoch %d loss: %5.5f" % (epoch+1, loss[epoch,0]))
        self.loss_last = loss[loss.shape[0]-1,0]
        self.validate(100,self.hidden,self.output)
        self.save()
    
    def test_inputs(self,x1,x2):
        """ Test the input data with ANN. The results are saved to results folder and 

        Parameters:
        x1: list
        x2: list

        returns list of lists [[test_inside_x1,test_inside_x2],[test_outside_x1,test_outside_x2]]
        """
        test_inside_x1 = []
        test_inside_x2 = []
        test_outside_x1 = []
        test_outside_x2 = []
        test_output = []
        for i in range(len(x1)):
            test_input = np.zeros([1,2])
            test_input[0,0] = float(x1[i])
            test_input[0,1] = float(x2[i])
            if test_input[0,0] < -4 or test_input[0,0] > 2:
                print("%.4f out of boundaries (x1 >= -4 and x1 <= 2)" % (test_input[0,0]))
                continue
            if test_input[0,1] < 2 or test_input[0,1] > 5:
                print("%.4f out of boundaries (x2 >= 2 and x2 <= 5)" % (test_input[0,1]))
                continue
            self.hidden.forward(test_input)
            self.output.forward(self.hidden.output_postactivation)
            if self.output.output_postactivation > 0.5:
                test_inside_x1.append(x1[i])
                test_inside_x2.append(x2[i])
                test_output.append([x1[i],x2[i],"Inside"])
                print("%.4f;%.4f,Outside" % (x1[i], x2[i]))
            else:
                test_outside_x1.append(x1[i])
                test_outside_x2.append(x2[i])
                test_output.append([x1[i],x2[i],"Outside"])
                print("%.4f;%.4f,Outside" % (x1[i], x2[i]))

        if not os.path.isdir('./results'):
            os.mkdir('./results')
        file_path = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
        file_path = "./results/result_"+file_path+".csv"
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file,delimiter=';')
            writer.writerows(test_output)
        return [[test_inside_x1,test_inside_x2],[test_outside_x1,test_outside_x2]]

    def save(self):
        """Method that saves weights, biases and training data to new folder in setup folder.
        """
        setup_path = './setup/'
        #validation_path = './validation/'

        if not os.path.isdir('./setup'): # jesltli neexistuje slozka download, tak ji vytvor a pomoci skriptu URL.py stahni z nistu
            os.mkdir('./setup')

        file_nm = time.strftime("%Y_%m_%d_%H_%M_%S/", time.gmtime())
        setup_path += file_nm
        os.mkdir(setup_path)
        savetxt(setup_path +'output_weights.csv', self.output.weights, delimiter=';')
        savetxt(setup_path +'hidden_weights.csv', self.hidden.weights, delimiter=';')
        savetxt(setup_path +'output_biases.csv', self.output.biases, delimiter=';')
        savetxt(setup_path +'hidden_biases.csv', self.hidden.biases, delimiter=';')
        td = [[],[],[]]
        inputs = self.train_data[0]
        targets = self.train_data[1]
        for i in range(self.train_data[1].shape[0]):
            td[0].append(inputs[i,0])
            td[1].append(inputs[i,1])
            td[2].append(targets[i,0])
        with open(setup_path+'train_data.csv', 'w', newline='') as file:
            writer = csv.writer(file,delimiter=';')
            writer.writerows(td)
    
    def load_weights(self,layer,path):
        """Applies weights to specified layer.

        Parameters:
        layer: class Layer()
        path: str
        """
        layer.weights = np.loadtxt(path,delimiter=';')
    
    def load_biases(self,layer,path):
        """Applies biases to specified layer.

        Parameters:
        layer: class Layer()
        path: str
        """
        layer.biases = np.loadtxt(path,delimiter=';')
        
    def load_training_data(self,path):
        """Loads training data from file.

        Parameters:
        path: str
        """
        data = []
        with open(path, newline='') as csvfile:
            td = csv.reader(csvfile, delimiter=';')
            for row in td:
                data.append(row)
            samples_ct = len(data[0])
            test_inputs = np.zeros([samples_ct,2])
            targets = np.zeros([samples_ct,1])
            for i in range(samples_ct):
                test_inputs[i,0] = data[0][i]
                test_inputs[i,1] = data[1][i]
                targets[i,0] = data[2][i]
            self.train_data = [test_inputs,targets]