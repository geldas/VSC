from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize as opt
import random

class Project:
    def __init__(self):
        self.rast_bound = 5.12
        self.schwef_bound = 500
        self.opt_X1 = []
        self.opt_X2 = []
        self.opt_Z = []
    
    def rastrigin(self,X,Y):
        """Computes Rastrigin function.
        """
        return 10*2+X**2-10*np.cos(2*np.pi*X) + Y**2-10*np.cos(2*np.pi*Y)
    
    def schwefel(self, X,Y):
        """Computes Schwefel function.
        """
        return 418.9829*2 - (X*np.sin(np.sqrt(np.absolute(X)))+Y*np.sin(np.sqrt(np.absolute(Y))))

    def rastrigin_ann(self):
        """Creates ANN for Rastrigin function approximation.
        """
        self.rastr_net = Sequential(name='Rastrigin')
        self.rastr_net.add(Dense(256, input_dim=2, activation ='tanh'))
        self.rastr_net.add(Dense(256, activation ='tanh'))
        self.rastr_net.add(Dense(128, activation ='tanh'))
        self.rastr_net.add(Dense(128, activation ='tanh'))
        self.rastr_net.add(Dense(128, activation ='tanh'))
        self.rastr_net.add(Dense(1, activation ='linear'))
        self.rastr_net.compile(loss='mse', optimizer='Adam', metrics=['mae'])
        self.su_rastr =""
        self.rastr_net.summary(line_length=50, print_fn=lambda x: self.get_summary_rastr(x + '\n'))
        
    def get_summary_rastr(self, s):
        """Converts summary from of Rastrigin ANN model to string.
        """
        self.su_rastr += s
    
    def get_summary_schwef(self, s):
        """Converts summary from of Schwefel ANN model to string.
        """
        self.su_schwef += s

    def schwefel_ann(self):
        """Creates ANN for Schwefel function approximation.
        """
        self.schwefel_net = Sequential(name='Schwefel')
        self.schwefel_net.add(Dense(256, input_dim=2, activation='relu'))
        self.schwefel_net.add(Dense(256, activation='relu'))
        self.schwefel_net.add(Dense(128, activation='relu'))
        self.schwefel_net.add(Dense(128, activation='relu'))
        self.schwefel_net.add(Dense(128, activation='relu'))
        self.schwefel_net.add(Dense(1, activation='linear'))
        self.su_schwef =""
        self.schwefel_net.compile(loss='mse', optimizer='Adam', metrics=['mae'])
        self.schwefel_net.summary(line_length=50, print_fn=lambda x: self.get_summary_schwef(x + '\n'))

    def create_dataset(self,n):
        """Creates training set.
        """
        rast_bound = 5.12
        schwef_bound = 500
        self.b = 20
        v1 = np.linspace(-rast_bound,rast_bound,self.b)
        v2 = np.linspace(-schwef_bound,schwef_bound,self.b)

        self.X_rast, self.Y_rast = np.meshgrid(v1,v1)
        self.X_schwef, self.Y_schwef = np.meshgrid(v2,v2)

        self.Z_rast = self.rastrigin(self.X_rast, self.Y_rast)
        self.z_schwef = self.schwefel(self.X_schwef, self.Y_schwef)

        self.x1 = np.random.rand(n,2)*(rast_bound*2)-rast_bound
        self.x2 = np.random.rand(n,2)*(schwef_bound*2)-schwef_bound

        self.yt1 = np.reshape(self.rastrigin(self.x1[:, 0], self.x1[:, 1]), (self.x1.shape[0], 1))
        self.yt2 = np.reshape(self.schwefel(self.x2[:, 0], self.x2[:, 1]), (self.x2.shape[0], 1))

    def train_schwefel(self):
        """Training of the ANN for Schwefel function.
        """
        self.schwef_history = self.schwefel_net.fit(self.x2, self.yt2, epochs=500, batch_size=64,verbose=0)
        self.loss = self.schwef_history.history['loss']

    def train_rastrigin(self):
        """Training of the ANN for Rastrigin function.
        """
        self.rastr_history = self.rastr_net.fit(self.x1, self.yt1, epochs=500, batch_size=64,verbose=0)
        self.loss = self.rastr_history.history['loss']

    def predict_rastrigin(self,x):
        """Function that is needed for scipy.optimize library.
        """
        return self.rastr_net.predict(np.array([[x[0],x[1]]]))

    def predict_schwefel(self,x):
        """Function that is needed for scipy.optimize library.
        """
        return self.schwefel_net.predict(np.array([[x[0],x[1]]]))

    def optimize_rastrigin(self,n):
        """Training is repeated for n iterations. Rastrigin function.
        """
        initial_guess = np.zeros(2)
        initial_guess[0] = random.uniform(-self.rast_bound,self.rast_bound)
        initial_guess[1] = random.uniform(-self.rast_bound,self.rast_bound)
        self.opt_X1.clear()
        self.opt_X2.clear()
        self.opt_Z.clear()
        for i in range(n):
            self.train_rastrigin()
            result = opt.minimize(self.predict_rastrigin,initial_guess,method="nelder-mead")
            self.x1 = np.append(self.x1,[result['x']],axis=0)
            new_y = self.rastrigin(result['x'][0],result['x'][1])
            self.yt1 = np.append(self.yt1,result.fun)
            print("Iteration: %d: x(opt) = [%.4f,%.4f], F*(x*(opt)) = %.4f" % (i,result['x'][0],result['x'][1],result.fun))
            self.opt_X1.append(result['x'][0])
            self.opt_X2.append(result['x'][1])
            self.opt_Z.append(result.fun)
    
    def graph_data_rast(self):
        """Data for plotting graph are created. Rastrigin function.
        """
        self.Z_rast2 = np.zeros((self.b, self.b))
        for i in range(self.b):
            for j in range(self.b):
                self.Z_rast2[i, j] = self.rastr_net.predict(np.array([[self.X_rast[i,j], self.Y_rast[i,j]]]))

    def graph_data_schwef(self):
        """Data for plotting graph are created. Schwefel function.
        """
        self.Z_schwef2 = np.zeros((self.b, self.b))
        for i in range(self.b):
            for j in range(self.b):
                self.Z_schwef2[i, j] = self.schwefel_net.predict(np.array([[self.X_schwef[i,j], self.Y_schwef[i,j]]]))        

    def optimize_schwefel(self,n):
        """Training is repeated for n iterations. Schwefel function.
        """
        initial_guess = np.zeros(2)
        initial_guess[0] = random.uniform(-self.schwef_bound,self.schwef_bound)
        initial_guess[1] = random.uniform(-self.schwef_bound,self.schwef_bound)
        
        for i in range(n):
            self.train_schwefel()
            result = opt.minimize(self.predict_schwefel,initial_guess,method="nelder-mead")
            self.x1 = np.append(self.x1,[result['x']],axis=0)
            new_y = self.schwefel(result['x'][0],result['x'][1])
            self.yt1 = np.append(self.yt1,[new_y])
            #print("Iteration: %d: x(opt) = [%.4f,%.4f], F(x(opt)) = %.4f" % (i,result['x'][0],result['x'][1],new_y))
            print("Iteration: %d: x(opt) = [%.4f,%.4f], F*(x*(opt)) = %.4f" % (i,result['x'][0],result['x'][1],result.fun))
            self.opt_X1.append(result['x'][0])
            self.opt_X2.append(result['x'][1])
            self.opt_Z.append(result.fun)