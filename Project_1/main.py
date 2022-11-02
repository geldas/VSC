import os

from tkinter import *
from tkinter import filedialog
from tkinter import Tk, font
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import csv
from threading import *

import simple_ann

# Initial weights and biases of the ANN are applied when the program starts. The files are stored in the init folder.
# The program allows the user to train the ANN again, using either own trainig data or letting the program generating randomly trainig data.
# Results of the training and trainig data are saved to the folder setup where new folder named according to time is created and.
# There are two validation data files with 5000 and 10000 randomly generated points that can be used for visualisation of results.
# The inputs can be tested by choosing .csv file, entering values to entries or by clicking in the area of the graph. 
# Results in the form of .csv file are saved to the results folder, they are written to the command line and are drawn to graph.

class Home(Frame):
    """Main Frame of GUI.
        
    """

    def __init__(self, parent, ann):
        """Initializes Main Frame

        """
        Frame.__init__(self, parent)
        self.load_path_inputs = ""
        self.x1_input = IntVar(parent)
        self.x2_input = IntVar(parent)
        self.x1 = []
        self.x2 = []
        self.x1_entry = DoubleVar(parent)
        self.x2_entry = DoubleVar(parent)
        self.weights = None
        self.biases = None
        self.ann = ann
        self.alpha_entry = IntVar(parent)
        self.training_data = "Not set"
        self.open_folder = "Not set"
        self.open_ow = self.open_iw = self.open_ob = self.open_ib = "Not set"
        self.epoch_entry = IntVar(parent)
        self.train_data_count_entry = IntVar(parent)

        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=8)

        #================================================================================================
        #left side of the GUI (settings)
        frame_left = Frame(self,width=400,height=650)
        frame_left.pack(side=LEFT,padx=20,pady=5)
        frame_left.pack_propagate(False)
        #----------------------------------------------------------------
        #import weights and biases
        label_import_settings=Label(frame_left,text="IMPORT SETTINGS:")
        label_import_settings.pack(side=TOP,pady=[10,0],anchor=NW)
        label_import_settings.config(font=("TkDefaultFont",12))
        label_load_settings_folder=Label(frame_left, text="IMPORT ANN SETTINGS (WEIGHTS AND BIASES):")
        label_load_settings_folder.pack(side=TOP,padx=[5,0],anchor=NW)
        
        frame_open_folder = Frame(frame_left)
        frame_open_folder.pack(side=TOP,anchor=NW)
        button_open_folder = Button(frame_open_folder, text="CHOOSE FOLDER", command=self.open_setup_folder)
        button_open_folder.pack(side=LEFT,padx=[10,0],pady=[0,10],anchor=NW)
        label_load_settings_files=Label(frame_left, text="CHOOSE INDIVIDUAL FILES:")
        label_load_settings_files.pack(side=TOP,padx=[5,0],anchor=NW)
        
        frame_open_output_weights = Frame(frame_left)
        frame_open_output_weights.pack(side=TOP,anchor=NW)
        button_open_output_weights = Button(frame_open_output_weights, text="OUTPUT LAYERS WEIGHTS", command=lambda:(self.open_setup_weights(self.ann.output,self.label_open_ow)))
        button_open_output_weights.pack(side=TOP,padx=[10,0],pady=[0,10],anchor=NW)
        self.label_open_ow=Label(frame_open_output_weights, text=self.open_ow)
        self.label_open_ow.pack(side=TOP,padx=[15,0],pady=[0,5],anchor=NW)
        self.label_open_ow.config(font=("TkDefaultFont",6))
        
        frame_open_output_biases = Frame(frame_left)
        frame_open_output_biases.pack(side=TOP,anchor=NW)
        button_open_output_biases = Button(frame_open_output_biases, text="OUTPUT LAYERS BIASES", command=lambda:(self.open_setup_biases(self.ann.output,self.label_open_ob)))
        button_open_output_biases.pack(side=TOP,padx=[10,0],pady=[0,10],anchor=NW)
        self.label_open_ob=Label(frame_open_output_biases, text=self.open_ob)
        self.label_open_ob.pack(side=TOP,padx=[15,0],pady=[0,5],anchor=NW)
        self.label_open_ob.config(font=("TkDefaultFont",6))
        
        frame_open_input_weights = Frame(frame_left)
        frame_open_input_weights.pack(side=TOP,anchor=NW)        
        button_open_hidden_weights = Button(frame_open_input_weights, text="HIDDEN LAYERS WEIGHTS", command=lambda:(self.open_setup_weights(self.ann.hidden,self.label_open_iw)))
        button_open_hidden_weights.pack(side=TOP,padx=[10,0],pady=[0,10],anchor=NW)
        self.label_open_iw=Label(frame_open_input_weights, text=self.open_iw)
        self.label_open_iw.pack(side=TOP,padx=[15,0],pady=[0,5],anchor=NW)
        self.label_open_iw.config(font=("TkDefaultFont",6))
        
        frame_open_input_biases = Frame(frame_left)
        frame_open_input_biases.pack(side=TOP,anchor=NW)
        button_open_hidden_biases = Button(frame_open_input_biases, text="HIDDEN LAYERS BIASES", command=lambda:(self.open_setup_biases(self.ann.hidden,self.label_open_ib)))
        button_open_hidden_biases.pack(side=TOP,padx=[10,0],pady=[0,10],anchor=NW)
        self.label_open_ib=Label(frame_open_input_biases, text=self.open_ib)
        self.label_open_ib.pack(side=TOP,padx=[15,0],pady=[0,5],anchor=NW)
        self.label_open_ib.config(font=("TkDefaultFont",6))
        #----------------------------------------------------------------
        #import inputs
        label_import_inputs=Label(frame_left, text="IMPORT INPUTS:")
        label_import_inputs.pack(side=TOP,pady=[10,0],anchor=NW)
        label_import_inputs.config(font=("TkDefaultFont",12))

        label_load_settings=Label(frame_left, text="IMPORT INPUTS FROM FILE:")
        label_load_settings.pack(side=TOP,pady=[10,5],anchor=NW)
        button_open_file_input = Button(frame_left, text="CHOOSE FILE", command=self.open_setup_inputs)
        button_open_file_input.pack(side=TOP,pady=[0,5],anchor=NW)

        label_write_settings=Label(frame_left, text="SET INPUTS MANUALLY:")
        label_write_settings.pack(side=TOP,pady=[0,5],anchor=NW)
        
        frame_x = Frame(frame_left)
        frame_x.pack(side=TOP,anchor=NW)
        label_x1 = Label(frame_x,text="x1:")
        label_x1.pack(side=LEFT,anchor=NW)
        self.entry_x1 = Entry(frame_x,textvariable=self.x1_entry,width=7)
        self.entry_x1.pack(side=LEFT,anchor=NW)
        label_x2 = Label(frame_x,text="x2:")
        label_x2.pack(side=LEFT,anchor=NW)
        self.entry_x2 = Entry(frame_x,textvariable=self.x2_entry,width=7)
        self.entry_x2.pack(side=LEFT,anchor=NW)
        button_test_inputs_manual = Button(frame_x, text="TEST", command=lambda:(self.test_inputs([float(self.x1_entry.get())], [float(self.x2_entry.get())])))
        button_test_inputs_manual.pack(side=LEFT,padx=[10,0],anchor=NW)        
        #----------------------------------------------------------------
        #train
        label_train=Label(frame_left, text="TRAIN:")
        label_train.pack(side=TOP,pady=[10,0],anchor=NW)
        label_train.config(font=("TkDefaultFont",12))

        # frame_alpha = Frame(frame_left)
        # frame_alpha.pack(side=TOP,anchor=NW)
        # label_alpha=Label(frame_alpha, text="Choose learning rate(default 0.05):")
        # label_alpha.pack(side=LEFT,pady=[0,5],anchor=NW)
        # self.entry_alpha = Entry(frame_alpha,textvariable=self.alpha_entry,width=8)
        # self.entry_alpha.pack(side=LEFT,anchor=NW)

        frame_training_data = Frame(frame_left)
        frame_training_data.pack(side=TOP,anchor=NW)
        button_open_file_input = Button(frame_training_data, text="CHOOSE TRAINING DATA", command=self.open_training_data)
        button_open_file_input.pack(side=LEFT,pady=[0,5],anchor=NW)
        self.label_training_data=Label(frame_training_data, text=self.training_data)
        self.label_training_data.pack(side=LEFT,pady=[0,5],anchor=NW)
        
        frame_epoch_count = Frame(frame_left)
        frame_epoch_count.pack(side=TOP,anchor=NW)
        label_epoch=Label(frame_epoch_count, text="Epoch count (default 1000):")
        label_epoch.pack(side=LEFT,pady=[0,5],anchor=NW)
        self.entry_epoch = Entry(frame_epoch_count,textvariable=self.epoch_entry,width=8)
        self.entry_epoch.pack(side=LEFT,anchor=NW)
        
        frame_train_data_count = Frame(frame_left)
        frame_train_data_count.pack(side=TOP,anchor=NW)
        label_train_data_count=Label(frame_train_data_count, text="Train data count (default 1000 IN, 1000 OUT)")
        label_train_data_count.pack(side=LEFT,pady=[0,5],anchor=NW)
        self.entry_train_data_count = Entry(frame_train_data_count,textvariable=self.train_data_count_entry,width=8)
        self.entry_train_data_count.pack(side=LEFT,anchor=NW)

        button_tr = Button(frame_left, text="TRAIN", command=self.threading)
        button_tr.pack(side=TOP,pady=[0,10],anchor=NW)

        # self.label_train_result=Label(frame_left, text="")
        # self.label_train_result.pack(side=TOP,anchor=NW)

        #================================================================================================
        #right side
        self.frame_graph = Frame(self,width=400,height=650)
        self.frame_graph.pack(side=LEFT,padx=5,pady=5)

        #----------------------------------------------------------------
        #initialize graph canvas and draw ellipse
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel('X1')
        self.axes.set_ylabel('X2')
        self.axes.set_title('Binary Classifier')
        self.axes.set_xlim([-4,2])
        self.axes.set_ylim([2,5])
        x1_range = np.linspace(-4,2,100)
        x2_range = np.linspace(2,5,100)
        x1, x2 = np.meshgrid(x1_range,x2_range)
        y_plot = 0.4444444*(x1+2)**2 + 2.3668639*(x2-3)**2 -1
        self.axes.contour(x1,x2,y_plot,[0])
        self.axes.grid()
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas = FigureCanvasTkAgg(self.figure, self.frame_graph)  
        self.canvas.get_tk_widget().pack(side=TOP,anchor=NW)

        #----------------------------------------------------------------
        #load initial weights and biases
        self.open_setup_folder("./init")
        self.label_open_ow["text"] = "Initial setup"
        self.label_open_iw["text"] = "Initial setup"
        self.label_open_ob["text"] = "Initial setup"
        self.label_open_ib["text"] = "Initial setup"
        #================================================================================================

    def onclick(self,event):
        """The method that allows the user to test the ANN by setting the inputs by clicking on the graph. When clicked in the area of the graph,
        the coordinates are passed to the ANN.
        """
        self.test_inputs([event.xdata],[event.ydata])

    def open_training_data(self):
        """The method loads training data from the .csv file to the ANN. The delimiter can be either comma or semicolon. The separetor in the decimal
        can be either comma or dot. These data can be used to train the ANN.
        """
        training_data = filedialog.askopenfilename()
        try:
            self.ann.load_training_data(training_data)
            #self.label_train_result[self.ann]
        except:
            print("Cannot open training data!")

    def test_inputs(self,x1,x2):
        """The method passes testing inputs to the ANN and the results are printed to the command prompt, drawn in graph and saved as .csv file in the
        results folder.

        Parameters:
        x1: list
        x2: list

        Example:
        self.test_inputs([-1,1],[3,4])
        """
        print("testing")
        results = self.ann.test_inputs(x1,x2)
        inside = results[0]
        outside = results[1]
        x1_el = np.linspace(-4,2,100)
        x2_el = np.linspace(2,5,100)
        x1, x2 = np.meshgrid(x1_el,x2_el)
        y_plot = 0.4444444*(x1+2)**2 + 2.3668639*(x2-3)**2 -1
        self.axes.clear()
        self.axes.set_xlabel('X1')
        self.axes.set_ylabel('X2')
        self.axes.set_title('Binary Classifier')
        self.axes.set_xlim([-4,2])
        self.axes.set_ylim([2,5])
        self.axes.contour(x1,x2,y_plot,[0])
        self.axes.grid()
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.axes.scatter(inside[0],inside[1], marker="+", color = 'blue', label = "INSIDE", s=20,alpha=0.7)
        self.axes.scatter(outside[0],outside[1], marker="x", color = 'red', label = "OUTSIDE", s=20,alpha=0.7)   
        self.axes.legend(loc="upper right")
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=TOP,anchor=NW)

    def open_setup_inputs(self):
        """The method opens dataset of inputs stored in .csv file and passes them to ANN to be used during testing.
        """
        setup_file = filedialog.askopenfilename()
        self.x1.clear()
        self.x2.clear()
        with open(setup_file, newline='') as csvfile:
            setup_reader = csv.reader(csvfile, delimiter=';', quotechar='|')
            for row in setup_reader:
                try:
                    if ',' in row[0]:
                        x = row[0].replace(',','.')
                        self.x1.append(float(x))
                    else:
                        self.x1.append(float(row[0]))
                    if ',' in row[1]:
                        x = row[1].replace(',','.')
                        self.x2.append(float(x))
                    else:
                        self.x2.append(float(row[1]))
                except:
                    print("Cannot open file!")
        self.test_inputs(self.x1,self.x2)

    def open_setup_weights(self,layer,label,setup_file=None):
        """The method opens the .csv file with weights when the button is pressed. The values are applied  to the hidden or output layer

        Parameters:
        layer: class ann.Layer()
        label: tkinter Label
        setup_file: str; when None filedialog is opened
        """
        if setup_file==None:
            setup_file = filedialog.askopenfilename()
        try:
            self.ann.load_weights(layer,setup_file)
            label["text"] = setup_file
        except:
            label["text"] = "Cannot open file!"
        

    def open_setup_biases(self,layer,label,setup_file=None):
        """The method opens the .csv file with biases when the button is pressed. The values are applied to the hidden or output layer

        Parameters:
        layer: class ann.Layer()
        label: tkinter Label
        setup_file: str; when None filedialog is opened
        """
        if setup_file==None:
            setup_file = filedialog.askopenfilename()
        try:
            self.ann.load_biases(layer,setup_file)
            label["text"] = setup_file
        except:
            label["text"] = "Cannot open file!"

    def open_setup_folder(self, setup_folder=None):
        """The method opens the folder with settings. The weights and biases are applied to the hidden and output layer.

        Parameters:
        setup_folder: str; when None filedialog is opened
        """
        if setup_folder==None:
            setup_folder = filedialog.askdirectory()
        
        self.open_setup_weights(self.ann.output,self.label_open_ow,setup_folder+'/output_weights.csv')
        self.open_setup_biases(self.ann.output,self.label_open_ob,setup_folder+'/output_biases.csv')
        self.open_setup_weights(self.ann.hidden,self.label_open_iw,setup_folder+'/hidden_weights.csv')
        self.open_setup_biases(self.ann.hidden,self.label_open_ib,setup_folder+'/hidden_biases.csv')

    def threading(self):
        """Creates another thread for training.
        """
        t1=Thread(target=self.train)
        t1.setDaemon(True)
        t1.start()


    def train(self):
        """Method is called, when the button Train is pressed. The ANN is then trained with trainig data or with randomly generated trainig data.
        """
        if self.training_data == "Not set":
            self.ann.train(self.train_data_count_entry.get(), self.epoch_entry.get(), 1000)
        else:
            self.ann.train(self.train_data_count_entry.get(), self.epoch_entry.get(), 1000, True)


def main():
    ann = simple_ann.ANN(0.01)
    root = Tk()
    root.title("Simple binary classificator")
    root.minsize(800,650)
    t = Home(root, ann)
    t.pack(side=LEFT,anchor=NW)
    root.mainloop()
    
if __name__ == '__main__':
    main()