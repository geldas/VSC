import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import ann_model
from threading import *

class Gui(QMainWindow):
    def __init__(self):
        """ Main window of GUI.
        """
        super().__init__()
        self.a = ann_model.Project()
        self.a.rastrigin_ann()
        self.a.schwefel_ann()
        self.actFunc = "rastrigin"
        self.running = False

        self.setWindowTitle('VSC project 4')
        self.setMinimumSize(QSize(800, 700))
        self.content = QWidget()
        layout = QGridLayout()

        self.rast_ni_input = QLineEdit('10')
        self.rast_ni_ts = QLineEdit('5000')
        self.schwef_ni_input = QLineEdit('10')
        self.schwef_ni_ts = QLineEdit('5000')
        
        self.changeFunc = QPushButton("Change function")
        self.changeFunc.clicked.connect(self.changeFunction)
        self.chosenFunc = QLabel("Rastrigin")
        self.longRunningBtn = QPushButton("Start")
        self.longRunningBtn.clicked.connect(self.threading)
        self.runningLabel = QLabel("")
        self.resultLabel = QLabel("")
        self.lossLabel = QLabel("")
        self.epochLabel = QLabel("")

        layout.addWidget(QLabel('Rastrigin'), 0, 0)
        layout.addWidget(QLabel('Schwefel'), 0, 3)

        layout.addWidget(QLabel('Size of training set'), 1, 0)
        layout.addWidget(QLabel('Number of iterations'), 2, 0)
        layout.addWidget(self.rast_ni_input, 2, 1)
        layout.addWidget(self.rast_ni_ts, 1, 1)
        layout.addWidget(QLabel(self.a.su_rastr), 3, 0, 1, 3)

        layout.addWidget(QLabel('Size of training set'), 1, 3)
        layout.addWidget(QLabel('Number of iterations'), 2, 3)
        layout.addWidget(self.schwef_ni_input, 2, 4)
        layout.addWidget(self.schwef_ni_ts, 1, 4)
        layout.addWidget(QLabel(self.a.su_schwef), 3, 3, 1, 3)
        layout.addWidget(QLabel('Optimization method: Nelder-Mead'),4,0)
        layout.addWidget(self.changeFunc,5,0)
        layout.addWidget(self.chosenFunc,5,1)
        layout.addWidget(self.longRunningBtn,6,0)
        layout.addWidget(self.runningLabel,6,1)
        layout.addWidget(self.resultLabel,7,0,1,2)
        layout.addWidget(self.lossLabel,7,2)
        layout.addWidget(self.epochLabel,7,3)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111,projection='3d')

        self.axes.set_xlabel('X1')
        self.axes.set_ylabel('X2')
        self.axes.set_zlabel('Z')
        self.axes.set_title('Result')
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 7, 0, 10, 10)

        self.content.setLayout(layout)
        self.setCentralWidget(self.content)

    def draw_rastrigin(self):
        """Draws rastrigin function approximated by ANN and optimal values of each iteration.
        """
        self.axes.clear()
        self.axes.plot_surface(self.a.X_rast, self.a.Y_rast, self.a.Z_rast2)
        self.axes.scatter3D(self.a.opt_X1, self.a.opt_X2, self.a.opt_Z, color = "red")
        self.axes.set_xlabel('X1')
        self.axes.set_ylabel('X2')
        self.axes.set_zlabel('Z')
        self.axes.set_title('Result')
        self.axes.grid()
        self.canvas.draw()


    def draw_schwefel(self):
        """Draws Rastrigin function approximated by ANN and optimal values of each iteration.
        """
        self.axes.clear()
        self.axes.plot_surface(self.a.X_schwef, self.a.Y_schwef, self.a.Z_schwef2)
        self.axes.scatter3D(self.a.opt_X1, self.a.opt_X2, self.a.opt_Z, color = "red")
        self.axes.set_xlabel('X1')
        self.axes.set_ylabel('X2')
        self.axes.set_zlabel('Z')
        self.axes.set_title('Result')
        self.axes.grid()
        self.canvas.draw()

    def threading(self):
        """Starts new thread so the GUI does not freeze.
        """
        self.t1=Thread(target=self.train)
        self.t1.setDaemon(True)
        self.t1.start()

    def changeFunction(self):
        """When the appropriate button is clicked, the function is changed.
        """
        if self.actFunc == "rastrigin":
            self.actFunc = "schwefel"
            self.chosenFunc.setText("Schwefel")
        else:
            self.actFunc = "rastrigin"
            self.chosenFunc.setText("Rastrigin")

    def train(self):
        """Starts the main process of the project.
        """
        if not self.running:
            self.running = True
            self.runningLabel.setText("Running...")
            if self.actFunc == "rastrigin":    
                self.a.create_dataset(int(self.rast_ni_ts.text()))
                self.a.optimize_rastrigin(int(self.rast_ni_input.text()))
                self.a.graph_data_rast()
                self.draw_rastrigin()
                i = self.a.opt_Z.index(min(self.a.opt_Z))
                resTxt = "Optimal value at point x*opt [{x1:.3f},{x2:.3f}]: F(x*opt)={fun:.3f}"
                self.resultLabel.setText(resTxt.format(x1=self.a.opt_X1[i],x2=self.a.opt_X2[i],fun=self.a.opt_Z[i]))
                lossTxt = "Loss:{l:.3f}"
                self.lossLabel.setText(lossTxt.format(l=self.a.loss[-1]))
                self.epochLabel.setText("Epochs: 500")
            else:
                self.a.create_dataset(int(self.schwef_ni_ts.text()))
                self.a.optimize_schwefel(int(self.schwef_ni_input.text()))
                self.a.graph_data_schwef()
                self.draw_schwefel()
                i = self.a.opt_Z.index(min(self.a.opt_Z))
                resTxt = "Optimal value at point x*opt [{x1:.3f},{x2:.3f}]: F(x*opt)={fun:.3f}"
                self.resultLabel.setText(resTxt.format(x1=self.a.opt_X1[i],x2=self.a.opt_X2[i],fun=self.a.opt_Z[i]))
                lossTxt = "Loss:{l:.3f}"
                self.lossLabel.setText(lossTxt.format(l=self.a.loss[-1]))
                self.epochLabel.setText("Epochs: 500")
            self.running = False
            self.runningLabel.setText("")

def main():
    app = QApplication(sys.argv)
    window = Gui()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()