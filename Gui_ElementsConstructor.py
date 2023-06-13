from PyQt5.QtWidgets import QWidget, QPushButton, QGridLayout, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QByteArray
from Modules import np


class CalibButtonMenu(QWidget):

    def __init__(self):
        super().__init__()

        self.createLayout()

    def createLayout(self):

        # create grid layout for menu
        self.menu_grid = QGridLayout()

        # create buttons
        self.button1 = QPushButton('Next', self)
        # add buttons to grid
        self.menu_grid.addWidget(self.button1, 0, 0)

        # create vertical layout for menu
        self.Layout = QVBoxLayout()
        self.Layout.addLayout(self.menu_grid)


class CalibImageDisplay(QWidget):

    def __init__(self, image, getPixelValue):
        super().__init__()
        self.drawnmarkers = []
        self.createLayout(image, getPixelValue)

    def createLayout(self, image, getPixelValue):
        self.label = QLabel()
        self.label.setPixmap(image)
        self.label.mousePressEvent = getPixelValue

        self.Layout = QVBoxLayout()
        self.Layout.addWidget(self.label)

    def update(self, image: QPixmap):
        self.label.setPixmap(image)
