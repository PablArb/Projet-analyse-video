from PyQt5.QtWidgets import QWidget, QPushButton, QGridLayout, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import QByteArray


class ButtonMenu(QWidget):

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


class ImageDisplay(QWidget):

    def __init__(self, image):
        super().__init__()

        self.createLayout(image)

    def createLayout(self, image):
        self.label = QLabel()
        # create a QPixmap object from the image file
        self.pixmap =  QPixmap(QByteArray(image))
        # resize the pixmap
        self.scaled_pixmap = self.pixmap.scaled(360, 640)
        # set the scaled pixmap as the image for the label
        self.label.setPixmap(self.scaled_pixmap)

        self.label.mousePressEvent = self.getPixelValue

        # create vertical layout for image
        self.Layout = QVBoxLayout()
        self.Layout.addWidget(self.label)

    def changeTo(self, image):
        self.pixmap = QPixmap(QByteArray(image))
        self.label.setPixmap(self.pixmap)

    def getPixelValue(self, event):
        # get the position of the mouse click relative to the label
        position = event.pos()

        xcoeff = self.pixmap.height()//self.scaled_pixmap.height()
        ycoeff = self.pixmap.width()//self.scaled_pixmap.width()

        x, y = position.x(), position.y()
        xim, yim = x * xcoeff, y * ycoeff

        # get the pixel value at the clicked position
        pixel_value = self.scaled_pixmap.toImage().pixel(x, y)

        # print the RGB values of the pixel
        r, g, b, _ = QColor(pixel_value).getRgb()
        print(f'Pixel value: x:{x}, y:{y}, r:{r}, g:{g}, b:{b}')
        return (xim, yim), (r, g, b)
