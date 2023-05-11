from PyQt5.QtWidgets import QWidget, QPushButton, QGridLayout, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QColor


class ButtonMenu(QWidget):

    def __init__(self):
        super().__init__()

        self.createLayout()

    def createLayout(self):

        # create grid layout for menu
        menu_grid = QGridLayout()

        # create buttons
        button1 = QPushButton('Button 1')
        button2 = QPushButton('Button 2')
        button3 = QPushButton('Button 3')
        button4 = QPushButton('Button 4')
        button5 = QPushButton('Button 5')

        # add buttons to grid
        menu_grid.addWidget(button1, 0, 0)
        menu_grid.addWidget(button2, 0, 1)
        menu_grid.addWidget(button3, 1, 0)
        menu_grid.addWidget(button4, 1, 1)
        menu_grid.addWidget(button5, 2, 0, 1, 2)  # span two columns

        # create vertical layout for menu
        menu_layout = QVBoxLayout()
        menu_layout.addLayout(menu_grid)
        self.Layout = menu_layout


class ImageDisplay(QWidget):

    def __init__(self):
        super().__init__()

        self.createLayout()

    def createLayout(self):
        label = QLabel()
        # create a QPixmap object from the image file
        pixmap = QPixmap('/Users/pabloarb/Desktop/TIPE/python - mesure/Projet-analyse-video/sides/fonctionels/test.png')
        self.pixmap = pixmap
        # resize the pixmap
        scaled_pixmap = pixmap.scaled(360, 640)
        self.scaled_pixmap = scaled_pixmap
        # set the scaled pixmap as the image for the label
        label.setPixmap(scaled_pixmap)

        label.mousePressEvent = self.getPixelValue

        # create vertical layout for image
        image_layout = QVBoxLayout()
        image_layout.addWidget(label)
        self.Layout = image_layout

    def getPixelValue(self, event):
        # get the position of the mouse click relative to the label
        position = event.pos()

        # get the pixel value at the clicked position
        pixel_value = self.scaled_pixmap.toImage().pixel(position.x(), position.y())

        # print the RGB values of the pixel
        r, g, b, _ = QColor(pixel_value).getRgb()
        print(f'Pixel value: {r}, {g}, {b}')