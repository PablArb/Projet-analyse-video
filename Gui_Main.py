import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.QtGui import QPixmap, QColor
from Gui_ElementsConstructor import CalibButtonMenu, CalibImageDisplay
from MainConstructor import Video



class CalibDisplay(QWidget):

    def __init__(self, video:Video):
        super().__init__()

        settings = video.settings
        Windsize = settings.calibWindowSize
        self.pinnedMarkers = []

        self.pixmap = QPixmap('/Users/pabloarb/Desktop/TIPE/python - mesure/Projet-analyse-video/sides/fonctionels/test.png')
        self.scaled_pixmap = self.pixmap.scaled(Windsize[0], Windsize[1])

        self.buttonMenu = CalibButtonMenu()
        self.imageDisplay = CalibImageDisplay(self.scaled_pixmap, self.getPixelValue)

        self.initUI()

    def initUI(self):

        # create horizontal layout to display grid layout and image side by side
        main_layout = QGridLayout()
        main_layout.addLayout(self.buttonMenu.Layout, 0, 0)
        main_layout.addLayout(self.imageDisplay.Layout, 0, 1)

        self.buttonMenu.button1.clicked.connect(lambda : self.imageDisplay.update())
        
        # set the main layout for the widget
        self.setLayout(main_layout)

        # set window size and title
        self.setWindowTitle('test')

        self.show()
    
    def getPixelValue(self, event):

        # get the position of the mouse click relative to the label
        position = event.pos()

        xcoeff = self.pixmap.height()//self.scaled_pixmap.height()
        ycoeff = self.pixmap.width()//self.scaled_pixmap.width()

        x, y = position.x(), position.y()
        pixel_value = self.scaled_pixmap.toImage().pixel(x, y)
        r, g, b, _ = QColor(pixel_value).getRgb()

        self.pinnedMarkers.append(ProvMarker((x * xcoeff, y * ycoeff), (r, g, b)))
        # calcul de la nouvelle image avec le marker puis mise a jour sur le display


        # print the RGB values of the pixel
        print(f'Pixel value: x:{x}, y:{y}, r:{r}, g:{g}, b:{b}')


class ProvMarker(object):
    def __init__(self, coord, val):
        self.coordinates = coord
        self.RGBvalue = val



if __name__ == '__main__':
    from Main import video
    app = QApplication(sys.argv)
    CalibDisplay(video)
    sys.exit(app.exec_())

