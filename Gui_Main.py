from Modules import np
from Modules import cv2
from PyQt5.QtWidgets import QWidget, QGridLayout
from PyQt5.QtGui import QPixmap, QColor, QImage
from Gui_ElementsConstructor import CalibButtonMenu, CalibImageDisplay
from SettingsConstructor import Settings
from MainConstructor import Frame
from VisualIdicatorsConstructor import drawPinnedMarkersIndicators



class CalibDisplay(QWidget):

    def __init__(self, image: np.array, settings: Settings):
        super().__init__()

        self.size = settings.calibWindowSize

        self.pinnedMarkers = []
        self.r = settings.pinnedMarkersIndicatorRadius

        self.npimage = image
        self.pixmap = self.imageFormater(self.npimage, self.size)
        self.buttonMenu = CalibButtonMenu()
        self.imageDisplay = CalibImageDisplay(self.pixmap, self.getPixelValue)

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
    
    def imageFormater (self, image: np.array, size: tuple) -> QPixmap:
        qimage = self.ndarrayToQimage(image)
        pixmap = QPixmap.fromImage(QImage(qimage))
        scaled_pixmap = pixmap.scaled(size[0], size[1])
        return scaled_pixmap

    def ndarrayToQimage(self, image):
        rgbIm = np.copy(image[..., ::-1])
        height, width, channels = rgbIm.shape
        bytes_per_line = channels * width
        qimage = QImage(rgbIm.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage

    def getPixelValue(self, event):

        # get the position of the mouse click relative to the label
        position = event.pos()

        xcoeff = self.npimage.shape[0]//self.pixmap.height()
        ycoeff = self.npimage.shape[1]//self.pixmap.width()

        x, y = position.x(), position.y()
        pixel_value = self.pixmap.toImage().pixel(x, y)
        r, g, b, _ = QColor(pixel_value).getRgb()

        self.pinnedMarkers.append(ProvMarker((x * xcoeff, y * ycoeff), [r, g, b]))
        # calcul de la nouvelle image avec le marker puis mise a jour sur le display

        newIm = drawPinnedMarkersIndicators(self.npimage, self.pinnedMarkers, self.r)
        self.pixmap = self.imageFormater(newIm, self.size)
        self.imageDisplay.update(self.pixmap)

        # print the RGB values of the pixel
        print(f'Pixel value: x:{x}, y:{y}, r:{r}, g:{g}, b:{b}')



class ProvMarker(object):
    def __init__(self, coord, val):
        self.coordinates = coord
        self.RGBvalue = val
        self.BGRvalue = cv2.cvtColor(np.uint8([[self.RGBvalue]]), cv2.COLOR_RGB2BGR)[0][0]
        self.HSVvalue = cv2.cvtColor(np.uint8([[self.BGRvalue]]), cv2.COLOR_BGR2HSV)[0][0]


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     video = Video()
#     CalibDisplay(video)
#     app.exec_()
