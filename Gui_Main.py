import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout

from Gui_ElementsConstructor import ButtonMenu, ImageDisplay


class CalibDisplay(QWidget):

    def __init__(self):
        super().__init__()

        self.buttonMenu = ButtonMenu()
        self.image = ImageDisplay()

        self.initUI()


    def initUI(self):

        # create horizontal layout to display grid layout and image side by side
        main_layout = QGridLayout()
        main_layout.addLayout(self.buttonMenu.Layout, 0, 0)
        main_layout.addLayout(self.image.Layout, 0, 1)
        
        # set the main layout for the widget
        self.setLayout(main_layout)

        # set window size and title
        self.setWindowTitle('ma première appli')

        self.show()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    x = CalibDisplay()
    CalibDisplay.buttonMenu.button1.clicked.connect(lambda : CalibDisplay.image.changeTo())
    sys.exit(app.exec_())

