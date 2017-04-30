import PyQt5 as qt
from PyQt5.QtWidgets import QWidget, QApplication
import sys

app = QApplication(sys.argv)
w = QWidget()
w.resize(1000, 650)
#w.move(300, 300)
w.setWindowTitle('Simple')
w.show()
sys.exit(app.exec_())

