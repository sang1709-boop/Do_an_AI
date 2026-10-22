"""Entrypoint app Windows. Chay: python dashboard.py"""
import sys
from PySide6.QtWidgets import QApplication
from app.gui.window import Dashboard


def main():
    application = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    return application.exec()


if __name__ == '__main__':
    sys.exit(main())
