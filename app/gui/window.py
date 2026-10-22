"""DRIVEGUARD AI — Automotive Cockpit UI for the existing drowsiness project.

Replace app/gui/window.py with this file. The AI engine and database remain unchanged.

Giao diện được cập nhật theo thiết kế Figma (sidebar có nhãn, header có đồng hồ,
telemetry, trang Cài đặt hai cột, hộp thoại xác nhận/chi tiết). Toàn bộ logic
(CameraWorker, signal/slot, EventStore, CSV, camera USB/DroidCam) được giữ nguyên.
"""
import csv
import queue
import re
import threading
import traceback
import time
from collections import deque
from datetime import datetime

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QColor, QFont, QIcon, QImage, QLinearGradient, QPainter, QPainterPath,
    QPen, QPixmap, QPolygonF,
)
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from app.paths import MODELS, REPORTS
from app.event_log import EventStore, setup_logging

# ------------------------------------------------------------------ design tokens
BG = '#080D16'
NAV = '#0C1421'
PANEL = '#101B2A'
PANEL_ALT = '#0D1725'
BORDER = '#203046'
TEXT = '#EDF5FF'
MUTED = '#91A4BB'
DIM = '#6E8296'
ACCENT = '#38D9E6'
OK = '#31D6A0'
DANGER = '#FF5263'
AMBER = '#FFBD5A'

FONT_UI = "'Inter', 'Segoe UI', 'Helvetica Neue', sans-serif"
FONT_MONO = "'JetBrains Mono', 'Cascadia Mono', 'Consolas', 'DejaVu Sans Mono', monospace"

CSS = f"""
* {{font-family: {FONT_UI}; font-size: 12px; color: {TEXT};}}
QMainWindow, QWidget#Root, QWidget#Page {{background: {BG};}}
QLabel {{background: transparent; border: none;}}
QToolTip {{background: #142132; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 8px;}}

/* ---- sidebar ---- */
QFrame#Rail {{background: {NAV}; border: none; border-right: 1px solid {BORDER};}}
QLabel#Brand {{font-size: 15px; font-weight: 800;}}
QLabel#BrandAccent {{font-size: 15px; font-weight: 800; color: {ACCENT};}}
QLabel#RailCaption {{font-size: 9px; font-weight: 700; color: {DIM};}}
QPushButton#Nav {{background: transparent; border: 1px solid transparent; border-radius: 10px;
  color: {MUTED}; font-size: 12px; font-weight: 600; text-align: left; padding: 11px 12px;}}
QPushButton#Nav:hover {{background: #14263A; color: white;}}
QPushButton#Nav[active="true"] {{background: #11303F; border: 1px solid #276175; color: {ACCENT};}}
QFrame#SideStatus {{background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 10px;}}
QLabel#SideKey {{font-size: 9px; font-weight: 700; color: {DIM};}}
QLabel#SideVal {{font-family: {FONT_MONO}; font-size: 10px; font-weight: 700;}}

/* ---- surfaces ---- */
QFrame#Panel {{background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px;}}
QFrame#VideoCard {{background: #0B1521; border: 1px solid #254054; border-radius: 15px;}}
QFrame#StatCard {{background: {PANEL}; border: 1px solid {BORDER}; border-radius: 13px;}}
QFrame#Strip {{background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 12px;}}
QFrame#Divider {{background: {BORDER}; border: none; max-height: 1px; min-height: 1px;}}
QFrame#InfoStrip {{background: #0F2230; border: 1px solid #1F4C5C; border-radius: 10px;}}
QFrame#Empty {{background: {PANEL}; border: 1px dashed #2D455D; border-radius: 14px;}}
QDialog#Dialog {{background: {PANEL}; border: 1px solid #2D455D; border-radius: 14px;}}

/* ---- typography ---- */
QLabel#Heading {{font-size: 22px; font-weight: 800;}}
QLabel#Title {{font-size: 14px; font-weight: 800;}}
QLabel#Subtitle {{color: {MUTED}; font-size: 12px;}}
QLabel#Caption {{font-size: 10px; font-weight: 700; color: {MUTED};}}
QLabel#Tag {{font-size: 9px; font-weight: 700; color: {DIM};}}
QLabel#Hint {{font-size: 11px; color: {DIM};}}
QLabel#Value {{font-family: {FONT_MONO}; font-size: 23px; font-weight: 800;}}
QLabel#ValueText {{font-size: 19px; font-weight: 800;}}
QLabel#Metric {{font-family: {FONT_MONO}; font-size: 15px; font-weight: 700;}}
QLabel#MetricText {{font-size: 13px; font-weight: 700;}}
QLabel#StripValue {{font-family: {FONT_MONO}; font-size: 16px; font-weight: 700;}}
QLabel#Clock {{font-family: {FONT_MONO}; font-size: 17px; font-weight: 700;}}
QLabel#ClockDate {{font-size: 10px; color: {DIM};}}
QLabel#Preview {{background: #050B12; border-radius: 10px; color: {MUTED}; font-size: 13px;}}
QLabel#Counter {{font-family: {FONT_MONO}; font-size: 28px; font-weight: 800; color: {ACCENT};}}

/* ---- chips & banner ---- */
QLabel#Chip {{color: {MUTED}; background: #0F1A28; border: 1px solid {BORDER};
  border-radius: 10px; padding: 5px 11px; font-size: 11px; font-weight: 700;}}
QLabel#Chip[tone="live"] {{color: {ACCENT}; background: #102C36; border: 1px solid #275762;}}
QLabel#Chip[tone="ok"] {{color: {OK}; background: #0E2B26; border: 1px solid #1F6B58;}}
QLabel#Chip[tone="warn"] {{color: {AMBER}; background: #33280F; border: 1px solid #7A5C1F;}}
QLabel#Chip[tone="alert"] {{color: #FFD9DD; background: #471923; border: 1px solid {DANGER};}}
QLabel#Banner {{background: #3A1620; border: 1px solid {DANGER}; border-left: 4px solid {DANGER};
  border-radius: 10px; color: #FFD9DD; padding: 10px 16px; font-size: 14px; font-weight: 800;}}

/* ---- buttons ---- */
QPushButton {{background: #192C40; border: 1px solid #2A4058; border-radius: 9px;
  padding: 9px 14px; font-size: 12px; font-weight: 700;}}
QPushButton:hover {{background: #244159;}}
QPushButton:disabled {{color: #687D91; background: #12202F; border-color: #233448;}}
QPushButton#Primary {{background: {ACCENT}; color: #06202A; border: none;}}
QPushButton#Primary:hover {{background: #73E8EF;}}
QPushButton#Primary:disabled {{background: #1D4350; color: #7999A2;}}
QPushButton#Danger {{background: #2A141B; border: 1px solid #A0434D; color: #FF9AA4;}}
QPushButton#Danger:hover {{background: #4A1F28; color: #FFE4E6;}}
QPushButton#Danger:disabled {{background: #12202F; border: 1px solid #233448; color: #687D91;}}
QPushButton#Ghost {{background: transparent; border: 1px solid #2D455D; color: {TEXT};}}
QPushButton#Ghost:hover {{background: #14263A;}}
QPushButton#Option {{background: #0D1827; border: 1px solid #2D455D; border-radius: 11px;
  text-align: left; padding: 13px 15px; font-weight: 600;}}
QPushButton#Option:hover {{background: #112033;}}
QPushButton#Option:checked {{background: #123040; border: 1px solid {ACCENT};}}

/* ---- inputs & progress ---- */
QProgressBar {{background: #26394A; border: none; border-radius: 3px;}}
QProgressBar::chunk {{background: {ACCENT}; border-radius: 3px;}}
QComboBox, QLineEdit {{background: #0D1827; border: 1px solid #2D455D;
  border-radius: 8px; padding: 9px 11px; selection-background-color: #1E6974;}}
QComboBox:disabled, QLineEdit:disabled {{color: #5C7186; background: #0A1320; border-color: #1B2B3D;}}
QLineEdit:focus, QComboBox:focus {{border: 1px solid {ACCENT};}}
QComboBox QAbstractItemView {{background: #142132; selection-background-color: #24516A;}}

/* ---- table ---- */
QTableWidget {{background: {PANEL}; alternate-background-color: {PANEL_ALT};
  gridline-color: {BORDER}; border: none; border-radius: 10px;}}
QHeaderView::section {{background: #142333; color: {MUTED}; border: none;
  border-bottom: 1px solid {BORDER}; padding: 9px 10px; font-size: 10px; font-weight: 700;}}
QTableWidget::item {{padding: 0 10px; border-bottom: 1px solid #16263A;}}
QTableWidget::item:selected {{background: #123A4A; border-top: 1px solid {ACCENT};
  border-bottom: 1px solid {ACCENT};}}
QScrollBar:vertical {{background: transparent; width: 9px; margin: 2px;}}
QScrollBar::handle:vertical {{background: #2A4058; border-radius: 4px; min-height: 28px;}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{height: 0;}}
"""


# ------------------------------------------------------------------ vector icons
def make_icon(kind, color, size=18):
    """Draw a small line icon (no external assets needed, crisp on HiDPI)."""
    scale = 2
    pix = QPixmap(size * scale, size * scale)
    pix.setDevicePixelRatio(scale)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    s = float(size)
    if kind == 'grid':
        g = s * 0.12
        c = s * 0.44
        for x in (g, s - g - c):
            for y in (g, s - g - c):
                p.drawRoundedRect(QRectF(x, y, c, c), 2, 2)
    elif kind == 'camera':
        p.drawRoundedRect(QRectF(s * 0.08, s * 0.26, s * 0.84, s * 0.58), 3, 3)
        p.drawEllipse(QPointF(s * 0.5, s * 0.55), s * 0.17, s * 0.17)
        p.drawLine(QPointF(s * 0.32, s * 0.26), QPointF(s * 0.4, s * 0.14))
        p.drawLine(QPointF(s * 0.4, s * 0.14), QPointF(s * 0.6, s * 0.14))
        p.drawLine(QPointF(s * 0.6, s * 0.14), QPointF(s * 0.68, s * 0.26))
    elif kind == 'history':
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.4, s * 0.4)
        p.drawLine(QPointF(s * 0.5, s * 0.5), QPointF(s * 0.5, s * 0.26))
        p.drawLine(QPointF(s * 0.5, s * 0.5), QPointF(s * 0.68, s * 0.6))
    elif kind == 'gear':
        import math
        cx = cy = s / 2
        for k in range(8):
            a = math.radians(k * 45)
            p.drawLine(QPointF(cx + math.cos(a) * s * 0.30, cy + math.sin(a) * s * 0.30),
                       QPointF(cx + math.cos(a) * s * 0.44, cy + math.sin(a) * s * 0.44))
        p.drawEllipse(QPointF(cx, cy), s * 0.27, s * 0.27)
        p.drawEllipse(QPointF(cx, cy), s * 0.10, s * 0.10)
    elif kind == 'shield':
        path = QPainterPath()
        path.moveTo(s * 0.5, s * 0.08)
        path.lineTo(s * 0.88, s * 0.22)
        path.lineTo(s * 0.88, s * 0.50)
        path.cubicTo(s * 0.88, s * 0.74, s * 0.68, s * 0.88, s * 0.5, s * 0.94)
        path.cubicTo(s * 0.32, s * 0.88, s * 0.12, s * 0.74, s * 0.12, s * 0.50)
        path.lineTo(s * 0.12, s * 0.22)
        path.closeSubpath()
        p.drawPath(path)
        p.drawEllipse(QPointF(s * 0.5, s * 0.48), s * 0.14, s * 0.14)
    p.end()
    return QIcon(pix)


class NavButton(QPushButton):
    """Sidebar item: line icon + label, restyled when active."""
    def __init__(self, kind, text, parent=None):
        super().__init__('   ' + text, parent)
        self.kind = kind
        self.setObjectName('Nav')
        self.setIconSize(QSize(18, 18))
        self.setFixedHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_active(False)

    def set_active(self, active):
        self.setProperty('active', active)
        self.setIcon(make_icon(self.kind, ACCENT if active else MUTED))
        self.style().unpolish(self)
        self.style().polish(self)


class OptionCard(QPushButton):
    """Selectable card (Figma settings): bold title + muted description."""
    def __init__(self, title, hint, parent=None):
        super().__init__(parent)
        self.setObjectName('Option')
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setMinimumHeight(66)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        box = QVBoxLayout(self)
        box.setContentsMargins(16, 11, 16, 11)
        box.setSpacing(2)
        for text, name in ((title, 'Title'), (hint, 'Hint')):
            item = label(text, name)
            item.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            box.addWidget(item)


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    status_ready = Signal(object)
    failure = Signal(str)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source
        self.stop_flag = threading.Event()
        self.commands = queue.Queue()

    def command(self, command):
        self.commands.put(command)

    def stop(self):
        self.stop_flag.set()

    def run(self):
        try:
            from app.realtime import main

            def show_frame(frame):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(image)

            main(frame_callback=show_frame, status_callback=self.status_ready.emit,
                 stop_event=self.stop_flag, command_queue=self.commands,
                 camera_source=self.source)
        except Exception:
            self.failure.emit(traceback.format_exc())


# ------------------------------------------------------------------ small helpers
def panel(name='Panel'):
    result = QFrame()
    result.setObjectName(name)
    return result


def label(value, name=None, wrap=False):
    result = QLabel(value)
    if name:
        result.setObjectName(name)
    result.setWordWrap(wrap)
    if name in ('Caption', 'Tag', 'RailCaption', 'SideKey'):
        font = result.font()
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.7)
        result.setFont(font)
    return result


def divider():
    line = QFrame()
    line.setObjectName('Divider')
    line.setFrameShape(QFrame.Shape.HLine)
    return line


def set_tone(widget, tone):
    """Switch a QSS [tone] variant and repolish only when it changed."""
    if widget.property('tone') != tone:
        widget.setProperty('tone', tone)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class TrendGraph(QWidget):
    """Simple realtime probability plot. No Matplotlib dependency."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = deque(maxlen=96)
        self.threshold = None
        self.setMinimumHeight(80)

    def clear(self):
        self.data.clear()
        self.threshold = None
        self.update()

    def append(self, value, threshold=None):
        self.data.append(None if value is None else float(value))
        self.threshold = threshold
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(40, 8, max(1, self.width() - 52), max(1, self.height() - 18))
        label_font = painter.font()
        label_font.setPixelSize(9)
        painter.setFont(label_font)
        for step in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = rect.bottom() - rect.height() * step
            painter.setPen(QPen(QColor('#284052' if 0 < step < 1 else '#1D3042'), 1))
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            painter.setPen(QColor(DIM))
            painter.drawText(QRectF(0, y - 7, 34, 14),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f'{int(step * 100)}%')
        if self.threshold is not None:
            y = rect.bottom() - rect.height() * min(1, max(0, float(self.threshold)))
            pen = QPen(QColor(AMBER), 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
        values = list(self.data)
        if len(values) < 2:
            painter.setPen(QColor('#6E8296'))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, 'Biểu đồ bắt đầu khi camera hoạt động')
            return
        segments, current = [], []
        for i, v in enumerate(values):
            if v is None:
                if current:
                    segments.append(current)
                current = []
                continue
            x = rect.left() + i * rect.width() / (len(values) - 1)
            y = rect.bottom() - min(1, max(0, v)) * rect.height()
            current.append(QPointF(x, y))
        if current:
            segments.append(current)
        gradient = QLinearGradient(0, rect.top(), 0, rect.bottom())
        top = QColor(ACCENT)
        top.setAlpha(85)
        bottom = QColor(ACCENT)
        bottom.setAlpha(0)
        gradient.setColorAt(0, top)
        gradient.setColorAt(1, bottom)
        for points in segments:
            if len(points) < 2:
                continue
            polygon = QPolygonF(points + [QPointF(points[-1].x(), rect.bottom()),
                                          QPointF(points[0].x(), rect.bottom())])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawPolygon(polygon)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(ACCENT), 2.0))
            path = QPainterPath(points[0])
            for point in points[1:]:
                path.lineTo(point)
            painter.drawPath(path)


class ConfirmStopDialog(QDialog):
    """Figma: 'Xác nhận dừng camera'."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('Dialog')
        self.setWindowTitle('Xác nhận dừng camera')
        self.setModal(True)
        self.setMinimumWidth(420)
        box = QVBoxLayout(self)
        box.setContentsMargins(24, 22, 24, 20)
        box.setSpacing(10)
        box.addWidget(label('XÁC NHẬN DỪNG CAMERA', 'Caption'))
        box.addWidget(label('Dừng camera?', 'Heading'))
        box.addWidget(label(
            'Phiên giám sát hiện tại sẽ kết thúc. Các cảnh báo đã ghi vẫn được lưu '
            'trong lịch sử và báo cáo CSV của phiên.', 'Subtitle', True))
        box.addSpacing(8)
        row = QHBoxLayout()
        row.addStretch()
        keep = QPushButton('Tiếp tục giám sát')
        keep.setObjectName('Ghost')
        stop = QPushButton('■  DỪNG CAMERA')
        stop.setObjectName('Danger')
        keep.clicked.connect(self.reject)
        stop.clicked.connect(self.accept)
        keep.setDefault(True)
        row.addWidget(keep)
        row.addWidget(stop)
        box.addLayout(row)


class AlertDetailDialog(QDialog):
    """Figma: 'Chi tiết cảnh báo' — built only from the selected history row."""
    def __init__(self, headers, values, parent=None):
        super().__init__(parent)
        self.setObjectName('Dialog')
        self.setWindowTitle('Chi tiết cảnh báo')
        self.setModal(True)
        self.setMinimumWidth(430)
        box = QVBoxLayout(self)
        box.setContentsMargins(24, 22, 24, 20)
        box.setSpacing(8)
        kind = values[2]
        color = DANGER if 'Buồn ngủ' in kind else AMBER if 'Mắt mở' in kind else TEXT
        box.addWidget(label('CHI TIẾT CẢNH BÁO', 'Caption'))
        head = label('⚠  ' + kind, 'Title')
        head.setStyleSheet(f'font-size:17px; font-weight:800; color:{color};')
        box.addWidget(head)
        box.addSpacing(6)
        for i, (name, value) in enumerate(zip(headers, values)):
            if i == 2:
                continue
            box.addWidget(divider())
            row = QHBoxLayout()
            row.addWidget(label(name.upper(), 'Caption'))
            row.addStretch()
            row.addWidget(label(value, 'Metric'))
            box.addLayout(row)
        box.addSpacing(10)
        close = QPushButton('Đóng')
        close.setObjectName('Ghost')
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close)
        box.addLayout(row)


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DRIVEGUARD AI  |  Giám sát tài xế thông minh')
        self.resize(1440, 875)
        self.setMinimumSize(1100, 730)
        self.setStyleSheet(CSS)
        self.worker = None
        self.last_image = None
        self.alert_active = False
        self.alert_count = 0
        self.events = []
        self.session_id = None
        self.active_alert = None
        self.last_status = None
        self._last_plot = 0.0
        self._clocks = []
        self.store = EventStore()
        self.logger = setup_logging()
        self._build()
        self.check_models()
        # UI-only refresh: clock, sidebar status, chip colours, mirrored metric strip.
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(400)
        self._ui_timer.timeout.connect(self._tick)
        self._ui_timer.start()
        self._tick()

    # ------------------------------------------------------------ shell
    def _build(self):
        root = QWidget()
        root.setObjectName('Root')
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        rail = panel('Rail')
        rail.setFixedWidth(204)
        sb = QVBoxLayout(rail)
        sb.setContentsMargins(14, 22, 14, 16)
        sb.setSpacing(6)

        brand = QHBoxLayout()
        brand.setSpacing(9)
        mark = QLabel()
        mark.setPixmap(make_icon('shield', ACCENT, 26).pixmap(26, 26))
        brand.addWidget(mark)
        brand.addWidget(label('DriveGuard', 'Brand'))
        brand.addWidget(label('AI', 'BrandAccent'))
        brand.addStretch()
        sb.addLayout(brand)
        sb.addSpacing(26)
        sb.addWidget(label('ĐIỀU HƯỚNG', 'RailCaption'))
        sb.addSpacing(2)

        self.nav = []
        for i, (kind, name) in enumerate((('grid', 'Tổng quan'), ('camera', 'Camera trực tiếp'),
                                          ('history', 'Lịch sử cảnh báo'), ('gear', 'Cài đặt'))):
            button = NavButton(kind, name)
            button.setToolTip(name)
            button.clicked.connect(lambda checked=False, index=i: self.switch_page(index))
            sb.addWidget(button)
            self.nav.append(button)
        sb.addStretch()

        status = panel('SideStatus')
        st = QVBoxLayout(status)
        st.setContentsMargins(12, 10, 12, 10)
        st.setSpacing(5)
        st.addWidget(label('TRẠNG THÁI HỆ THỐNG', 'RailCaption'))
        self.side_vals = {}
        for key, caption in (('camera', 'CAMERA'), ('model', 'MÔ HÌNH AI'), ('session', 'PHIÊN')):
            row = QHBoxLayout()
            row.addWidget(label(caption, 'SideKey'))
            row.addStretch()
            value = label('--', 'SideVal')
            row.addWidget(value)
            st.addLayout(row)
            self.side_vals[key] = value
        sb.addWidget(status)
        version = label('v1.0', 'Caption')
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sb.addSpacing(4)
        sb.addWidget(version)
        outer.addWidget(rail)

        self.pages = QStackedWidget()
        outer.addWidget(self.pages, 1)
        for build in (self.overview_page, self.camera_page,
                      self.history_page, self.settings_page):
            page = QWidget()
            page.setObjectName('Page')
            layout = QVBoxLayout(page)
            layout.setContentsMargins(26, 20, 26, 18)
            layout.setSpacing(12)
            build(layout)
            self.pages.addWidget(page)
        self.switch_page(0)

    def switch_page(self, index):
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.nav):
            button.set_active(i == index)
        self.update_video()
        if index == 2:
            self.refresh_history()

    def title(self, layout, headline, subtitle, show_chip=False):
        line = QHBoxLayout()
        line.setSpacing(12)
        box = QVBoxLayout()
        box.setSpacing(3)
        box.addWidget(label(headline, 'Heading'))
        box.addWidget(label(subtitle, 'Subtitle'))
        line.addLayout(box, 1)
        if show_chip:
            self.run_chip = label('●  STANDBY', 'Chip')
            self.run_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            line.addWidget(self.run_chip, 0, Qt.AlignmentFlag.AlignVCenter)
        clock = QVBoxLayout()
        clock.setSpacing(0)
        time_label = label('--:--:--', 'Clock')
        date_label = label('', 'ClockDate')
        time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        date_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        clock.addWidget(time_label)
        clock.addWidget(date_label)
        line.addSpacing(6)
        line.addLayout(clock)
        self._clocks.append((time_label, date_label))
        layout.addLayout(line)

    # ------------------------------------------------------------ Tổng quan
    def overview_page(self, layout):
        self.title(layout, 'DRIVEGUARD  /  LIVE MONITOR',
                   'Giám sát hành vi tài xế • AI Computer Vision', True)
        self.banner = label('⚠  CẢNH BÁO BUỒN NGỦ — HÃY TẬP TRUNG VÀ DỪNG XE AN TOÀN', 'Banner')
        self.banner.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.banner.hide()
        layout.addWidget(self.banner)

        stats = QGridLayout()
        stats.setSpacing(11)
        self.fields = {}
        stat_descriptions = (
            ('state', 'TÌNH TRẠNG TÀI XẾ', 'Chưa khởi động', 'AI'),
            ('prob', 'NGUY CƠ BUỒN NGỦ', '--', 'REALTIME'),
            ('fps', 'TỐC ĐỘ CAMERA', '--', 'CAMERA'),
            ('alerts', 'CẢNH BÁO PHIÊN NÀY', '0', 'PHIÊN'),
        )
        for i, (key, caption, default, tag) in enumerate(stat_descriptions):
            card = panel('StatCard')
            v = QVBoxLayout(card)
            v.setContentsMargins(17, 12, 17, 12)
            v.setSpacing(7)
            head = QHBoxLayout()
            head.addWidget(label(caption, 'Caption'))
            head.addStretch()
            head.addWidget(label(tag, 'Tag'))
            v.addLayout(head)
            value = label(default, 'ValueText' if key == 'state' else 'Value')
            if key == 'state':
                value.setStyleSheet(f'font-size:19px; font-weight:800; color:{TEXT};')
            v.addWidget(value)
            self.fields[key] = value
            if key == 'prob':
                self.prob_bar = QProgressBar()
                self.prob_bar.setRange(0, 100)
                self.prob_bar.setValue(0)
                self.prob_bar.setFixedHeight(6)
                self.prob_bar.setTextVisible(False)
                v.addWidget(self.prob_bar)
            else:
                v.addSpacing(6)
            stats.addWidget(card, 0, i)
            stats.setColumnStretch(i, 1)
        layout.addLayout(stats)

        middle = QHBoxLayout()
        middle.setSpacing(12)
        camera_card = panel('VideoCard')
        v = QVBoxLayout(camera_card)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        header = QHBoxLayout()
        header.addWidget(label('◉  CAMERA / LIVE FEED', 'Title'))
        header.addStretch()
        self.camera_chip = label('CAM 01  ·  OFFLINE', 'Chip')
        header.addWidget(self.camera_chip)
        v.addLayout(header)
        self.preview = label(self._placeholder_html(
            'CAMERA CHƯA KẾT NỐI', 'Bấm “Bắt đầu giám sát” để kích hoạt'), 'Preview', True)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview.setMinimumHeight(210)
        v.addWidget(self.preview, 1)
        footer = QHBoxLayout()
        footer.addWidget(label('MEDIA PIPE  •  FACE & POSE LANDMARKS', 'Caption'))
        footer.addStretch()
        footer.addWidget(label('LIVE ANALYSIS', 'Caption'))
        v.addLayout(footer)
        middle.addWidget(camera_card, 7)

        side = panel()
        side.setMinimumWidth(300)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(16, 14, 16, 12)
        sv.setSpacing(5)
        sv.addWidget(label('AI TELEMETRY', 'Title'))
        sv.addWidget(label('Phân tích khuôn mặt và trạng thái', 'Subtitle'))
        sv.addSpacing(4)
        self.metrics = {}
        rows = (
            ('ear', 'EAR  /  ĐỘ MỞ MẮT', 'Eye Aspect Ratio'),
            ('mar', 'MAR  /  ĐỘ MỞ MIỆNG', 'Mouth Aspect Ratio'),
            ('long', 'MẮT MỞ BẤT THƯỜNG', 'Xác suất từ mô hình phụ'),
            ('face', 'NHẬN DIỆN KHUÔN MẶT', 'MediaPipe Face Landmarker'),
            ('lighting', 'CHẾ ĐỘ ÁNH SÁNG', 'Ngày / đêm'),
            ('posture', 'TƯ THẾ TÀI XẾ', 'MediaPipe Pose'),
        )
        for n, (key, caption, hint) in enumerate(rows):
            sv.addWidget(divider())
            line = QHBoxLayout()
            names = QVBoxLayout()
            names.setSpacing(0)
            names.addWidget(label(caption, 'Caption'))
            names.addWidget(label(hint, 'Hint'))
            line.addLayout(names)
            line.addStretch()
            numeric = key in ('ear', 'mar', 'long')
            val = label('--', 'Metric' if numeric else 'MetricText', numeric)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if numeric:
                val.setMaximumWidth(185)
            line.addWidget(val)
            self.metrics[key] = val
            sv.addLayout(line)
            if key in ('ear', 'mar', 'long'):
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(0)
                bar.setFixedHeight(4)
                bar.setTextVisible(False)
                setattr(self, f'{key}_bar', bar)
                sv.addWidget(bar)
        sv.addStretch()
        middle.addWidget(side, 3)
        layout.addLayout(middle, 1)

        trend_card = panel()
        trend = QVBoxLayout(trend_card)
        trend.setContentsMargins(15, 9, 15, 7)
        trend.setSpacing(2)
        chart_title = QHBoxLayout()
        chart_title.addWidget(label('XÁC SUẤT BUỒN NGỦ  /  REALTIME TREND', 'Caption'))
        chart_title.addStretch()
        legend_line = label('━ xác suất', 'Caption')
        legend_line.setStyleSheet(f'color:{ACCENT};')
        legend_thr = label('╌ ngưỡng cảnh báo', 'Caption')
        legend_thr.setStyleSheet(f'color:{AMBER};')
        chart_title.addWidget(legend_line)
        chart_title.addSpacing(10)
        chart_title.addWidget(legend_thr)
        trend.addLayout(chart_title)
        self.trend_graph = TrendGraph()
        trend.addWidget(self.trend_graph)
        layout.addWidget(trend_card)

        actions = QHBoxLayout()
        actions.setSpacing(9)
        self.start_btn = QPushButton('▶   BẮT ĐẦU GIÁM SÁT')
        self.start_btn.setObjectName('Primary')
        self.stop_btn = QPushButton('■   DỪNG CAMERA')
        self.stop_btn.setObjectName('Danger')
        self.stop_btn.setEnabled(False)
        for button in (self.start_btn, self.stop_btn):
            button.setMinimumHeight(43)
            button.setMinimumWidth(190)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_camera)
        # Dừng camera đi qua hộp thoại xác nhận (Figma), sau đó gọi đúng stop_camera() cũ.
        self.stop_btn.clicked.connect(self.confirm_stop)
        actions.addWidget(self.start_btn)
        actions.addWidget(self.stop_btn)
        actions.addStretch()
        self.model_state = label('', 'Subtitle', True)
        actions.addWidget(self.model_state)
        layout.addLayout(actions)

    @staticmethod
    def _placeholder_html(headline, hint):
        return (f'<div style="text-align:center;">'
                f'<div style="font-size:34px; color:{DIM};">▣</div><br>'
                f'<div style="font-size:13px; font-weight:800; color:{MUTED};">{headline}</div>'
                f'<div style="font-size:11px; color:{DIM};">{hint}</div></div>')

    # ------------------------------------------------------------ Camera trực tiếp
    def camera_page(self, layout):
        self.title(layout, 'CAMERA  /  FULL SCREEN VIEW',
                   'Khung hình gốc từ camera • Giữ hướng hình phù hợp với dữ liệu huấn luyện')
        card = panel('VideoCard')
        box = QVBoxLayout(card)
        box.setContentsMargins(12, 12, 12, 12)
        box.setSpacing(8)
        camera_header = QHBoxLayout()
        camera_header.addWidget(label('◉  LIVE CAMERA', 'Title'))
        camera_header.addStretch()
        advanced_button = QPushButton('⚙  Tùy chọn')
        advanced_button.setObjectName('Ghost')
        advanced_button.setCheckable(True)
        camera_header.addWidget(advanced_button)
        box.addLayout(camera_header)
        self.full_preview = label(self._placeholder_html('CHƯA CÓ KHUNG HÌNH',
                                                         'Khởi động giám sát ở trang Tổng quan'),
                                  'Preview', True)
        self.full_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.full_preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        box.addWidget(self.full_preview, 1)
        layout.addWidget(card, 1)

        # Metric strip (mirrors live values; refreshed by _tick, read-only).
        strip = panel('Strip')
        grid = QHBoxLayout(strip)
        grid.setContentsMargins(18, 10, 18, 10)
        grid.setSpacing(10)
        self.strip = {}
        for key, caption in (('fps', 'TỐC ĐỘ'), ('ear', 'EAR'), ('mar', 'MAR'),
                             ('state', 'TRẠNG THÁI'), ('prob', 'NGUY CƠ BUỒN NGỦ')):
            cell = QVBoxLayout()
            cell.setSpacing(2)
            cell.addWidget(label(caption, 'Caption'))
            value = label('--', 'StripValue')
            cell.addWidget(value)
            self.strip[key] = value
            grid.addLayout(cell, 1)
        layout.addWidget(strip)

        buttons = QHBoxLayout()
        buttons.setSpacing(7)
        for name, cmd in (
            ('↻  Reset', 'r'), ('−  Ngưỡng', '['), ('+  Ngưỡng', ']'),
            ('☼  Ngày/đêm', 'n'), ('◎  Học tư thế', 'c'), ('◈  Lọc tư thế', 'v')
        ):
            button = QPushButton(name)
            button.setObjectName('Ghost')
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, c=cmd: self.send(c))
            buttons.addWidget(button)
        buttons.addStretch()
        controls = QWidget()
        controls.setLayout(buttons)
        layout.addWidget(controls)
        self.detail_status = label('Sẵn sàng', 'Subtitle', True)
        layout.addWidget(self.detail_status)

        # Mặc định trang Camera chỉ hiện video. Công cụ vẫn có thể mở lại,
        # toàn bộ signal/slot và lệnh điều khiển AI vẫn được giữ nguyên.
        strip.hide()
        controls.hide()
        self.detail_status.hide()

        def toggle_advanced(visible):
            strip.setVisible(visible)
            controls.setVisible(visible)
            self.detail_status.setVisible(visible)
            advanced_button.setText('Ẩn tùy chọn' if visible else '⚙  Tùy chọn')

        advanced_button.toggled.connect(toggle_advanced)

    # ------------------------------------------------------------ Lịch sử cảnh báo
    def history_page(self, layout):
        self.title(layout, 'LỊCH SỬ CẢNH BÁO',
                   'Lưu các sự kiện giám sát vào SQLite và xuất báo cáo CSV')
        top = QHBoxLayout()
        top.setSpacing(12)
        self.history_cards = {}
        for key, caption in (('total', 'TỔNG CẢNH BÁO'), ('sessions', 'PHIÊN CÓ CẢNH BÁO')):
            card = panel('StatCard')
            cv = QVBoxLayout(card)
            cv.setContentsMargins(18, 10, 22, 10)
            cv.setSpacing(0)
            number = label('--', 'Counter')
            cv.addWidget(number)
            cv.addWidget(label(caption, 'Caption'))
            self.history_cards[key] = number
            top.addWidget(card)
        top.addStretch()
        refresh = QPushButton('↻  Làm mới')
        refresh.setObjectName('Ghost')
        refresh.clicked.connect(self.refresh_history)
        export = QPushButton('↓  Xuất CSV')
        export.setObjectName('Primary')
        export.clicked.connect(self.export_history)
        for button in (refresh, export):
            button.setMinimumHeight(38)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        top.addWidget(refresh, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(export, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(top)

        info = panel('InfoStrip')
        il = QHBoxLayout(info)
        il.setContentsMargins(14, 8, 14, 8)
        il.addWidget(label('ⓘ  Nhấp đúp vào một dòng để xem chi tiết đợt cảnh báo. '
                           'Dữ liệu được lưu bằng SQLite; CSV xuất theo yêu cầu.', 'Subtitle'))
        layout.addWidget(info)

        table_card = panel()
        tc = QVBoxLayout(table_card)
        tc.setContentsMargins(0, 12, 0, 0)
        tc.setSpacing(6)
        head = QHBoxLayout()
        head.setContentsMargins(16, 0, 16, 0)
        head.addWidget(label('SỰ KIỆN ĐÃ GHI NHẬN', 'Caption'))
        head.addStretch()
        tc.addLayout(head)

        self.history_table = QTableWidget(0, 8)
        self.history_table.setHorizontalHeaderLabels([
            'Bắt đầu', 'Kết thúc', 'Loại', 'Thời lượng (s)',
            'Xác suất cao nhất', 'EAR', 'MAR', 'Phiên'
        ])
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setShowGrid(False)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.verticalHeader().setDefaultSectionSize(36)
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for column in range(3, 7):  # cột số liệu căn phải, khớp với ô dữ liệu
            self.history_table.horizontalHeaderItem(column).setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        for column, width in enumerate((170, 170, 160, 120, 150, 90, 90)):
            self.history_table.setColumnWidth(column, width)
        self.history_table.cellDoubleClicked.connect(self.show_alert_detail)

        # Stack: 0 = bảng, 1 = trạng thái trống (Figma: "Lịch sử trống")
        self.history_stack = QStackedWidget()
        self.history_stack.addWidget(self.history_table)
        empty = panel('Empty')
        ev = QVBoxLayout(empty)
        ev.setAlignment(Qt.AlignmentFlag.AlignCenter)
        glyph = QLabel()
        glyph.setPixmap(make_icon('history', DIM, 30).pixmap(30, 30))
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ev.addWidget(glyph)
        empty_title = label('Chưa có lịch sử cảnh báo', 'Title')
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ev.addWidget(empty_title)
        empty_hint = label('Các sự kiện sẽ xuất hiện khi hệ thống phát hiện buồn ngủ trong phiên giám sát.',
                           'Subtitle', True)
        empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ev.addWidget(empty_hint)
        self.history_stack.addWidget(empty)
        tc.addWidget(self.history_stack, 1)

        foot = QHBoxLayout()
        foot.setContentsMargins(16, 0, 16, 10)
        self.history_summary = label('Chưa có dữ liệu', 'Hint')
        foot.addWidget(self.history_summary)
        foot.addStretch()
        tc.addLayout(foot)
        layout.addWidget(table_card, 1)

    def refresh_history(self):
        try:
            rows = self.store.list_alerts(limit=500)
            self.history_table.setRowCount(len(rows))
            for i, r in enumerate(rows):
                items = (
                    r['started_at'], r['ended_at'] or 'Đang cảnh báo', r['kind'],
                    f"{r['duration_seconds']:.1f}" if r['duration_seconds'] is not None else '--',
                    f"{100 * r['max_probability']:.1f}%" if r['max_probability'] is not None else '--',
                    f"{r['ear']:.3f}" if r['ear'] is not None else '--',
                    f"{r['mar']:.3f}" if r['mar'] is not None else '--', str(r['session_id'])
                )
                for j, value in enumerate(items):
                    self.history_table.setItem(i, j, QTableWidgetItem(str(value)))
            total = self.store.count_alerts()
            self.history_summary.setText(
                f'Tổng cộng {total} đợt cảnh báo  •  Hiển thị tối đa 500 đợt gần nhất'
            )
            self.decorate_history(rows, total)  # chỉ định dạng hiển thị
        except Exception:
            self.logger.exception('Không tải được lịch sử')
            QMessageBox.warning(self, 'Lỗi nhật ký', 'Không tải được lịch sử. Xem app.log.')

    def decorate_history(self, rows, total):
        """Presentation only: colour the alert type, mono numerics, counters, empty state."""
        try:
            mono = QFont('JetBrains Mono')
            mono.setStyleHint(QFont.StyleHint.Monospace)
            mono.setPixelSize(11)
            for i in range(self.history_table.rowCount()):
                for j in range(self.history_table.columnCount()):
                    item = self.history_table.item(i, j)
                    if item is None:
                        continue
                    if j == 2:
                        text = item.text()
                        item.setForeground(QColor(DANGER if 'Buồn ngủ' in text else
                                                  AMBER if 'Mắt mở' in text else TEXT))
                        bold = item.font()
                        bold.setBold(True)
                        item.setFont(bold)
                    elif j != 7:
                        item.setFont(mono)
                    if 3 <= j <= 6:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    else:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.history_cards['total'].setText(str(total))
            self.history_cards['sessions'].setText(str(len({r['session_id'] for r in rows})))
            self.history_stack.setCurrentIndex(0 if rows else 1)
        except Exception:
            self.logger.exception('Không định dạng được bảng lịch sử')

    def show_alert_detail(self, row, _column=0):
        table = self.history_table
        values = [table.item(row, c).text() if table.item(row, c) else '--'
                  for c in range(table.columnCount())]
        headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
        AlertDetailDialog(headers, values, self).exec()

    def export_history(self):
        target, _ = QFileDialog.getSaveFileName(
            self, 'Xuất lịch sử cảnh báo', str(REPORTS / 'lich_su_canh_bao.csv'), 'CSV (*.csv)'
        )
        if not target:
            return
        try:
            self.store.export_csv(target)
            QMessageBox.information(self, 'Thành công', f'Đã xuất lịch sử:\n{target}')
        except Exception:
            self.logger.exception('Không xuất được CSV')
            QMessageBox.critical(self, 'Lỗi xuất file', 'Không xuất được CSV. Xem app.log.')

    # ------------------------------------------------------------ Cài đặt
    def settings_page(self, layout):
        self.title(layout, 'CÀI ĐẶT HỆ THỐNG', 'Chọn camera và cấu hình nguồn hình ảnh')
        columns = QHBoxLayout()
        columns.setSpacing(12)

        card = panel()
        box = QVBoxLayout(card)
        box.setContentsMargins(24, 22, 24, 22)
        box.setSpacing(9)
        box.addWidget(label('NGUỒN CAMERA', 'Title'))
        box.addWidget(label('Loại nguồn hình ảnh', 'Subtitle'))

        # QComboBox thật vẫn là nguồn dữ liệu cho start_camera(); hai thẻ chọn chỉ điều khiển nó.
        self.source_type = QComboBox()
        self.source_type.addItems(['Webcam / Camera USB', 'DroidCam qua Wi-Fi'])
        self.source_type.setVisible(False)
        box.addWidget(self.source_type)
        self.opt_webcam = OptionCard('Webcam / Camera USB',
                                     'Kết nối trực tiếp qua cổng USB hoặc camera tích hợp')
        self.opt_droid = OptionCard('DroidCam qua Wi-Fi',
                                    'Dùng điện thoại làm camera qua mạng nội bộ')
        for index, option in enumerate((self.opt_webcam, self.opt_droid)):
            option.clicked.connect(lambda checked=False, i=index: self.source_type.setCurrentIndex(i))
            box.addWidget(option)

        box.addSpacing(4)
        box.addWidget(label('Chỉ số webcam', 'Subtitle'))
        self.camera_index = QComboBox()
        self.camera_index.addItems([str(i) for i in range(6)])
        try:
            from app.camera_cfg import DUNG_DROIDCAM, DROIDCAM_URL, WEBCAM_INDEX
            self.camera_index.setCurrentText(str(WEBCAM_INDEX))
            self.source_type.setCurrentIndex(1 if DUNG_DROIDCAM else 0)
        except ImportError:
            DROIDCAM_URL = ''
            self.camera_index.setCurrentText('0')
        box.addWidget(self.camera_index)
        box.addWidget(label('URL DroidCam', 'Subtitle'))
        self.droid_url = QLineEdit(DROIDCAM_URL)
        self.droid_url.setPlaceholderText('http://192.168.1.10:4747/video')
        box.addWidget(self.droid_url)
        notice = panel('InfoStrip')
        nl = QHBoxLayout(notice)
        nl.setContentsMargins(12, 8, 12, 8)
        nl.addWidget(label('ⓘ  Đổi nguồn camera: hãy dừng giám sát rồi bắt đầu lại.', 'Subtitle', True))
        box.addSpacing(4)
        box.addWidget(notice)
        box.addStretch()
        columns.addWidget(card, 3)

        info = panel()
        v = QVBoxLayout(info)
        v.setContentsMargins(24, 22, 24, 22)
        v.setSpacing(8)
        v.addWidget(label('THÔNG TIN HỆ THỐNG', 'Title'))
        v.addWidget(label('Ngưỡng mô hình đọc từ models/model.pkl. Tab Camera cho phép điều chỉnh bước 0.05.', 'Subtitle', True))
        v.addSpacing(4)
        v.addWidget(label('TỆP MÔ HÌNH', 'Caption'))
        self.model_rows = {}
        for name in ('model.pkl', 'face_landmarker.task', 'pose_landmarker_lite.task'):
            row = QHBoxLayout()
            row.addWidget(label(name, 'Subtitle'))
            row.addStretch()
            value = label('--', 'SideVal')
            row.addWidget(value)
            v.addLayout(row)
            self.model_rows[name] = value
        v.addSpacing(6)
        v.addWidget(divider())
        v.addWidget(label('LƯU TRỮ', 'Caption'))
        v.addWidget(label('Nhật ký: SQLite  •  Xuất CSV: reports/  •  EXE: %LOCALAPPDATA%\\AI_Driver_Monitor', 'Subtitle', True))
        v.addStretch()
        columns.addWidget(info, 2)
        layout.addLayout(columns, 1)

        tips = panel()
        tl = QHBoxLayout(tips)
        tl.setContentsMargins(24, 16, 24, 16)
        tl.setSpacing(24)
        for caption, text in (
            ('NGƯỠNG CẢNH BÁO', 'Điều chỉnh theo bước 0.05 ở tab Camera trực tiếp.'),
            ('ĐIỀU KHIỂN TRỰC TIẾP', 'Reset, ngày/đêm, học và lọc tư thế nằm ở tab Camera trực tiếp.'),
            ('ĐỔI NGUỒN CAMERA', 'Dừng giám sát rồi bắt đầu lại để áp dụng nguồn mới.'),
        ):
            cell = QVBoxLayout()
            cell.setSpacing(3)
            cell.addWidget(label(caption, 'Caption'))
            cell.addWidget(label(text, 'Subtitle', True))
            tl.addLayout(cell, 1)
        layout.addWidget(tips)

        self.source_type.currentIndexChanged.connect(self._sync_source_ui)
        self._sync_source_ui()

    def _sync_source_ui(self, _index=None):
        webcam = self.source_type.currentIndex() == 0
        self.opt_webcam.setChecked(webcam)
        self.opt_droid.setChecked(not webcam)
        self.camera_index.setEnabled(webcam)
        self.droid_url.setEnabled(not webcam)

    # ------------------------------------------------------------ UI-only refresh
    def _tick(self):
        now = datetime.now()
        for time_label, date_label in self._clocks:
            time_label.setText(f'{now:%H:%M:%S}')
            date_label.setText(f'{now:%d/%m/%Y}')

        cam_text = self.camera_chip.text().replace('●', '').replace('CAM 01  ·', '').strip()
        models_ok = all((MODELS / n).is_file() for n in
                        ('model.pkl', 'face_landmarker.task', 'pose_landmarker_lite.task'))
        for key, text, color in (
            ('camera', cam_text or '--', OK if cam_text == 'LIVE' else AMBER if cam_text == 'CONNECTING' else MUTED),
            ('model', 'SẴN SÀNG' if models_ok else 'THIẾU FILE', OK if models_ok else DANGER),
            ('session', f'#{self.session_id}' if self.session_id is not None else '—', TEXT),
        ):
            self.side_vals[key].setText(text)
            self.side_vals[key].setStyleSheet(f'color:{color};')

        running = self.worker is not None
        if running and self.alert_active:
            set_tone(self.run_chip, 'alert')
        elif running and 'MONITORING' in self.run_chip.text():
            set_tone(self.run_chip, 'ok')
        elif running:
            set_tone(self.run_chip, 'warn')
        else:
            set_tone(self.run_chip, 'idle')
        set_tone(self.camera_chip, 'live' if cam_text == 'LIVE' else
                 'warn' if cam_text == 'CONNECTING' else 'idle')

        # Mirror live values onto the full-screen camera page strip.
        self.strip['fps'].setText(self.fields['fps'].text())
        self.strip['ear'].setText(self.metrics['ear'].text())
        self.strip['mar'].setText(self.metrics['mar'].text())
        self.strip['prob'].setText(self.fields['prob'].text())
        self.strip['state'].setText(self.fields['state'].text())
        match = re.search(r'color:\s*(#[0-9A-Fa-f]{6})', self.fields['state'].styleSheet())
        self.strip['state'].setStyleSheet(f'color:{match.group(1) if match else TEXT};')

        if self.pages.currentIndex() == 3:
            for name, value in self.model_rows.items():
                present = (MODELS / name).is_file()
                value.setText('● SẴN SÀNG' if present else '● THIẾU')
                value.setStyleSheet(f'color:{OK if present else DANGER};')

    def confirm_stop(self):
        if self.worker is None:
            return
        if ConfirmStopDialog(self).exec() == QDialog.DialogCode.Accepted:
            self.stop_camera()

    # --------------------------- Keep original AI integration & event persistence ---------------------------
    def check_models(self):
        # Original project enables MediaPipe Pose by default.
        # Avoid importing app.realtime here: that would load the AI stack during UI startup.
        required = ['model.pkl', 'face_landmarker.task', 'pose_landmarker_lite.task']
        missing = [name for name in required if not (MODELS / name).is_file()]
        if missing:
            self.model_state.setText('Thiếu model: ' + ', '.join(missing))
            self.model_state.setStyleSheet(f'color:{DANGER};')
        else:
            self.model_state.setText('●  Mô hình AI đã sẵn sàng')
            self.model_state.setStyleSheet(f'color:{OK};')
        self.start_btn.setEnabled(not missing and self.worker is None)
        return not missing

    def start_camera(self):
        if self.worker is not None or not self.check_models():
            return
        source = (int(self.camera_index.currentText()) if self.source_type.currentIndex() == 0
                  else self.droid_url.text().strip())
        if not isinstance(source, int) and not source:
            QMessageBox.warning(self, 'Thiếu URL', 'Hãy nhập URL DroidCam.')
            return
        self.alert_count = 0
        self.events.clear()
        self.alert_active = False
        self.active_alert = None
        self.last_status = None
        self.last_image = None
        self.trend_graph.clear()
        self._last_plot = 0.0
        try:
            self.session_id = self.store.start_session(str(source))
            self.logger.info('Bắt đầu phiên %s, camera=%s', self.session_id, source)
        except Exception:
            self.logger.exception('Lỗi tạo phiên')
            QMessageBox.critical(self, 'Lỗi nhật ký', 'Không thể tạo phiên giám sát. Kiểm tra app.log.')
            return
        self.fields['alerts'].setText('0')
        self.worker = CameraWorker(source, self)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_ready.connect(self.on_status)
        self.worker.failure.connect(self.on_failure)
        self.worker.finished.connect(self.on_finished)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.fields['state'].setText('Đang khởi tạo...')
        self.camera_chip.setText('●  CONNECTING')
        self.run_chip.setText('●  CONNECTING')
        self.worker.start()

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.fields['state'].setText('Đang dừng...')

    def send(self, command):
        if self.worker is not None:
            self.worker.command(command)

    @Slot(QImage)
    def on_frame(self, image):
        self.last_image = image
        self.update_video()

    def update_video(self):
        if self.last_image is None:
            return
        # Preserve the camera orientation. Never mirror the frame in the Qt UI.
        pixmap = QPixmap.fromImage(self.last_image)
        for item in (self.preview, self.full_preview):
            if item.isVisible() and item.width() > 0 and item.height() > 0:
                item.setPixmap(pixmap.scaled(
                    item.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'preview'):
            self.update_video()

    @Slot(object)
    def on_status(self, d):
        active = bool(d.get('alert') or d.get('alert_long'))
        kind = ('Buồn ngủ' if d.get('alert') else
                'Mắt mở bất thường' if d.get('alert_long') else None)
        probability = d.get('probability') if d.get('alert') else d.get('probability_long')
        try:
            if kind != self.active_alert:
                if self.active_alert is not None:
                    self.store.end_alert(self.session_id)
                if kind is not None:
                    self.store.start_alert(self.session_id, kind, probability, d)
                    self.alert_count += 1
                    self.events.append((datetime.now().isoformat(timespec='seconds'), kind))
                self.active_alert = kind
            elif kind is not None:
                self.store.update_alert_peak(self.session_id, probability, d)
        except Exception:
            self.logger.exception('Không ghi được sự kiện cảnh báo')
        self.last_status = d
        self.alert_active = active
        if d.get('alert'):
            state, color = 'BUỒN NGỦ!', DANGER
        elif d.get('alert_long'):
            state, color = 'MẮT MỞ BẤT THƯỜNG', AMBER
        elif not d.get('face'):
            state, color = 'CHƯA THẤY MẶT', AMBER
        elif d.get('probability') is None:
            state, color = 'ĐANG PHÂN TÍCH', ACCENT
        else:
            state, color = 'ĐANG THEO DÕI', OK
        self.fields['state'].setText(state)
        self.fields['state'].setStyleSheet(f'font-size:19px; font-weight:800; color:{color};')
        self.run_chip.setText('●  ALERT' if active else '●  MONITORING')
        self.camera_chip.setText('●  LIVE')
        if active:
            self.banner.setText(
                '⚠  NGUY HIỂM: TÀI XẾ CÓ DẤU HIỆU BUỒN NGỦ — HÃY DỪNG XE AN TOÀN' if d.get('alert')
                else '⚠  CẢNH BÁO: MẮT MỞ NHƯNG NHÌN TRÂN TRÂN'
            )
        self.banner.setVisible(active)
        prob = d.get('probability')
        self.fields['prob'].setText('--' if prob is None else f'{prob * 100:.1f}%')
        self.prob_bar.setValue(0 if prob is None else int(round(max(0, min(1, prob)) * 100)))
        self.prob_bar.setStyleSheet(
            f'QProgressBar::chunk {{background:{DANGER if active else ACCENT}; border-radius:3px;}}'
        )
        self.fields['fps'].setText(f"{d.get('fps', 0):.0f} FPS")
        self.fields['alerts'].setText(str(self.alert_count))
        ear, mar, long_p = d.get('ear'), d.get('mar'), d.get('probability_long')
        self.metrics['ear'].setText('--' if ear is None else f'{ear:.3f}')
        self.metrics['mar'].setText('--' if mar is None else f'{mar:.3f}')
        self.metrics['long'].setText(
            'Không có model' if d.get('threshold_long') is None else
            '--' if long_p is None else f'{long_p * 100:.1f}%'
        )
        # EAR/MAR bars are visual scales only; they are NOT alert decision thresholds.
        self.ear_bar.setValue(0 if ear is None else int(max(0, min(1, ear / 0.5)) * 100))
        self.mar_bar.setValue(0 if mar is None else int(max(0, min(1, mar / 1.0)) * 100))
        self.long_bar.setValue(0 if long_p is None else int(max(0, min(1, long_p)) * 100))
        self.metrics['face'].setText('Đã nhận diện' if d.get('face') else 'Không tìm thấy')
        self.metrics['lighting'].setText(
            f"{str(d.get('lighting', '--')).upper()}  /  {d.get('lighting_mode', '--')}"
        )
        self.metrics['posture'].setText(str(d.get('posture', '--')))
        self.detail_status.setText(
            f"Ngưỡng: {d.get('threshold', 0):.2f}   •   Mẫu: {d.get('samples', 0)}   •   "
            f"FPS: {d.get('fps', 0):.0f}   •   Trạng thái: {state}"
        )
        current = time.monotonic()
        if current - self._last_plot >= 0.25:
            self._last_plot = current
            self.trend_graph.append(prob, d.get('threshold'))

    @Slot(str)
    def on_failure(self, details):
        self.logger.error('Realtime thất bại: %s', details)
        self.fields['state'].setText('LỖI CAMERA / AI')
        self.fields['state'].setStyleSheet(f'font-size:19px; font-weight:800; color:{DANGER};')
        QMessageBox.critical(self, 'Lỗi camera / mô hình', details[-3500:])

    @Slot()
    def on_finished(self):
        self.worker = None
        self.stop_btn.setEnabled(False)
        self.banner.hide()
        self.prob_bar.setValue(0)
        self.camera_chip.setText('●  OFFLINE')
        self.run_chip.setText('●  STANDBY')
        self.check_models()
        if self.fields['state'].text() != 'LỖI CAMERA / AI':
            self.fields['state'].setText('ĐÃ DỪNG')
            self.fields['state'].setStyleSheet(f'font-size:19px; font-weight:800; color:{MUTED};')
        try:
            if self.session_id is not None:
                self.store.end_alert(self.session_id)
                self.store.end_session(self.session_id, self.alert_count)
                self.logger.info('Đã kết thúc phiên %s', self.session_id)
                self.session_id = None
                self.active_alert = None
        except Exception:
            self.logger.exception('Lỗi kết thúc phiên')
        self.save_events()
        if self.pages.currentIndex() == 2:
            self.refresh_history()

    def save_events(self):
        if not self.events:
            return
        try:
            path = REPORTS / f'canh_bao_{datetime.now():%Y%m%d_%H%M%S}.csv'
            with path.open('w', encoding='utf-8-sig', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(['Thời gian', 'Loại cảnh báo'])
                writer.writerows(self.events)
            self.events.clear()
        except OSError:
            self.logger.exception('Không ghi được CSV cảnh báo của phiên')

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
            if not self.worker.wait(3500):
                QMessageBox.information(self, 'Camera chưa dừng',
                                        'Đợi camera hoàn tất rồi đóng ứng dụng.')
                event.ignore()
                return
        if self.session_id is not None:
            try:
                self.store.end_alert(self.session_id)
                self.store.end_session(self.session_id, self.alert_count)
                self.session_id = None
            except Exception:
                self.logger.exception('Lỗi đóng phiên khi thoát')
        self.save_events()
        super().closeEvent(event)