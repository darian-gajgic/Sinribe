"""Dark-glass Qt theme for Sinribe.

The palette constants are lifted verbatim from /home/sinep/Sinlate/theme.py:17-32 so Sinribe,
Sinlate and the wisprflow overlay read as one family of tools against the Win7-Aero desktop.
Only the delivery mechanism differs: Sinlate hand-rolls ttk styles, here it becomes QSS.
"""

from __future__ import annotations

# ---- palette (dark, glassy -- sits with the desktop's Win7-Aero theme) ---
BG = "#0c0d11"        # window backdrop
CARD = "#191c24"      # card / panel body
CARD_HI = "#262a35"   # top bevel highlight
BORDER = "#39415a"
ACCENT = "#5b93ff"    # brand blue
ACCENT2 = "#93b8ff"   # secondary accent
OKC = "#3ddc84"       # success
REC = "#ff5c5c"       # error
WARN = "#ffb454"      # warning banner
FG = "#f2f4f9"
SUBTLE = "#96a0b4"
BTN = "#242835"
BTN_HOV = "#2e3342"
BTN_BRD = "#404862"

QSS = f"""
QWidget {{
    background: {BG};
    color: {FG};
    font-family: "Inter", "Segoe UI", "Cantarell", "DejaVu Sans", sans-serif;
    font-size: 10.5pt;
}}

QFrame#Card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QFrame#Banner {{
    background: #2a2418;
    border: 1px solid {WARN};
    border-radius: 8px;
}}
QLabel#BannerText {{ color: {WARN}; background: transparent; }}

QLabel#H1 {{ font-size: 17pt; font-weight: 600; color: {FG}; background: transparent; }}
QLabel#H2 {{ font-size: 11.5pt; font-weight: 600; color: {ACCENT2}; background: transparent; }}
QLabel#Subtle {{ color: {SUBTLE}; background: transparent; }}
QLabel#Mono {{ font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
               font-size: 9.5pt; color: {SUBTLE}; background: transparent; }}
QLabel {{ background: transparent; }}

/* ---- buttons ---- */
QPushButton {{
    background: {BTN};
    border: 1px solid {BTN_BRD};
    border-radius: 7px;
    padding: 7px 15px;
    color: {FG};
}}
QPushButton:hover {{ background: {BTN_HOV}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {CARD_HI}; }}
QPushButton:disabled {{ color: #5d6478; border-color: #2b3145; background: #1b1e27; }}

QPushButton#Primary {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: #06101f;
    font-weight: 600;
    padding: 9px 24px;
}}
QPushButton#Primary:hover {{ background: {ACCENT2}; border-color: {ACCENT2}; }}
QPushButton#Primary:disabled {{ background: #2b3547; border-color: #2b3547; color: #5d6478; }}

QPushButton#Danger {{ border-color: {REC}; color: {REC}; }}
QPushButton#Danger:hover {{ background: #34222a; }}
QPushButton#Danger:disabled {{ border-color: #2b3145; color: #5d6478; background: #1b1e27; }}

QPushButton#Ghost {{
    background: transparent; border: none; color: {ACCENT2};
    padding: 4px 8px; text-align: left;
}}
QPushButton#Ghost:hover {{ color: {FG}; }}

QPushButton#Play {{
    background: {BTN}; border: 1px solid {BTN_BRD}; border-radius: 14px;
    min-width: 28px; max-width: 28px; min-height: 28px; max-height: 28px; padding: 0px;
}}
QPushButton#Play:hover {{ border-color: {ACCENT}; color: {ACCENT2}; }}

/* ---- inputs ---- */
QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {{
    background: #12141b;
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 9px;
    selection-background-color: {ACCENT};
    selection-color: #06101f;
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QLineEdit:read-only {{ color: {SUBTLE}; }}
QLineEdit#DropTarget[dragActive="true"] {{ border: 1px dashed {ACCENT}; background: #16202f; }}

QComboBox::drop-down {{ border: none; width: 22px; }}
/* A CSS triangle: zero-size box whose borders form the arrowhead. Without the explicit
   width/height:0 Qt reserves the border box and renders a small square instead. */
QComboBox::down-arrow {{
    image: none; width: 0px; height: 0px;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {SUBTLE}; margin-right: 9px;
}}
QComboBox::down-arrow:hover {{ border-top-color: {ACCENT2}; }}
QComboBox QAbstractItemView {{
    background: {CARD}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT}; selection-color: #06101f;
    outline: none; padding: 4px;
}}
QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; background: {BTN}; border: none; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {BTN_HOV}; }}

QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {BTN_BRD}; background: #12141b;
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

/* ---- progress ---- */
QProgressBar {{
    background: #12141b;
    border: 1px solid {BORDER};
    border-radius: 9px;
    height: 18px;
    text-align: center;
    color: {FG};
    font-size: 9pt;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 8px; }}
QProgressBar#Phase {{ height: 6px; border-radius: 3px; }}
QProgressBar#Phase::chunk {{ background: {ACCENT2}; border-radius: 3px; }}

/* ---- log pane ---- */
QPlainTextEdit#Log {{
    font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
    font-size: 9pt; color: {SUBTLE}; background: #0a0b0f;
}}

/* ---- misc chrome ---- */
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BTN_BRD}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QToolTip {{
    background: {CARD_HI}; color: {FG};
    border: 1px solid {BORDER}; border-radius: 5px; padding: 5px 8px;
}}
"""


def apply_theme(app) -> None:
    """Apply the Fusion base style + Sinribe QSS to a QApplication."""
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")  # the only cross-platform style that fully honours QSS colours
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(BG))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(FG))
    pal.setColor(QPalette.ColorRole.Base, QColor("#12141b"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(CARD))
    pal.setColor(QPalette.ColorRole.Text, QColor(FG))
    pal.setColor(QPalette.ColorRole.Button, QColor(BTN))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(FG))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#06101f"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(CARD_HI))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(FG))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
