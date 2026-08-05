"""Post-run speaker panel: listen to each detected person, rename them, re-export.

Renaming re-renders from the in-memory turns (and the sidecar JSON on disk), so it costs a
few milliseconds rather than re-running the GPU stages.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from .. import audio
from ..textfmt import hms


class SpeakerRow(QWidget):
    play_requested = Signal(float, float)   # start, end
    renamed = Signal()

    def __init__(self, name: str, stats: dict, parent=None):
        super().__init__(parent)
        self.original = name
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.play_btn = QPushButton("▶")
        self.play_btn.setObjectName("Play")
        self.play_btn.setToolTip("Play a sample of this speaker")
        start, end = stats.get("longest", (0.0, 0.0))
        self.play_btn.clicked.connect(
            lambda: self.play_requested.emit(float(start), float(end)))
        lay.addWidget(self.play_btn)

        tag = QLabel(name)
        tag.setMinimumWidth(78)
        tag.setObjectName("H2")
        lay.addWidget(tag)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(f"rename {name}…")
        self.edit.textChanged.connect(lambda _: self.renamed.emit())
        lay.addWidget(self.edit, 1)

        share = QLabel(f"{stats['share'] * 100:.0f}%  ·  {hms(stats['seconds'])}")
        share.setObjectName("Subtle")
        share.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        share.setMinimumWidth(110)
        lay.addWidget(share)

    def new_name(self) -> str:
        return self.edit.text().strip() or self.original


class SpeakersPanel(QFrame):
    """Shown once a job completes."""
    reexport_requested = Signal(dict)   # {"Person 1": "Prof. Müller", ...}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._rows: list[SpeakerRow] = []
        self._source: Path | None = None
        self._tmp: Path | None = None

        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)

        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(16, 14, 16, 14)
        self.v.setSpacing(10)

        head = QLabel("Speakers")
        head.setObjectName("H2")
        self.v.addWidget(head)

        hint = QLabel("Listen, give people real names, then re-export. "
                      "Re-exporting is instant — nothing is transcribed again.")
        hint.setObjectName("Subtle")
        hint.setWordWrap(True)
        self.v.addWidget(hint)

        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(8)
        self.v.addLayout(self.rows_box)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.reexport_btn = QPushButton("Re-export with these names")
        self.reexport_btn.setObjectName("Primary")
        self.reexport_btn.setEnabled(False)
        self.reexport_btn.clicked.connect(self._emit_reexport)
        btns.addWidget(self.reexport_btn)
        self.v.addLayout(btns)

    def load(self, stats: dict[str, dict], source: Path) -> None:
        self.clear()
        self._source = Path(source)
        for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["seconds"]):
            row = SpeakerRow(name, s)
            row.play_requested.connect(self._play)
            row.renamed.connect(lambda: self.reexport_btn.setEnabled(True))
            self.rows_box.addWidget(row)
            self._rows.append(row)
        self.reexport_btn.setEnabled(False)

    def clear(self) -> None:
        self._player.stop()
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows = []

    def _play(self, start: float, end: float) -> None:
        if not self._source or not self._source.exists():
            return
        length = max(2.0, min(7.0, end - start))
        try:
            self._player.stop()
            fd = Path(tempfile.gettempdir()) / f"sinribe-preview-{id(self)}.wav"
            audio.extract_clip(self._source, fd, start, length)
            self._tmp = fd
            self._player.setSource(QUrl.fromLocalFile(str(fd)))
            self._audio_out.setVolume(0.9)
            self._player.play()
        except Exception:  # noqa: BLE001 - a failed preview must never break the panel
            pass

    def names(self) -> dict[str, str]:
        return {r.original: r.new_name() for r in self._rows}

    def _emit_reexport(self) -> None:
        self.reexport_requested.emit(self.names())
