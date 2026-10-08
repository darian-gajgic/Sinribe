"""Sinribe main window."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

from .. import audio, presets, summary as summary_mod
from ..config import (
    ASR_MODELS, DIAR_PIPELINES, DOWNLOADS_DIR, LANGUAGES, diar_available, model_available,
    save_config, supports_hotwords,
)
from ..fetch import is_url, purge_old_downloads
from ..merge import speaker_stats
from ..pipeline import JobSpec, free_vram_mib, gpu_tenants, purge_old_jobs
from ..render import markdown, sidecar, subtitles
from ..textfmt import human_duration, hms
from .speakers_panel import SpeakersPanel
from .worker import start_job, start_probe


def _card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("Card")
    v = QVBoxLayout(f)
    v.setContentsMargins(16, 14, 16, 14)
    v.setSpacing(10)
    if title:
        lbl = QLabel(title)
        lbl.setObjectName("H2")
        v.addWidget(lbl)
    return f, v


class MainWindow(QWidget):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Sinribe")
        self.setAcceptDrops(True)
        self.setMinimumSize(760, 640)
        self.resize(880, 860)

        self._thread = None
        self._worker = None
        self._result: dict | None = None
        self._url_meta = None
        self._probe_thread = None
        self._probe_worker = None
        # Length of whatever source is currently loaded, so the quality slider can say what its
        # position costs in minutes rather than in multipliers.
        self._source_seconds = 0.0
        # Monotonic, not wall clock: a 4-hour job started before midnight would otherwise
        # report a negative elapsed time once the date rolls over.
        self._started_at = time.monotonic()
        self._last_detail = ""
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._update_elapsed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        host = QWidget()
        scroll.setWidget(host)
        root = QVBoxLayout(host)
        root.setContentsMargins(22, 20, 22, 22)
        root.setSpacing(14)

        self._build_header(root)
        self._build_banner(root)
        self._build_input(root)
        self._build_options(root)
        self._build_run(root)
        self._build_results(root)
        root.addStretch(1)

        self._refresh_vram_banner()
        purge_old_jobs(int(self.cfg.get("cache_days", 14)))
        if not self.cfg.get("keep_downloads", True):
            purge_old_downloads(DOWNLOADS_DIR, int(self.cfg.get("download_cache_days", 14)))

    # ---------------------------------------------------------------- header
    def _build_header(self, root: QVBoxLayout) -> None:
        box = QVBoxLayout()
        box.setSpacing(2)
        t = QLabel("Sinribe")
        t.setObjectName("H1")
        box.addWidget(t)
        s = QLabel("Offline transcription with speaker labels — runs entirely on this machine")
        s.setObjectName("Subtle")
        box.addWidget(s)
        root.addLayout(box)

    def _build_banner(self, root: QVBoxLayout) -> None:
        self.banner = QFrame()
        self.banner.setObjectName("Banner")
        bl = QHBoxLayout(self.banner)
        bl.setContentsMargins(12, 9, 12, 9)
        self.banner_text = QLabel("")
        self.banner_text.setObjectName("BannerText")
        self.banner_text.setWordWrap(True)
        bl.addWidget(self.banner_text, 1)
        self.banner.hide()
        root.addWidget(self.banner)

    def _refresh_vram_banner(self) -> None:
        free = free_vram_mib()
        if 0 <= free < 4000:
            tenants = ", ".join(gpu_tenants()) or "another process"
            self.banner_text.setText(
                f"Only {free} MiB of VRAM free — {tenants} is holding the rest. Sinribe will "
                f"automatically use a smaller precision, which is slower. Closing that process "
                f"first will speed things up."
            )
            self.banner.show()
        else:
            self.banner.hide()

    # ---------------------------------------------------------------- input
    def _build_input(self, root: QVBoxLayout) -> None:
        card, v = _card()
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(8)

        lab = QLabel("Audio or video file")
        lab.setObjectName("H2")
        g.addWidget(lab, 0, 0, 1, 3)

        self.file_edit = QLineEdit()
        self.file_edit.setObjectName("DropTarget")
        self.file_edit.setPlaceholderText("Choose a file, or drag one onto this window…")
        self.file_edit.setReadOnly(True)
        g.addWidget(self.file_edit, 1, 0, 1, 2)
        b = QPushButton("Browse…")
        b.clicked.connect(self._pick_file)
        g.addWidget(b, 1, 2)

        self.file_info = QLabel("")
        self.file_info.setObjectName("Mono")
        self.file_info.setWordWrap(True)
        g.addWidget(self.file_info, 2, 0, 1, 3)

        lab_or = QLabel("or paste a YouTube / podcast link")
        lab_or.setObjectName("H2")
        g.addWidget(lab_or, 3, 0, 1, 3)

        self.url_edit = QLineEdit(str(self.cfg.get("last_url", "")))
        self.url_edit.setPlaceholderText("https://www.youtube.com/watch?v=…")
        self.url_edit.setClearButtonEnabled(True)
        self.url_edit.textChanged.connect(self._on_url_changed)
        self.url_edit.returnPressed.connect(self._probe_url)
        g.addWidget(self.url_edit, 4, 0, 1, 2)
        self.check_btn = QPushButton("Check")
        self.check_btn.setToolTip("Look up the title and length without downloading yet")
        self.check_btn.clicked.connect(self._probe_url)
        g.addWidget(self.check_btn, 4, 2)

        self.url_info = QLabel("")
        self.url_info.setObjectName("Mono")
        self.url_info.setWordWrap(True)
        g.addWidget(self.url_info, 5, 0, 1, 3)

        lab2 = QLabel("Save transcript to")
        lab2.setObjectName("H2")
        g.addWidget(lab2, 6, 0, 1, 3)

        self.out_edit = QLineEdit(str(self.cfg.get("output_dir", "")))
        self.out_edit.setReadOnly(True)
        g.addWidget(self.out_edit, 7, 0, 1, 2)
        b2 = QPushButton("Browse…")
        b2.clicked.connect(self._pick_output)
        g.addWidget(b2, 7, 2)
        g.setColumnStretch(0, 1)

        v.addLayout(g)
        root.addWidget(card)

    # ---------------------------------------------------------------- options
    def _build_options(self, root: QVBoxLayout) -> None:
        card, v = _card()
        head = QHBoxLayout()
        lab = QLabel("Options")
        lab.setObjectName("H2")
        head.addWidget(lab)
        head.addStretch(1)
        self.opts_toggle = QPushButton("Show ▾")
        self.opts_toggle.setObjectName("Ghost")
        self.opts_toggle.clicked.connect(self._toggle_options)
        head.addWidget(self.opts_toggle)
        v.addLayout(head)

        self.opts_body = QWidget()
        g = QGridLayout(self.opts_body)
        g.setContentsMargins(0, 4, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(9)

        g.addWidget(QLabel("Model"), 0, 0)
        self.model_cb = QComboBox()
        for name in ASR_MODELS:
            ready = model_available(name)
            self.model_cb.addItem(name if ready else f"{name}  (not installed)", name)
            if not ready:
                self.model_cb.model().item(self.model_cb.count() - 1).setEnabled(False)
        i = self.model_cb.findData(str(self.cfg.get("asr_model", "auto")))
        self.model_cb.setCurrentIndex(max(0, i))
        # The model choice decides which model the quality caption describes and whether the
        # hotwords field is usable at all, so both have to follow it.
        self.model_cb.currentIndexChanged.connect(self._sync_quality)
        self.model_cb.setToolTip(
            "'auto' lets the quality setting below pick the model, which is usually what you\n"
            "want.\n\n"
            "The German models are fine-tuned on German alone. That is not automatically\n"
            "better: on a hard far-field interview large-v3 beat them clearly, and they are\n"
            "worth trying on clean, close-miked German. Measure with sinribe-eval rather than\n"
            "assuming. Install them with tools/convert_german_models.sh.")
        g.addWidget(self.model_cb, 0, 1)

        g.addWidget(QLabel("Language"), 0, 2)
        self.lang_cb = QComboBox()
        for code, name in LANGUAGES:
            self.lang_cb.addItem(name, code)
        idx = self.lang_cb.findData(self.cfg.get("language", "auto"))
        self.lang_cb.setCurrentIndex(max(0, idx))
        g.addWidget(self.lang_cb, 0, 3)

        g.addWidget(QLabel("Speakers"), 1, 0)
        self.spk_cb = QComboBox()
        self.spk_cb.addItem("Auto-detect", "auto")
        self.spk_cb.addItem("Exactly…", "exact")
        self.spk_cb.addItem("Between…", "range")
        i = self.spk_cb.findData(self.cfg.get("speaker_mode", "auto"))
        self.spk_cb.setCurrentIndex(max(0, i))
        self.spk_cb.currentIndexChanged.connect(self._sync_speaker_inputs)
        g.addWidget(self.spk_cb, 1, 1)

        spk_row = QHBoxLayout()
        self.num_spin = QSpinBox()
        self.num_spin.setRange(1, 20)
        self.num_spin.setValue(int(self.cfg.get("num_speakers", 2)))
        self.min_spin = QSpinBox()
        self.min_spin.setRange(1, 20)
        self.min_spin.setValue(int(self.cfg.get("min_speakers", 1)))
        self.max_spin = QSpinBox()
        self.max_spin.setRange(1, 20)
        self.max_spin.setValue(int(self.cfg.get("max_speakers", 6)))
        self.to_label = QLabel("to")
        for w in (self.num_spin, self.min_spin, self.to_label, self.max_spin):
            spk_row.addWidget(w)
        spk_row.addStretch(1)
        g.addLayout(spk_row, 1, 2, 1, 2)

        g.addWidget(QLabel("Diarizer"), 2, 0)
        self.diar_cb = QComboBox()
        for name in DIAR_PIPELINES:
            ready = diar_available(name)
            self.diar_cb.addItem(name if ready else f"{name}  (not installed)", name)
            if not ready:
                # Selectable-but-doomed is the worst of both: the job decodes the audio, starts
                # diarizing, and only then discovers the pipeline was never downloaded.
                self.diar_cb.model().item(self.diar_cb.count() - 1).setEnabled(False)
        i = self.diar_cb.findData(str(self.cfg.get("diar_pipeline", DIAR_PIPELINES[0])))
        self.diar_cb.setCurrentIndex(max(0, i))
        self.diar_cb.setToolTip(
            "community-1 is the current pyannote pipeline and is installed.\n"
            "speaker-diarization-3.1 needs its licence accepted on huggingface.co and then\n"
            "downloading; until it is in the local cache it cannot be selected, because the\n"
            "workers run fully offline and would fail part-way through the job.")
        g.addWidget(self.diar_cb, 2, 1, 1, 3)

        g.addWidget(QLabel("Names & terms"), 3, 0)
        self.hotwords_edit = QLineEdit(str(self.cfg.get("hotwords", "")))
        self.hotwords_edit.setPlaceholderText("Fujitsu, Siemens, VR-Bank, Dr. Meier …")
        self.hotwords_edit.setToolTip(
            "Proper nouns and jargon to expect, comma-separated. Whisper mangles names it has\n"
            "no reason to predict — in one interview 'Fujitsu' came out as 'fiuzi', 'jitze'\n"
            "and 'future service'.\n\n"
            "A weak lever, measured: it rescued a name the model was already close to and left\n"
            "the badly-heard ones alone, at a small cost to overall accuracy. Worth trying;\n"
            "worth checking with sinribe-eval before trusting.")
        g.addWidget(self.hotwords_edit, 3, 1, 1, 3)

        # RUNGS run fastest-first, so the raw slider position already reads left = quick,
        # right = careful, which is the direction a control labelled "Quality" implies. The
        # caption carries the actual promise: the target speed, and the WER it measured.
        g.addWidget(QLabel("Quality"), 4, 0)
        quality = QHBoxLayout()
        quality.setSpacing(10)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(0, len(presets.RUNGS) - 1)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.setPageStep(1)
        self.speed_slider.setMinimumWidth(180)
        current = presets.for_target(self.cfg.get("speed_target", 6))
        self.speed_slider.setValue(presets.RUNGS.index(current))
        self.speed_slider.valueChanged.connect(self._sync_quality)
        quality.addWidget(self.speed_slider, 1)
        self.speed_label = QLabel()
        self.speed_label.setMinimumWidth(150)
        quality.addWidget(self.speed_label)
        g.addLayout(quality, 4, 1, 1, 3)

        accuracy = QHBoxLayout()
        accuracy.setSpacing(22)
        self.cb_refine = QCheckBox("Voice-print speaker check")
        self.cb_refine.setChecked(bool(self.cfg.get("refine_speakers", True)))
        self.cb_refine.setToolTip(
            "After transcribing, compare every sentence against each speaker's voice and\n"
            "correct the diarizer where the match is decisive. Fixes short answers that get\n"
            "absorbed into the previous question. Costs a few seconds.")
        accuracy.addWidget(self.cb_refine)
        accuracy.addStretch(1)
        g.addLayout(accuracy, 5, 0, 1, 4)

        checks = QHBoxLayout()
        checks.setSpacing(22)
        self.cb_json = QCheckBox("Sidecar .json")
        self.cb_json.setChecked(bool(self.cfg.get("write_json", True)))
        self.cb_json.setToolTip("Word-level data — lets you rename speakers and re-export "
                                "later without transcribing again.")
        self.cb_srt = QCheckBox(".srt")
        self.cb_srt.setChecked(bool(self.cfg.get("write_srt", True)))
        self.cb_vtt = QCheckBox(".vtt")
        self.cb_vtt.setChecked(bool(self.cfg.get("write_vtt", True)))
        self.cb_llm = QCheckBox("LLM chapters + summary")
        self.cb_llm.setChecked(bool(self.cfg.get("llm_enrich", False)))
        self.cb_llm.setToolTip(f"Uses {self.cfg.get('llm_model')} on your local Ollama. "
                               f"Offline. Adds a few minutes at the end.")
        self.cb_review = QCheckBox("Flag uncertain passages")
        self.cb_review.setChecked(bool(self.cfg.get("review_section", True)))
        self.cb_review.setToolTip(
            "Append the passages the recogniser was least sure of, with timestamps.\n"
            "Measured on a hard interview, 57% of the words it flags are genuinely wrong —\n"
            "so this is the list to check by ear instead of re-listening to the whole\n"
            "recording. Every flagged passage is listed, in time order; a hard 65-minute\n"
            "interview produces a few hundred.")
        for w in (self.cb_json, self.cb_srt, self.cb_vtt, self.cb_llm, self.cb_review):
            checks.addWidget(w)
        checks.addStretch(1)
        g.addLayout(checks, 6, 0, 1, 4)

        # The summary gets its own row because it is a different kind of output from the ones
        # above: not another rendering of the transcript, but a second job that reads it.
        summary_row = QHBoxLayout()
        summary_row.setSpacing(22)
        self.cb_summary = QCheckBox("Summary + HTML presentation")
        self.cb_summary.setChecked(bool(self.cfg.get("web_summary", False)))
        self.cb_summary.setToolTip(
            "Write a briefing on the recording and render it as a standalone web page:\n"
            "headline figures, a 60-second version, one section per topic with forecast,\n"
            "reasoning, risks, impact and a recommendation, question checklists and sources.\n"
            "Saved next to the transcript as '<name> - Summary.html' and '.md'.\n\n"
            "Adds a few minutes at the end. Written by Claude Code using the login already\n"
            "on this machine, so it needs no API key.")
        self.cb_research = QCheckBox("Verify with web research")
        self.cb_research.setChecked(bool(self.cfg.get("summary_research", False)))
        self.cb_research.setToolTip(
            "Let the model search the web while writing: check the recording's checkable\n"
            "claims, add what has happened since, and collect real source links.\n"
            "Slower and costs more. Needs the Claude provider; the local model has no\n"
            "network access and will say so rather than inventing citations.")
        self.cb_summary.toggled.connect(self._sync_summary)
        summary_row.addWidget(self.cb_summary)
        summary_row.addWidget(self.cb_research)
        summary_row.addWidget(QLabel("Written by"))
        self.summary_cb = QComboBox()
        for value, label in (("auto", "auto"), ("claude-code", "Claude Code"),
                             ("claude", "Claude API (key)"), ("ollama", "local Ollama")):
            self.summary_cb.addItem(label, value)
        i = self.summary_cb.findData(str(self.cfg.get("summary_provider", "auto")))
        self.summary_cb.setCurrentIndex(max(0, i))
        self.summary_cb.currentIndexChanged.connect(self._sync_summary)
        self.summary_cb.setToolTip(
            "'auto' tries Claude Code, then the Claude API, then the local Ollama.\n\n"
            "Claude Code needs no API key: it runs the `claude` command you already use and\n"
            "signs in with the login on this machine, so a summary costs subscription usage\n"
            "rather than a separate bill. It sends the transcript once per job and writes each\n"
            "section with the earlier ones in view.\n\n"
            "Claude API (key) is the same models over the direct API, which enforces the output\n"
            "shape in the decoder rather than asking for it. Needs ANTHROPIC_API_KEY or a key\n"
            "in ~/.config/sinribe/anthropic_key.\n\n"
            "Local Ollama needs no network at all, but reads the recording in chunks and writes\n"
            "fewer, thinner sections.")
        summary_row.addWidget(self.summary_cb)
        self.summary_status = QLabel("")
        self.summary_status.setObjectName("Subtle")
        summary_row.addWidget(self.summary_status)
        summary_row.addStretch(1)
        g.addLayout(summary_row, 7, 0, 1, 4)

        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)

        self.opts_body.hide()
        v.addWidget(self.opts_body)
        root.addWidget(card)
        self._sync_speaker_inputs()
        self._sync_summary()
        self._sync_quality()

    def _toggle_options(self) -> None:
        vis = self.opts_body.isVisible()
        self.opts_body.setVisible(not vis)
        self.opts_toggle.setText("Show ▾" if vis else "Hide ▴")

    def _sync_speaker_inputs(self) -> None:
        mode = self.spk_cb.currentData()
        self.num_spin.setVisible(mode == "exact")
        for w in (self.min_spin, self.to_label, self.max_spin):
            w.setVisible(mode == "range")

    # ---------------------------------------------------------------- run
    def _sync_summary(self) -> None:
        """Grey out what the current choice cannot do, and name the model that would write it.

        The status label is the honest answer to "what happens if I press Start": it asks the
        provider layer, which probes for a key or a running Ollama without making a request.
        """
        on = self.cb_summary.isChecked()
        for w in (self.cb_research, self.summary_cb, self.summary_status):
            w.setEnabled(on)
        if not on:
            self.summary_status.setText("")
            return

        provider = self.summary_cb.currentData()
        # Web research needs a provider that can search. On the local path the checkbox is not
        # just ineffective, it is misleading, so it is made unreachable rather than explained
        # afterwards. Asked of the provider rather than hardcoded: both cloud paths can search.
        can_search = provider == "auto" or summary_mod.can_research(provider)
        self.cb_research.setEnabled(can_search)
        if not can_search:
            self.cb_research.setChecked(False)

        cfg = dict(self.cfg)
        cfg["summary_provider"] = provider
        try:
            chosen = summary_mod.pick(cfg)
            skipped = list(getattr(chosen, "skipped", []) or [])
            if skipped:
                # "auto" fell back. Said in the row itself, because a brief written by the 4B
                # local model while the user believes Claude wrote it is the failure this
                # label exists to prevent.
                self.summary_status.setText(f"\u2192 {chosen.label} (fallback, see tooltip)")
                self.summary_status.setToolTip("Skipped:\n" + "\n".join(skipped))
            else:
                self.summary_status.setText(f"\u2192 {chosen.label}")
                self.summary_status.setToolTip("")
        except summary_mod.Unavailable as e:
            # The specific reason, not a generic "unavailable": the whole point of probing here
            # is that the user learns what to fix before pressing Start rather than after.
            reason = str(e).strip().splitlines()[0]
            self.summary_status.setText(f"cannot run: {reason[:70]}"
                                        + ("\u2026" if len(reason) > 70 else ""))
            self.summary_status.setToolTip(str(e))

    def _build_run(self, root: QVBoxLayout) -> None:
        card, v = _card()
        top = QHBoxLayout()
        self.phase_lbl = QLabel("Ready")
        self.phase_lbl.setObjectName("H2")
        top.addWidget(self.phase_lbl)
        top.addStretch(1)
        self.detail_lbl = QLabel("")
        self.detail_lbl.setObjectName("Subtle")
        top.addWidget(self.detail_lbl)
        v.addLayout(top)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setFormat("%p%")
        v.addWidget(self.bar)

        row = QHBoxLayout()
        self.start_btn = QPushButton("Start transcription")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self._start)
        self.start_btn.setEnabled(False)
        row.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("Danger")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        row.addWidget(self.cancel_btn)
        row.addStretch(1)

        self.log_toggle = QPushButton("Show log ▾")
        self.log_toggle.setObjectName("Ghost")
        self.log_toggle.clicked.connect(self._toggle_log)
        row.addWidget(self.log_toggle)
        v.addLayout(row)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("Log")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        self.log_view.setFixedHeight(170)
        self.log_view.hide()
        v.addWidget(self.log_view)
        root.addWidget(card)

    def _toggle_log(self) -> None:
        vis = self.log_view.isVisible()
        self.log_view.setVisible(not vis)
        self.log_toggle.setText("Show log ▾" if vis else "Hide log ▴")

    # ---------------------------------------------------------------- results
    def _build_results(self, root: QVBoxLayout) -> None:
        self.speakers = SpeakersPanel()
        self.speakers.reexport_requested.connect(self._reexport)
        self.speakers.hide()
        root.addWidget(self.speakers)

        self.done_card, v = _card()
        row = QHBoxLayout()
        self.done_lbl = QLabel("")
        self.done_lbl.setWordWrap(True)
        row.addWidget(self.done_lbl, 1)
        v.addLayout(row)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.open_folder_btn = QPushButton("Open folder")
        self.open_folder_btn.clicked.connect(self._open_folder)
        btns.addWidget(self.open_folder_btn)
        self.open_md_btn = QPushButton("Open transcript")
        self.open_md_btn.clicked.connect(self._open_md)
        btns.addWidget(self.open_md_btn)
        # The primary action when there is a summary: it is the thing the user will actually
        # read, and the transcript is the evidence behind it.
        self.open_summary_btn = QPushButton("Open summary")
        self.open_summary_btn.setObjectName("Primary")
        self.open_summary_btn.clicked.connect(self._open_summary)
        self.open_summary_btn.hide()
        btns.addWidget(self.open_summary_btn)
        v.addLayout(btns)
        self.done_card.hide()
        root.addWidget(self.done_card)

    # ---------------------------------------------------------------- pickers
    def _pick_file(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(audio.AUDIO_SUFFIXES))
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an audio or video file",
            str(self.cfg.get("last_input_dir", str(Path.home()))),
            f"Media files ({exts});;All files (*)")
        if path:
            self._set_file(Path(path))

    def _pick_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose where to save transcripts", self.out_edit.text() or str(Path.home()))
        if path:
            self.out_edit.setText(path)
            self.cfg["output_dir"] = path

    def _set_file(self, path: Path) -> None:
        # A file and a URL are mutually exclusive sources; picking one clears the other so the
        # window never shows two inputs with no indication of which would actually be used.
        self.url_edit.blockSignals(True)
        self.url_edit.clear()
        self.url_edit.blockSignals(False)
        self.url_info.clear()
        self._url_meta = None

        self.file_edit.setText(str(path))
        self.cfg["last_input_dir"] = str(path.parent)
        try:
            info = audio.probe(path)
            self._source_seconds = info.duration
            self.file_info.setText(
                f"{hms(info.duration)}  ·  {info.codec}  ·  {info.channels}ch @ "
                f"{info.sample_rate} Hz  ·  estimated ~{self._estimate()}")
            self.start_btn.setEnabled(True)
            self._sync_quality()
        except audio.AudioError as e:
            self._source_seconds = 0.0
            self.file_info.setText(f"⚠  {e}")
            self.start_btn.setEnabled(False)

    # ---------------------------------------------------------------- url source
    def _on_url_changed(self, text: str) -> None:
        text = text.strip()
        self._url_meta = None
        self.url_info.clear()
        if text:
            self.file_edit.clear()
            self.file_info.clear()
        valid = is_url(text)
        self.check_btn.setEnabled(valid)
        # Checking first is optional — Start downloads anyway — so don't force the extra click.
        self.start_btn.setEnabled(valid or bool(self.file_edit.text()))
        if text and not valid:
            self.url_info.setText("⚠  that does not look like a link")

    def _probe_url(self) -> None:
        url = self.url_edit.text().strip()
        if not is_url(url):
            return
        self.check_btn.setEnabled(False)
        self.url_info.setText("looking it up…")
        cookies = str(self.cfg.get("yt_cookies_from_browser") or "") or None
        self._probe_thread, self._probe_worker = start_probe(url, cookies)
        self._probe_worker.done.connect(self._on_probe_done)
        self._probe_worker.failed.connect(self._on_probe_failed)

    def _on_probe_done(self, meta) -> None:
        self._url_meta = meta
        self._source_seconds = float(meta.duration or 0.0)
        self.check_btn.setEnabled(True)
        bits = [meta.title]
        if meta.uploader:
            bits.append(meta.uploader)
        if meta.duration:
            bits.append(hms(meta.duration))
        bits.append(f"estimated ~{self._estimate()}")
        self.url_info.setText("  ·  ".join(bits))
        self._sync_quality()
        self.start_btn.setEnabled(True)

    def _on_probe_failed(self, msg: str) -> None:
        self.check_btn.setEnabled(True)
        self.url_info.setText(f"⚠  {msg}")

    # ---------------------------------------------------------------- drag & drop
    def dragEnterEvent(self, ev) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            self.file_edit.setProperty("dragActive", True)
            self.file_edit.style().polish(self.file_edit)

    def dragLeaveEvent(self, ev) -> None:
        self.file_edit.setProperty("dragActive", False)
        self.file_edit.style().polish(self.file_edit)

    def dropEvent(self, ev) -> None:
        self.file_edit.setProperty("dragActive", False)
        self.file_edit.style().polish(self.file_edit)
        for url in ev.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_file():
                self._set_file(p)
                break

    # ---------------------------------------------------------------- job control
    def _rung(self) -> presets.Rung:
        return presets.RUNGS[self.speed_slider.value()]

    def _estimate(self, rung: presets.Rung | None = None) -> str:
        """How long the loaded file will take at this quality.

        Prefers what this machine has actually done over the shipped benchmark figure.
        """
        secs = getattr(self, "_source_seconds", 0.0)
        if not secs:
            return "?"
        rung = rung or self._rung()
        return human_duration(rung.duration_for(secs, presets.observed_rtf(self.cfg, rung)))

    def _sync_quality(self) -> None:
        """Keep the slider's caption honest about what the current position costs and buys."""
        rung = self._rung()
        model = presets.model_for({"asr_model": self.model_cb.currentData(),
                                   "speed_target": rung.target_rtf})
        # The abstract multiplier means little; "~22 min for this file" is the number someone
        # actually weighs against a fraction of a percent of accuracy.
        est = self._estimate(rung)
        self.speed_label.setText(rung.label() + (f"\n≈ {est} for this file" if est != "?" else ""))

        best = min((r.measured_wer for r in presets.RUNGS if r.measured_wer is not None),
                   default=None)
        lines = [rung.note]
        if rung.measured_wer is not None and best is not None and rung.measured_wer > best:
            lines.append(f"About {rung.measured_wer - best:.1%} more wrong words than the most "
                         f"accurate setting.")
        if not rung.recommended:
            rec = presets.recommended()
            lines.append(f"Recommended: {rec.label().replace('  ★ recommended', '')}"
                         + (f" (≈ {self._estimate(rec)})" if self._estimate(rec) != "?" else ""))
        # A pinned model silently overrides the rung's choice, which is how someone ends up
        # running a model that measured worse without any sign of it on screen.
        if self.model_cb.currentData() == "auto":
            lines.append(f"Model: {model}")
        else:
            lines.append(f"Model: {model} — pinned in the dropdown, overriding this setting's "
                         f"choice of {rung.model}. The measured numbers above are for "
                         f"{rung.model}.")
        if not model_available(model):
            lines.append("NOT INSTALLED — run tools/convert_german_models.sh")
        self.speed_label.setToolTip("\n\n".join(lines))
        self.speed_slider.setToolTip(self.speed_label.toolTip())

        # Better to make the broken combination unreachable than to explain it afterwards.
        usable = supports_hotwords(model)
        self.hotwords_edit.setEnabled(usable)
        if usable:
            self.hotwords_edit.setPlaceholderText("Fujitsu, Siemens, VR-Bank, Dr. Meier …")
        else:
            self.hotwords_edit.setPlaceholderText(f"not supported by {model}")
            self.hotwords_edit.setToolTip(
                f"{model} returns an empty transcript when it is given expected names, so the\n"
                f"field is disabled here. large-v3 and large-v3-turbo-german both support it.")

    def _collect_cfg(self) -> dict:
        self.cfg.update({
            "language": self.lang_cb.currentData(),
            "speaker_mode": self.spk_cb.currentData(),
            "num_speakers": self.num_spin.value(),
            "min_speakers": self.min_spin.value(),
            "max_speakers": self.max_spin.value(),
            "diar_pipeline": self.diar_cb.currentData(),
            "asr_model": self.model_cb.currentData(),
            "hotwords": self.hotwords_edit.text().strip(),
            "speed_target": self._rung().target_rtf,
            "refine_speakers": self.cb_refine.isChecked(),
            "write_json": self.cb_json.isChecked(),
            "write_srt": self.cb_srt.isChecked(),
            "write_vtt": self.cb_vtt.isChecked(),
            "llm_enrich": self.cb_llm.isChecked(),
            "review_section": self.cb_review.isChecked(),
            "web_summary": self.cb_summary.isChecked(),
            "summary_research": self.cb_research.isChecked(),
            "summary_provider": self.summary_cb.currentData(),
            "output_dir": self.out_edit.text(),
            "last_url": self.url_edit.text().strip(),
        })
        return self.cfg

    def _start(self) -> None:
        url = self.url_edit.text().strip()
        src: Path | None = None
        if url:
            if not is_url(url):
                QMessageBox.warning(self, "Sinribe", "That does not look like a link.")
                return
        else:
            src = Path(self.file_edit.text())
            if not src.is_file():
                QMessageBox.warning(self, "Sinribe", "Choose an audio file or paste a link first.")
                return
        out = Path(self.out_edit.text() or self.cfg["output_dir"]).expanduser()
        cfg = self._collect_cfg()
        save_config(cfg)

        # Checked here as well as in the pipeline, because here it costs a dialog and there it
        # costs the user starting a job, walking away, and coming back to a setup error.
        if cfg.get("web_summary"):
            try:
                provider = summary_mod.pick(cfg)
            except summary_mod.Unavailable as e:
                QMessageBox.warning(
                    self, "Sinribe",
                    f"The summary cannot be generated:\n\n{e}\n\n"
                    f"Untick 'Summary + HTML presentation' to transcribe without it.")
                return
            skipped = list(getattr(provider, "skipped", []) or [])
            if cfg.get("summary_research") and not provider.supports_research:
                QMessageBox.warning(
                    self, "Sinribe",
                    f"Web research was requested, but {provider.label} cannot search the web."
                    + ("\n\nSkipped:\n" + "\n".join(skipped) if skipped else "")
                    + "\n\nFix the Claude provider, or untick 'Verify with web research'.")
                return
            if skipped:
                answer = QMessageBox.question(
                    self, "Sinribe",
                    f"The summary will be written by {provider.label}, because:\n\n"
                    + "\n".join(skipped) + "\n\nContinue with this provider?")
                if answer != QMessageBox.StandardButton.Yes:
                    return

        self.log_view.clear()
        self.speakers.hide()
        self.done_card.hide()
        self._result = None
        self.bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._set_inputs_enabled(False)
        self._started_at = time.monotonic()
        self._tick.start()
        self._refresh_vram_banner()

        spec = JobSpec(input_path=src, output_dir=out, cfg=dict(cfg),
                       url=url or None)
        self._thread, self._worker = start_job(spec)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

    def _cancel(self) -> None:
        if self._worker:
            self.phase_lbl.setText("Cancelling…")
            self.cancel_btn.setEnabled(False)
            self._worker.cancel()

    def _set_inputs_enabled(self, on: bool) -> None:
        for w in (self.model_cb, self.lang_cb, self.spk_cb, self.diar_cb, self.num_spin,
                  self.min_spin, self.max_spin, self.cb_json, self.cb_srt, self.cb_vtt,
                  self.cb_llm, self.cb_summary):
            w.setEnabled(on)
        # Re-derives which of the summary controls are legal, instead of enabling all of them
        # and letting "web research" come back on the local provider.
        if on:
            self._sync_summary()
        else:
            for w in (self.cb_research, self.summary_cb):
                w.setEnabled(False)

    def _render_detail(self) -> None:
        secs = time.monotonic() - self._started_at
        bits = [d for d in (self._last_detail, f"{human_duration(secs)} elapsed") if d]
        self.detail_lbl.setText("  ·  ".join(bits))

    def _update_elapsed(self) -> None:
        self._render_detail()

    # ---------------------------------------------------------------- signals
    def _on_progress(self, frac: float, phase: str, detail: str) -> None:
        self.bar.setValue(int(frac * 1000))
        self.phase_lbl.setText(phase)
        self._last_detail = detail
        self._render_detail()

    def _on_log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def _on_finished(self, result: dict) -> None:
        self._tick.stop()
        self._result = result
        self.bar.setValue(1000)
        self.phase_lbl.setText("Done")
        # What this run demonstrated, which on a resumed job is not its wall clock. The old
        # version only caught a job served ENTIRELY from cache (elapsed < 5s) and printed the
        # end-to-end factor for everything else — so a re-run that reused all three of the
        # recommended rung's passes and re-ran only diarization announced "29.5× realtime" for a
        # setting labelled 10.7×, and then taught the slider to promise that speed for good.
        rtf = presets.trustworthy_rtf(result)
        note = presets.speed_note(result).strip(" ()")
        # The "resumed" wording is a fallback for a job too fast to have done anything, not a
        # rule about the clock: a two-minute clip really can decode in under five seconds, and
        # that run has a speed worth reporting.
        if rtf:
            detail = f"{human_duration(result['elapsed'])}  ·  {note}"
        elif result["elapsed"] < 5.0:
            detail = f"{human_duration(result['elapsed'])}  ·  resumed from cache"
        else:
            detail = human_duration(result["elapsed"]) + (f"  ·  {note}" if note else "")
        self.detail_lbl.setText(detail)
        # Teach the estimate what this machine actually does — but only from a run that did the
        # work. record_rtf ignores a zero, which is what a cache-served run reports.
        presets.record_rtf(self.cfg, result.get("speed_target", self.cfg.get("speed_target")), rtf)
        self._sync_quality()
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._set_inputs_enabled(True)

        self.speakers.load(result["stats"], Path(result["source"]))
        self.speakers.show()
        self._refresh_done_label()
        self.done_card.show()

    def _refresh_done_label(self) -> None:
        r = self._result
        if not r:
            return
        names = ", ".join(sorted(r["stats"]))
        lines = [
            f"Transcribed <b>{Path(r['source']).name}</b> — {len(r['stats'])} "
            f"speakers ({names}), {len(r['turns'])} turns.",
            f"Wrote {len(r['written'])} files to "
            f"<code>{Path(r['markdown_path']).parent}</code>",
        ]
        summary = r.get("summary")
        if summary:
            meta = summary.get("meta", {})
            note = f"{len(summary['sections'])} sections by {meta.get('label', 'a model')}"
            if meta.get("research"):
                checks = (summary.get("check") or {}).get("checks", [])
                stale = sum(1 for c in checks
                            if c.get("status") in ("outdated", "incorrect", "disputed"))
                note += (f", checked against the web ({stale} of {len(checks)} claims outdated "
                         f"or disputed)")
            elif meta.get("research_error"):
                note += f", <b>without the web check</b> ({meta['research_error'][:160]})"
            if r.get("summary_seconds"):
                note += f", in {human_duration(float(r['summary_seconds']))}"
            lines.append(f"Summary: {note}.")
        elif r.get("summary_error"):
            # Stated in the window, not just logged. The transcript is fine and the job says
            # "Done", so a summary that quietly failed would otherwise be noticed days later by
            # someone looking for a file that was never written.
            lines.append(f"<b>The summary was not written:</b> {r['summary_error']}")
        self.done_lbl.setText("<br>".join(lines))
        has_summary = bool(r.get("summary_path"))
        self.open_summary_btn.setVisible(has_summary)
        # Whichever of the two the user most likely wants gets the accent. The theme selects on
        # the object name, and Qt only re-reads that on an explicit unpolish/polish.
        self.open_md_btn.setObjectName("" if has_summary else "Primary")
        style = self.open_md_btn.style()
        style.unpolish(self.open_md_btn)
        style.polish(self.open_md_btn)

    def _on_failed(self, msg: str, kind: str) -> None:
        self._tick.stop()
        self.phase_lbl.setText("Failed")
        self.detail_lbl.setText("")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._set_inputs_enabled(True)
        self.log_view.show()
        self.log_toggle.setText("Hide log ▴")
        self._on_log(f"ERROR: {msg}")
        hint = ""
        if "licence" in msg or "gated" in msg.lower() or kind == "load":
            hint = ("\n\nIf this mentions a gated repository, accept the model licence on "
                    "huggingface.co with the account whose token is in "
                    "~/.config/sinribe/hf_token.")
        QMessageBox.critical(self, "Sinribe — transcription failed", msg + hint)

    def _on_cancelled(self) -> None:
        self._tick.stop()
        self.phase_lbl.setText("Cancelled")
        self.detail_lbl.setText("Partial work is cached — starting again will resume.")
        self.bar.setValue(0)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._set_inputs_enabled(True)

    # ---------------------------------------------------------------- re-export
    def _reexport(self, names: dict[str, str]) -> None:
        if not self._result:
            return
        turns = self._result["turns"]
        sidecar.apply_names(turns, names)
        stats = speaker_stats(turns)
        out_dir = Path(self._result["markdown_path"]).parent
        stem = Path(self._result["source"]).stem

        markdown.write(out_dir / f"{stem}.md",
                       markdown.render(turns, stats, self._result,
                                       title=self._result.get("title"),
                                       enrichment=self._result.get("enrichment"),
                                       review=bool(self.cfg.get("review_section", True)),
                                       review_max=int(self.cfg.get("review_max_spans", 0))))
        if self.cfg.get("write_json", True):
            payload = sidecar.build(turns, stats, self._result,
                                    enrichment=self._result.get("enrichment"),
                                    settings=dict(self.cfg))
            sidecar.write(out_dir / f"{stem}.sinribe.json", payload)
        if self.cfg.get("write_srt", True):
            subtitles.write(out_dir / f"{stem}.srt", subtitles.render_srt(turns))
        if self.cfg.get("write_vtt", True):
            subtitles.write(out_dir / f"{stem}.vtt", subtitles.render_vtt(turns))

        self._result["stats"] = stats
        self.speakers.load(stats, Path(self._result["source"]))
        self._refresh_done_label()
        self._on_log(f"re-exported with names: {', '.join(f'{k}->{v}' for k, v in names.items())}")

    # ---------------------------------------------------------------- open
    def _open_folder(self) -> None:
        if self._result:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(Path(self._result["markdown_path"]).parent)))

    def _open_md(self) -> None:
        if self._result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._result["markdown_path"]))

    def _open_summary(self) -> None:
        path = (self._result or {}).get("summary_path")
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    # ---------------------------------------------------------------- close
    def closeEvent(self, ev) -> None:
        if self._worker and self._thread and self._thread.isRunning():
            r = QMessageBox.question(
                self, "Sinribe", "A transcription is still running. Cancel it and quit?")
            if r != QMessageBox.StandardButton.Yes:
                ev.ignore()
                return
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait(8000)
        self._collect_cfg()
        save_config(self.cfg)
        ev.accept()
