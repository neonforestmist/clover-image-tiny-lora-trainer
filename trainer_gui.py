#!/usr/bin/env python3
"""Simple native desktop app for Clover LoRA training and Core ML creation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QAction, QCloseEvent, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import trainer_core as core
import train_lora


class ProcessThread(QThread):
    line_received = Signal(str)
    progress_changed = Signal(int)
    process_completed = Signal(int)

    def __init__(self, command: list[str], workflow: str) -> None:
        super().__init__()
        self.command = command
        self.workflow = workflow
        self.process: subprocess.Popen[str] | None = None

    def run(self) -> None:
        progress = 0
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=core.ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                clean = line.rstrip()
                self.line_received.emit(clean)
                if self.workflow == "training":
                    progress = core.training_progress(clean, progress)
                else:
                    progress = core.coreml_progress(clean, progress)
                self.progress_changed.emit(progress)
            code = self.process.wait()
        except Exception as error:  # noqa: BLE001 - display process errors
            self.line_received.emit(f"Could not run command: {error}")
            code = 1
        self.process_completed.emit(code)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


class DatasetPreviewThread(QThread):
    preview_ready = Signal(str, object)
    preview_failed = Signal(str)

    def __init__(self, dataset: str) -> None:
        super().__init__()
        self.dataset = dataset

    def run(self) -> None:
        try:
            status, samples = core.dataset_preview(self.dataset)
            self.preview_ready.emit(status, samples)
        except Exception as error:  # noqa: BLE001 - display dataset errors
            self.preview_failed.emit(str(error))


class CloverTrainerWindow(QMainWindow):
    def __init__(self, *, selected_tab: str = "training", demo: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Clover LoRA Trainer")
        self.resize(1240, 820)
        self.setMinimumSize(1040, 650)

        self.process_thread: ProcessThread | None = None
        self.preview_thread: DatasetPreviewThread | None = None
        self.active_log: QPlainTextEdit | None = None
        self.active_progress: QProgressBar | None = None
        self.active_status: QLabel | None = None
        self.active_workflow = ""
        self.training_output_dir = ""

        self._build_menu()
        self._build_window()
        self._preview_coreml_command()

        if demo:
            self.dataset_edit.setText("data/example-monet")
            self.dataset_edit.setCursorPosition(0)
            self._load_dataset_preview()
        if selected_tab == "coreml":
            self.tabs.setCurrentWidget(self.coreml_tab)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_window(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QLabel("Clover LoRA Trainer")
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        self.tabs = QTabWidget()
        self.training_tab = self._build_training_tab()
        self.coreml_tab = self._build_coreml_tab()
        self.tabs.addTab(self.training_tab, "Train Style")
        self.tabs.addTab(self.coreml_tab, "Core ML")
        self.tabs.currentChanged.connect(self._tab_changed)
        layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_training_tab(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        layout.addWidget(self._heading("Train a style"))
        intro = QLabel(
            "Choose one dataset folder containing images/ and metadata.jsonl. "
            "Clover fills the trigger phrase from the captions; you can edit it before training."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setVerticalSpacing(10)
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setMinimumWidth(780)
        self.dataset_edit.setPlaceholderText("my-style/ (contains images/ and metadata.jsonl)")
        form.addRow(
            "Dataset folder",
            self._path_row(self.dataset_edit, self._choose_dataset_folder, "Choose folder…"),
        )
        self.trigger_edit = QLineEdit()
        self.trigger_edit.setMinimumWidth(780)
        self.trigger_edit.setPlaceholderText("e.g. Storybook Anime Style")
        form.addRow("Trigger phrase", self.trigger_edit)
        layout.addLayout(form)

        self.training_destination = QLabel(
            "The style name and output folder are created from the dataset folder name."
        )
        self.training_destination.setWordWrap(True)
        layout.addWidget(self.training_destination)

        buttons = QHBoxLayout()
        self.load_preview_button = QPushButton("Check dataset")
        self.load_preview_button.clicked.connect(self._load_dataset_preview)
        self.training_start_button = QPushButton("Train style")
        self.training_start_button.clicked.connect(self._start_training)
        self.training_stop_button = QPushButton("Stop")
        self.training_stop_button.setEnabled(False)
        self.training_stop_button.clicked.connect(self._stop_process)
        buttons.addWidget(self.load_preview_button)
        buttons.addStretch(1)
        buttons.addWidget(self.training_stop_button)
        buttons.addWidget(self.training_start_button)
        layout.addLayout(buttons)

        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_status = QLabel("Choose your dataset folder to begin.")
        self.training_status.setWordWrap(True)
        layout.addWidget(self.training_progress)
        layout.addWidget(self.training_status)

        self.preview_status = QLabel("Expected layout: images/ and metadata.jsonl")
        self.preview_status.setWordWrap(True)
        layout.addWidget(self.preview_status)
        self.dataset_gallery = self._image_list(116)
        self.dataset_gallery.setFixedHeight(174)
        layout.addWidget(self.dataset_gallery)

        self.training_details_button = self._details_button("log and samples")
        self.training_details_button.toggled.connect(
            lambda shown: self._toggle_details(
                self.training_details_button,
                self.training_details,
                shown,
                "log and samples",
            )
        )
        layout.addWidget(self.training_details_button)

        self.training_details = self._build_training_details()
        self.training_details.hide()
        layout.addWidget(self.training_details)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _build_training_details(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 2, 0, 0)
        self.training_output_tabs = QTabWidget()
        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.sample_gallery = self._image_list(116)
        self.training_output_tabs.addTab(self.training_log, "Log")
        self.training_output_tabs.addTab(self.sample_gallery, "Samples")
        self.training_output_tabs.setMinimumHeight(180)
        layout.addWidget(self.training_output_tabs)
        return panel

    def _build_coreml_tab(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        layout.addWidget(self._heading("Create an iPhone Core ML style"))
        intro = QLabel(
            "Choose the trained LoRA and an output folder. Clover creates the same "
            "small style package used by the published iOS style repositories."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setVerticalSpacing(10)
        self.style_file_edit = QLineEdit()
        self.style_file_edit.setMinimumWidth(780)
        self.style_file_edit.setPlaceholderText(
            "outputs/my-style-lora/pytorch_lora_weights.safetensors"
        )
        self.style_file_edit.setCursorPosition(0)
        form.addRow(
            "Trained style",
            self._path_row(self.style_file_edit, self._choose_style_file, "Choose file…"),
        )
        self.coreml_output_edit = QLineEdit("coreml-models/my-style-coreml")
        self.coreml_output_edit.setMinimumWidth(780)
        self.coreml_output_edit.setCursorPosition(0)
        form.addRow(
            "Output folder",
            self._path_row(
                self.coreml_output_edit,
                self._choose_coreml_output,
                "Choose folder…",
            ),
        )
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.coreml_start_button = QPushButton("Create Core ML style")
        self.coreml_start_button.clicked.connect(self._start_coreml)
        self.coreml_stop_button = QPushButton("Stop")
        self.coreml_stop_button.setEnabled(False)
        self.coreml_stop_button.clicked.connect(self._stop_process)
        buttons.addStretch(1)
        buttons.addWidget(self.coreml_stop_button)
        buttons.addWidget(self.coreml_start_button)
        layout.addLayout(buttons)

        self.coreml_progress = QProgressBar()
        self.coreml_progress.setRange(0, 100)
        self.coreml_progress.setValue(0)
        self.coreml_status = QLabel("Choose the two paths, then create the style.")
        self.coreml_status.setWordWrap(True)
        layout.addWidget(self.coreml_progress)
        layout.addWidget(self.coreml_status)

        self.coreml_details_button = self._details_button("log and outputs")
        self.coreml_details_button.toggled.connect(self._toggle_coreml_details)
        layout.addWidget(self.coreml_details_button)

        self.coreml_advanced = self._build_coreml_advanced()
        self.coreml_advanced.hide()
        layout.addWidget(self.coreml_advanced)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _build_coreml_advanced(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(10)

        self.coreml_preview_button = QPushButton("Refresh command preview")
        self.coreml_preview_button.clicked.connect(self._preview_coreml_command)
        layout.addWidget(self.coreml_preview_button)
        self.coreml_command_box = QPlainTextEdit()
        self.coreml_command_box.setReadOnly(True)
        self.coreml_command_box.setMaximumHeight(74)
        layout.addWidget(self.coreml_command_box)

        self.coreml_log = QPlainTextEdit()
        self.coreml_log.setReadOnly(True)
        self.coreml_log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.coreml_log.setMinimumHeight(150)
        layout.addWidget(self.coreml_log)

        self.artifacts_tree = QTreeWidget()
        self.artifacts_tree.setHeaderLabels(["Output", "Type or size", "Path"])
        self.artifacts_tree.setRootIsDecorated(False)
        self.artifacts_tree.setMinimumHeight(110)
        layout.addWidget(self.artifacts_tree)
        self.refresh_artifacts_button = QPushButton("Refresh outputs")
        self.refresh_artifacts_button.clicked.connect(self._refresh_artifacts)
        layout.addWidget(self.refresh_artifacts_button)

        return panel

    @staticmethod
    def _scroll_page(page: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        font = label.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        label.setFont(font)
        return label

    @staticmethod
    def _details_button(label: str) -> QToolButton:
        button = QToolButton()
        button.setText(f"Show {label}")
        button.setCheckable(True)
        button.setArrowType(Qt.ArrowType.RightArrow)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        return button

    @staticmethod
    def _toggle_details(
        button: QToolButton,
        panel: QWidget,
        shown: bool,
        label: str,
    ) -> None:
        panel.setVisible(shown)
        button.setArrowType(
            Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow
        )
        button.setText(f"Hide {label}" if shown else f"Show {label}")

    def _toggle_coreml_details(self, shown: bool) -> None:
        self._toggle_details(
            self.coreml_details_button,
            self.coreml_advanced,
            shown,
            "log and outputs",
        )
        if shown:
            self._preview_coreml_command()
            self._refresh_artifacts()

    @staticmethod
    def _path_row(
        edit: QLineEdit,
        callback: Callable[[], None],
        button_title: str,
    ) -> QWidget:
        row = QWidget()
        row.setMinimumWidth(900)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton(button_title)
        button.clicked.connect(callback)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row

    @staticmethod
    def _image_list(icon_size: int) -> QListWidget:
        widget = QListWidget()
        widget.setViewMode(QListWidget.ViewMode.IconMode)
        widget.setIconSize(QSize(icon_size, icon_size))
        widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        widget.setMovement(QListWidget.Movement.Static)
        widget.setSpacing(8)
        widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        return widget

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Clover Trainer",
            "Clover Image Tiny LoRA Trainer\n\n"
            "A native Python desktop app for training styles and saving Core ML models.\n"
            "Code: Apache-2.0\nModel derivatives: CreativeML Open RAIL-M",
        )

    def _tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.coreml_tab:
            for edit in (
                self.style_file_edit,
                self.coreml_output_edit,
            ):
                edit.setCursorPosition(0)
        self.tabs.setFocus()

    def _choose_dataset_folder(self) -> None:
        self._choose_directory(self.dataset_edit, "Choose training images")
        if self.dataset_edit.text():
            self._load_dataset_preview()

    def _choose_coreml_output(self) -> None:
        self._choose_directory(self.coreml_output_edit, "Choose Core ML output")

    def _choose_style_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a Clover style",
            str(core.ROOT),
            "SafeTensors files (*.safetensors)",
        )
        if path:
            self.style_file_edit.setText(path)
            self.style_file_edit.setCursorPosition(0)

    def _choose_directory(self, edit: QLineEdit, title: str) -> None:
        current = core.resolve_path(edit.text())
        start = current if current.is_dir() else core.ROOT
        path = QFileDialog.getExistingDirectory(self, title, str(start))
        if path:
            edit.setText(path)
            edit.setCursorPosition(0)

    def _load_dataset_preview(self) -> None:
        if self.preview_thread is not None and self.preview_thread.isRunning():
            return
        if not self.dataset_edit.text().strip():
            QMessageBox.information(
                self,
                "Choose a dataset folder",
                "Choose the folder containing images/ and metadata.jsonl.",
            )
            return
        try:
            values = core.local_training_values(self.dataset_edit.text())
        except (OSError, ValueError, KeyError) as error:
            self.preview_status.setText(str(error))
            self.training_status.setText("The dataset folder is not ready.")
            self.training_destination.setText(
                "The style name and output folder are created from the dataset folder name."
            )
            self.dataset_gallery.clear()
            return
        self.training_output_dir = str(values["output_dir"])
        self.trigger_edit.setText(str(values["trigger"]))
        self.training_destination.setText(
            f"Style: {values['style']}  ·  Saves to: {self.training_output_dir}"
        )
        self.load_preview_button.setEnabled(False)
        self.preview_status.setText("Loading images…")
        self.preview_thread = DatasetPreviewThread(self.dataset_edit.text())
        self.preview_thread.preview_ready.connect(self._dataset_preview_ready)
        self.preview_thread.preview_failed.connect(self._dataset_preview_failed)
        self.preview_thread.finished.connect(self._dataset_preview_finished)
        self.preview_thread.start()

    def _dataset_preview_ready(self, status: str, samples: object) -> None:
        self.preview_status.setText(status)
        self.training_status.setText("Dataset ready. Select Train style to begin.")
        self._populate_images(self.dataset_gallery, list(samples), 116)

    def _dataset_preview_failed(self, error: str) -> None:
        self.preview_status.setText(f"Could not load images: {error}")
        self.dataset_gallery.clear()

    def _dataset_preview_finished(self) -> None:
        self.load_preview_button.setEnabled(True)
        if self.preview_thread is not None:
            self.preview_thread.deleteLater()
        self.preview_thread = None

    def _populate_images(
        self,
        widget: QListWidget,
        samples: list[tuple[Any, str]],
        icon_size: int,
    ) -> None:
        widget.clear()
        for source, caption in samples:
            pixmap = self._pixmap(source, icon_size)
            item = QListWidgetItem(caption[:42] + ("…" if len(caption) > 42 else ""))
            item.setToolTip(caption)
            if pixmap is not None and not pixmap.isNull():
                item.setIcon(QIcon(pixmap))
            item.setSizeHint(QSize(icon_size + 28, icon_size + 34))
            widget.addItem(item)

    @staticmethod
    def _pixmap(source: Any, icon_size: int) -> QPixmap | None:
        if isinstance(source, Path):
            pixmap = QPixmap(str(source))
        else:
            try:
                from PIL.ImageQt import ImageQt

                pixmap = QPixmap.fromImage(ImageQt(source))
            except Exception:  # noqa: BLE001 - thumbnails are optional
                return None
        return pixmap.scaled(
            QSize(icon_size, icon_size),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _start_training(self) -> None:
        if self.process_thread is not None:
            return
        if not self.dataset_edit.text().strip():
            QMessageBox.information(
                self,
                "Choose a dataset folder",
                "Choose the folder containing images/ and metadata.jsonl.",
            )
            return
        self.training_status.setText("Preparing the trainer…")
        QApplication.processEvents()
        try:
            values = core.local_training_values(
                self.dataset_edit.text(),
                self.trigger_edit.text() or None,
            )
            self.trigger_edit.setText(str(values["trigger"]))
            self.training_output_dir = str(values["output_dir"])
            self.training_destination.setText(
                f"Style: {values['style']}  ·  Saves to: {self.training_output_dir}"
            )
            config = core.make_config(values)
            command = core.training_command(
                config,
                train_lora.BASE_MODEL,
                "Full training",
                False,
                fetch=True,
            )
        except Exception as error:  # noqa: BLE001 - show startup errors
            self.training_status.setText("Could not start training.")
            QMessageBox.warning(self, "Could not start training", str(error))
            return
        self._start_process(
            command,
            workflow="training",
            log=self.training_log,
            progress=self.training_progress,
            status=self.training_status,
        )

    def _coreml_requirements(self) -> list[core.Requirement]:
        return core.coreml_style_requirements(
            self.style_file_edit.text(),
            self.coreml_output_edit.text(),
        )

    def _check_coreml_requirements(self) -> None:
        requirements = self._coreml_requirements()
        missing = sum(item.status == "Missing" for item in requirements)
        self.coreml_status.setText(
            "Ready to create the Core ML style."
            if missing == 0
            else f"{missing} path(s) still need attention."
        )

    def _preview_coreml_command(self) -> None:
        try:
            command = core.coreml_style_preview(
                self.style_file_edit.text(),
                self.coreml_output_edit.text(),
            )
            self.coreml_command_box.setPlainText(command)
        except Exception as error:  # noqa: BLE001 - keep main flow usable
            self.coreml_command_box.setPlainText(f"Could not build command: {error}")

    def _start_coreml(self) -> None:
        if self.process_thread is not None:
            return
        requirements = self._coreml_requirements()
        missing = [item.name for item in requirements if item.status == "Missing"]
        self._check_coreml_requirements()
        if missing:
            QMessageBox.warning(
                self,
                "Core ML setup is incomplete",
                "Choose or install these items first:\n\n" + "\n".join(missing),
            )
            return
        command = core.coreml_style_command(
            self.style_file_edit.text(),
            self.coreml_output_edit.text(),
        )
        self.coreml_command_box.setPlainText(core.display_command(command))
        self._start_process(
            command,
            workflow="coreml",
            log=self.coreml_log,
            progress=self.coreml_progress,
            status=self.coreml_status,
        )

    def _start_process(
        self,
        command: list[str],
        *,
        workflow: str,
        log: QPlainTextEdit,
        progress: QProgressBar,
        status: QLabel,
    ) -> None:
        log.clear()
        log.appendPlainText("$ " + shlex.join(command))
        progress.setValue(0)
        status.setText("Running…")
        self.active_log = log
        self.active_progress = progress
        self.active_status = status
        self.active_workflow = workflow
        self._set_process_buttons(running=True)

        self.process_thread = ProcessThread(command, workflow)
        self.process_thread.line_received.connect(log.appendPlainText)
        self.process_thread.progress_changed.connect(progress.setValue)
        self.process_thread.process_completed.connect(self._process_completed)
        self.process_thread.start()
        self.statusBar().showMessage(
            "Training is running" if workflow == "training" else "Saving Core ML model"
        )

    def _stop_process(self) -> None:
        if self.process_thread is None:
            return
        self.process_thread.stop()
        if self.active_status is not None:
            self.active_status.setText("Stopping…")

    def _process_completed(self, code: int) -> None:
        if self.active_status is not None:
            self.active_status.setText(
                "Complete."
                if code == 0
                else f"Failed with exit code {code}. Open the log for the exact error."
            )
        if code == 0 and self.active_progress is not None:
            self.active_progress.setValue(100)
        if self.active_workflow == "training":
            samples = [
                (path, path.name) for path in core.sample_images(self.training_output_dir)
            ]
            self._populate_images(self.sample_gallery, samples, 116)
            if code == 0:
                weights = core.resolve_path(self.training_output_dir) / (
                    "pytorch_lora_weights.safetensors"
                )
                if weights.is_file():
                    self.style_file_edit.setText(str(weights))
                    output_name = Path(self.training_output_dir).name.removesuffix("-lora")
                    self.coreml_output_edit.setText(
                        str(core.ROOT / "coreml-models" / f"{output_name}-coreml")
                    )
            else:
                self.training_details.show()
                self.training_details_button.setChecked(True)
                self.training_output_tabs.setCurrentWidget(self.training_log)
        else:
            self._refresh_artifacts()
            if code != 0:
                self.coreml_advanced.show()
                self.coreml_details_button.setChecked(True)
        self._set_process_buttons(running=False)
        if self.process_thread is not None:
            self.process_thread.deleteLater()
        self.process_thread = None
        self.active_log = None
        self.active_progress = None
        self.active_status = None
        self.active_workflow = ""
        self.statusBar().showMessage("Ready")

    def _set_process_buttons(self, *, running: bool) -> None:
        self.training_start_button.setEnabled(not running)
        self.coreml_start_button.setEnabled(not running)
        self.training_stop_button.setEnabled(
            running and self.active_workflow == "training"
        )
        self.coreml_stop_button.setEnabled(
            running and self.active_workflow == "coreml"
        )

    def _refresh_artifacts(self) -> None:
        self.artifacts_tree.clear()
        for artifact in core.coreml_style_artifacts(self.coreml_output_edit.text()):
            QTreeWidgetItem(
                self.artifacts_tree,
                [artifact.name, artifact.detail, str(artifact.path)],
            )
        self.artifacts_tree.resizeColumnToContents(0)
        self.artifacts_tree.resizeColumnToContents(1)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.process_thread is not None and self.process_thread.isRunning():
            answer = QMessageBox.question(
                self,
                "A process is still running",
                "Stop it and close the application?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.process_thread.stop()
            self.process_thread.wait(3000)
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab", choices=("training", "coreml"), default="training")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="show the bundled local dataset in the preview on startup",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Clover LoRA Trainer")
    window = CloverTrainerWindow(selected_tab=args.tab, demo=args.demo)
    window.show()
    window.raise_()
    window.activateWindow()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
