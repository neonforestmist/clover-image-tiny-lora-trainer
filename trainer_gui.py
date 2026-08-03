#!/usr/bin/env python3
"""Native desktop application for Clover LoRA training and Core ML export."""

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
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
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
    QSpinBox,
    QSplitter,
    QTabWidget,
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
        except Exception as error:  # noqa: BLE001 - report process startup errors
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
        except Exception as error:  # noqa: BLE001 - show dataset errors in the app
            self.preview_failed.emit(str(error))


class CloverTrainerWindow(QMainWindow):
    def __init__(self, *, selected_tab: str = "training", demo: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Clover Image Tiny LoRA Trainer")
        self.resize(1220, 840)
        self.setMinimumSize(980, 680)

        self.process_thread: ProcessThread | None = None
        self.preview_thread: DatasetPreviewThread | None = None
        self.active_log: QPlainTextEdit | None = None
        self.active_progress: QProgressBar | None = None
        self.active_status: QLabel | None = None
        self.active_workflow = ""

        self._build_menu()
        self._build_window()
        self._load_preset(next(iter(core.CONFIGS)))

        if demo:
            self.dataset_edit.setText("data/example-monet")
            self.dataset_edit.setCursorPosition(0)
            self._load_dataset_preview()
            self._preview_training_command()
        if selected_tab == "coreml":
            self.tabs.setCurrentWidget(self.coreml_tab)
            self._preview_coreml_command()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("Help")
        about_action = QAction("About Clover Trainer", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_window(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        title = QLabel("Clover Image Tiny LoRA Trainer")
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 6)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle = QLabel(
            "Train compact style files and export the shared stateful Core ML model. "
            "All work runs as local processes on this computer."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.training_tab = self._build_training_tab()
        self.coreml_tab = self._build_coreml_tab()
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.coreml_tab, "Core ML")
        self.tabs.currentChanged.connect(self._tab_changed)
        layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_training_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(10, 12, 10, 10)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_training_form())
        splitter.addWidget(self._build_training_run_panel())
        splitter.setSizes([450, 720])
        layout.addWidget(splitter)
        return tab

    def _build_training_form(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(390)
        container = QWidget()
        layout = QVBoxLayout(container)

        dataset_group = QGroupBox("1. Style and dataset")
        dataset_form = QFormLayout(dataset_group)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(core.CONFIGS))
        self.preset_combo.currentTextChanged.connect(self._load_preset)
        dataset_form.addRow("Recipe", self.preset_combo)
        self.style_edit = QLineEdit()
        dataset_form.addRow("Style name", self.style_edit)
        self.dataset_edit = QLineEdit()
        dataset_form.addRow("Dataset", self._path_row(self.dataset_edit, self._choose_dataset_folder))
        self.trigger_edit = QLineEdit()
        dataset_form.addRow("Trigger phrase", self.trigger_edit)
        self.validation_prompt_edit = QPlainTextEdit()
        self.validation_prompt_edit.setFixedHeight(58)
        dataset_form.addRow("Validation prompt", self.validation_prompt_edit)
        self.load_preview_button = QPushButton("Load dataset preview")
        self.load_preview_button.clicked.connect(self._load_dataset_preview)
        dataset_form.addRow("", self.load_preview_button)
        layout.addWidget(dataset_group)

        settings_group = QGroupBox("2. Training settings")
        settings_form = QFormLayout(settings_group)
        self.base_model_edit = QLineEdit(train_lora.BASE_MODEL)
        self.base_model_edit.setCursorPosition(0)
        settings_form.addRow("Base model", self.base_model_edit)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["5-step smoke test", "Full training"])
        settings_form.addRow("Run mode", self.mode_combo)

        steps_rank = QWidget()
        steps_rank_layout = QHBoxLayout(steps_rank)
        steps_rank_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(5, 10000)
        self.rank_combo = QComboBox()
        self.rank_combo.addItems(["4", "8", "16", "32"])
        steps_rank_layout.addWidget(QLabel("Steps"))
        steps_rank_layout.addWidget(self.steps_spin)
        steps_rank_layout.addSpacing(8)
        steps_rank_layout.addWidget(QLabel("Rank"))
        steps_rank_layout.addWidget(self.rank_combo)
        settings_form.addRow("Plan", steps_rank)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setRange(0.000001, 1.0)
        self.learning_rate_spin.setSingleStep(0.00001)
        settings_form.addRow("Learning rate", self.learning_rate_spin)

        batch_row = QWidget()
        batch_layout = QHBoxLayout(batch_row)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.accumulation_spin = QSpinBox()
        self.accumulation_spin.setRange(1, 128)
        batch_layout.addWidget(QLabel("Batch"))
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addSpacing(8)
        batch_layout.addWidget(QLabel("Accumulation"))
        batch_layout.addWidget(self.accumulation_spin)
        settings_form.addRow("Batching", batch_row)

        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["fp16", "bf16", "no"])
        settings_form.addRow("Mixed precision", self.precision_combo)
        self.checkpoint_spin = QSpinBox()
        self.checkpoint_spin.setRange(1, 10000)
        settings_form.addRow("Checkpoint interval", self.checkpoint_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        settings_form.addRow("Seed", self.seed_spin)
        self.output_edit = QLineEdit()
        settings_form.addRow("Output folder", self._path_row(self.output_edit, self._choose_output_folder))
        self.hub_edit = QLineEdit()
        settings_form.addRow("Hugging Face repo", self.hub_edit)
        self.push_checkbox = QCheckBox("Push the finished style after training")
        settings_form.addRow("", self.push_checkbox)
        layout.addWidget(settings_group)
        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_training_run_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        preview_group = QGroupBox("Dataset preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_status = QLabel("Select Load dataset preview to inspect up to 12 pairs.")
        self.preview_status.setWordWrap(True)
        preview_layout.addWidget(self.preview_status)
        self.dataset_gallery = self._image_list()
        self.dataset_gallery.setMinimumHeight(200)
        preview_layout.addWidget(self.dataset_gallery, 1)
        layout.addWidget(preview_group, 2)

        run_group = QGroupBox("3. Review and run")
        run_layout = QVBoxLayout(run_group)
        self.training_command_box = QPlainTextEdit()
        self.training_command_box.setReadOnly(True)
        self.training_command_box.setPlaceholderText("Preview the command before starting.")
        self.training_command_box.setMaximumHeight(98)
        run_layout.addWidget(self.training_command_box)

        buttons = QHBoxLayout()
        self.training_preview_button = QPushButton("Preview command")
        self.training_preview_button.clicked.connect(self._preview_training_command)
        self.training_start_button = QPushButton("Start training")
        self.training_start_button.clicked.connect(self._start_training)
        self.training_stop_button = QPushButton("Stop")
        self.training_stop_button.setEnabled(False)
        self.training_stop_button.clicked.connect(self._stop_process)
        buttons.addWidget(self.training_preview_button)
        buttons.addWidget(self.training_start_button)
        buttons.addWidget(self.training_stop_button)
        run_layout.addLayout(buttons)

        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_status = QLabel("Ready")
        run_layout.addWidget(self.training_progress)
        run_layout.addWidget(self.training_status)
        layout.addWidget(run_group)

        output_tabs = QTabWidget()
        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.sample_gallery = self._image_list()
        output_tabs.addTab(self.training_log, "Log")
        output_tabs.addTab(self.sample_gallery, "Validation samples")
        layout.addWidget(output_tabs, 2)
        return panel

    def _build_coreml_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 12, 10, 10)

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(self._build_coreml_input_panel())
        top_splitter.addWidget(self._build_coreml_check_panel())
        top_splitter.setSizes([580, 580])
        layout.addWidget(top_splitter, 2)

        run_group = QGroupBox("3. Run selected Core ML step")
        run_layout = QVBoxLayout(run_group)
        self.coreml_command_box = QPlainTextEdit()
        self.coreml_command_box.setReadOnly(True)
        self.coreml_command_box.setMaximumHeight(88)
        self.coreml_command_box.setPlaceholderText("Preview the command before starting.")
        run_layout.addWidget(self.coreml_command_box)

        buttons = QHBoxLayout()
        self.coreml_preview_button = QPushButton("Preview command")
        self.coreml_preview_button.clicked.connect(self._preview_coreml_command)
        self.coreml_start_button = QPushButton("Run selected step")
        self.coreml_start_button.clicked.connect(self._start_coreml)
        self.coreml_stop_button = QPushButton("Stop")
        self.coreml_stop_button.setEnabled(False)
        self.coreml_stop_button.clicked.connect(self._stop_process)
        buttons.addWidget(self.coreml_preview_button)
        buttons.addWidget(self.coreml_start_button)
        buttons.addWidget(self.coreml_stop_button)
        run_layout.addLayout(buttons)

        self.coreml_progress = QProgressBar()
        self.coreml_status = QLabel("Ready")
        run_layout.addWidget(self.coreml_progress)
        run_layout.addWidget(self.coreml_status)
        layout.addWidget(run_group)

        log_group = QGroupBox("Conversion log")
        log_layout = QVBoxLayout(log_group)
        self.coreml_log = QPlainTextEdit()
        self.coreml_log.setReadOnly(True)
        self.coreml_log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.coreml_log)
        layout.addWidget(log_group, 2)
        return tab

    def _build_coreml_input_panel(self) -> QWidget:
        group = QGroupBox("1. Inputs and workflow")
        form = QFormLayout(group)
        self.model_dir_edit = QLineEdit("/path/to/Clover-Image-Tiny")
        self.model_dir_edit.setCursorPosition(0)
        form.addRow("Clover model folder", self._path_row(self.model_dir_edit, self._choose_model_folder))
        self.style_file_edit = QLineEdit("outputs/monet-lora/pytorch_lora_weights.safetensors")
        self.style_file_edit.setCursorPosition(0)
        form.addRow("Style .safetensors", self._path_row(self.style_file_edit, self._choose_style_file))
        self.coreml_output_edit = QLineEdit("coreml-models/clover-stateful")
        self.coreml_output_edit.setCursorPosition(0)
        form.addRow("Core ML output", self._path_row(self.coreml_output_edit, self._choose_coreml_output))
        self.coreml_action_combo = QComboBox()
        self.coreml_action_combo.addItems(core.COREML_ACTIONS)
        self.coreml_action_combo.currentTextChanged.connect(self._coreml_action_changed)
        form.addRow("Workflow step", self.coreml_action_combo)
        self.psnr_spin = QDoubleSpinBox()
        self.psnr_spin.setRange(1, 100)
        self.psnr_spin.setDecimals(1)
        self.psnr_spin.setValue(35.0)
        form.addRow("Minimum PSNR", self.psnr_spin)
        note = QLabel(
            "Run the steps in order: export the stateful U-Net, compile it for Xcode, "
            "then validate base and style parity."
        )
        note.setWordWrap(True)
        form.addRow("", note)
        return group

    def _build_coreml_check_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        requirements_group = QGroupBox("2. Requirements")
        requirements_layout = QVBoxLayout(requirements_group)
        self.requirements_tree = QTreeWidget()
        self.requirements_tree.setHeaderLabels(["Requirement", "Status", "Details"])
        self.requirements_tree.setRootIsDecorated(False)
        self.requirements_tree.setAlternatingRowColors(True)
        self.requirements_tree.header().setStretchLastSection(True)
        requirements_layout.addWidget(self.requirements_tree)
        self.check_requirements_button = QPushButton("Check requirements")
        self.check_requirements_button.clicked.connect(self._check_coreml_requirements)
        requirements_layout.addWidget(self.check_requirements_button)
        layout.addWidget(requirements_group, 2)

        artifacts_group = QGroupBox("Output artifacts")
        artifacts_layout = QVBoxLayout(artifacts_group)
        self.artifacts_tree = QTreeWidget()
        self.artifacts_tree.setHeaderLabels(["Artifact", "Type or size", "Path"])
        self.artifacts_tree.setRootIsDecorated(False)
        self.artifacts_tree.header().setStretchLastSection(True)
        artifacts_layout.addWidget(self.artifacts_tree)
        self.refresh_artifacts_button = QPushButton("Refresh artifacts")
        self.refresh_artifacts_button.clicked.connect(self._refresh_artifacts)
        artifacts_layout.addWidget(self.refresh_artifacts_button)
        layout.addWidget(artifacts_group, 1)
        return panel

    @staticmethod
    def _path_row(edit: QLineEdit, callback: Callable[[], None]) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Choose…")
        button.clicked.connect(callback)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row

    @staticmethod
    def _image_list() -> QListWidget:
        widget = QListWidget()
        widget.setViewMode(QListWidget.ViewMode.IconMode)
        widget.setIconSize(QSize(140, 140))
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
            "Native Python desktop controls for Diffusers LoRA training and Core ML export.\n"
            "Code: Apache-2.0\nModel derivatives: CreativeML Open RAIL-M",
        )

    def _tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.coreml_tab:
            for edit in (
                self.model_dir_edit,
                self.style_file_edit,
                self.coreml_output_edit,
            ):
                edit.setCursorPosition(0)
        self.tabs.setFocus()

    def _choose_dataset_folder(self) -> None:
        self._choose_directory(self.dataset_edit, "Choose local imagefolder dataset")

    def _choose_output_folder(self) -> None:
        self._choose_directory(self.output_edit, "Choose training output folder")

    def _choose_model_folder(self) -> None:
        self._choose_directory(self.model_dir_edit, "Choose Clover model folder")

    def _choose_coreml_output(self) -> None:
        self._choose_directory(self.coreml_output_edit, "Choose Core ML output folder")

    def _choose_style_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Clover style weights",
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

    def _load_preset(self, name: str) -> None:
        if not name or name not in core.CONFIGS:
            return
        values = core.config_values(name)
        self.style_edit.setText(values["style"])
        self.dataset_edit.setText(values["dataset"])
        self.trigger_edit.setText(values["trigger"])
        self.validation_prompt_edit.setPlainText(values["validation_prompt"])
        self.output_edit.setText(values["output_dir"])
        self.steps_spin.setValue(values["max_train_steps"])
        self.rank_combo.setCurrentText(str(values["rank"]))
        self.learning_rate_spin.setValue(values["learning_rate"])
        self.batch_spin.setValue(values["train_batch_size"])
        self.accumulation_spin.setValue(values["gradient_accumulation_steps"])
        self.precision_combo.setCurrentText(values["mixed_precision"])
        self.checkpoint_spin.setValue(values["checkpointing_steps"])
        self.seed_spin.setValue(values["seed"])
        self.hub_edit.setText(values["hub_model_id"])
        for edit in (
            self.style_edit,
            self.dataset_edit,
            self.trigger_edit,
            self.output_edit,
            self.hub_edit,
        ):
            edit.setCursorPosition(0)

    def _training_values(self) -> dict[str, Any]:
        return {
            "style": self.style_edit.text(),
            "dataset": self.dataset_edit.text(),
            "trigger": self.trigger_edit.text(),
            "validation_prompt": self.validation_prompt_edit.toPlainText(),
            "output_dir": self.output_edit.text(),
            "max_train_steps": self.steps_spin.value(),
            "rank": int(self.rank_combo.currentText()),
            "learning_rate": self.learning_rate_spin.value(),
            "train_batch_size": self.batch_spin.value(),
            "gradient_accumulation_steps": self.accumulation_spin.value(),
            "mixed_precision": self.precision_combo.currentText(),
            "checkpointing_steps": self.checkpoint_spin.value(),
            "seed": self.seed_spin.value(),
            "hub_model_id": self.hub_edit.text(),
        }

    def _load_dataset_preview(self) -> None:
        if self.preview_thread is not None and self.preview_thread.isRunning():
            return
        self.load_preview_button.setEnabled(False)
        self.preview_status.setText("Loading dataset preview…")
        self.preview_thread = DatasetPreviewThread(self.dataset_edit.text())
        self.preview_thread.preview_ready.connect(self._dataset_preview_ready)
        self.preview_thread.preview_failed.connect(self._dataset_preview_failed)
        self.preview_thread.finished.connect(self._dataset_preview_finished)
        self.preview_thread.start()

    def _dataset_preview_ready(self, status: str, samples: object) -> None:
        self.preview_status.setText(status)
        self._populate_images(self.dataset_gallery, list(samples))

    def _dataset_preview_failed(self, error: str) -> None:
        self.preview_status.setText(f"Dataset preview failed: {error}")
        self.dataset_gallery.clear()

    def _dataset_preview_finished(self) -> None:
        self.load_preview_button.setEnabled(True)
        if self.preview_thread is not None:
            self.preview_thread.deleteLater()
        self.preview_thread = None

    def _populate_images(self, widget: QListWidget, samples: list[tuple[Any, str]]) -> None:
        widget.clear()
        for source, caption in samples:
            pixmap = self._pixmap(source)
            item = QListWidgetItem(caption[:54] + ("…" if len(caption) > 54 else ""))
            item.setToolTip(caption)
            if pixmap is not None and not pixmap.isNull():
                item.setIcon(QIcon(pixmap))
            item.setSizeHint(QSize(168, 184))
            widget.addItem(item)

    @staticmethod
    def _pixmap(source: Any) -> QPixmap | None:
        if isinstance(source, Path):
            pixmap = QPixmap(str(source))
        else:
            try:
                from PIL.ImageQt import ImageQt

                pixmap = QPixmap.fromImage(ImageQt(source))
            except Exception:  # noqa: BLE001 - thumbnails are optional
                return None
        return pixmap.scaled(
            QSize(140, 140),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    def _preview_training_command(self) -> None:
        try:
            command = core.training_preview(
                self._training_values(),
                self.base_model_edit.text(),
                self.mode_combo.currentText(),
                self.push_checkbox.isChecked(),
            )
            self.training_command_box.setPlainText(command)
            self.training_status.setText("Command ready")
        except Exception as error:  # noqa: BLE001
            self.training_status.setText(f"Could not build command: {error}")

    def _start_training(self) -> None:
        if self.process_thread is not None:
            return
        try:
            values = self._training_values()
            config = core.make_config(values)
            command = core.training_command(
                config,
                self.base_model_edit.text(),
                self.mode_combo.currentText(),
                self.push_checkbox.isChecked(),
                fetch=True,
            )
            self.training_command_box.setPlainText(core.display_command(command))
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Could not start training", str(error))
            return
        self._start_process(
            command,
            workflow="training",
            log=self.training_log,
            progress=self.training_progress,
            status=self.training_status,
        )

    def _coreml_action_changed(self, _action: str = "") -> None:
        self._preview_coreml_command()
        self._check_coreml_requirements()

    def _coreml_requirements(self) -> list[core.Requirement]:
        return core.coreml_requirements(
            self.coreml_action_combo.currentText(),
            self.model_dir_edit.text(),
            self.style_file_edit.text(),
            self.coreml_output_edit.text(),
        )

    def _check_coreml_requirements(self) -> None:
        requirements = self._coreml_requirements()
        self.requirements_tree.clear()
        for requirement in requirements:
            QTreeWidgetItem(
                self.requirements_tree,
                [requirement.name, requirement.status, requirement.detail],
            )
        self.requirements_tree.resizeColumnToContents(0)
        self.requirements_tree.resizeColumnToContents(1)
        missing = sum(requirement.status == "Missing" for requirement in requirements)
        self.coreml_status.setText(
            "Ready for this step" if missing == 0 else f"{missing} requirement(s) need attention"
        )

    def _preview_coreml_command(self) -> None:
        try:
            command = core.coreml_preview(
                self.coreml_action_combo.currentText(),
                self.model_dir_edit.text(),
                self.style_file_edit.text(),
                self.coreml_output_edit.text(),
                self.psnr_spin.value(),
            )
            self.coreml_command_box.setPlainText(command)
        except Exception as error:  # noqa: BLE001
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
                "Core ML requirements are missing",
                "Resolve these requirements before running:\n\n" + "\n".join(missing),
            )
            return
        command = core.coreml_command(
            self.coreml_action_combo.currentText(),
            self.model_dir_edit.text(),
            self.style_file_edit.text(),
            self.coreml_output_edit.text(),
            self.psnr_spin.value(),
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
        status.setText("Running")
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
        self.statusBar().showMessage("Training is running" if workflow == "training" else "Core ML task is running")

    def _stop_process(self) -> None:
        if self.process_thread is None:
            return
        self.process_thread.stop()
        if self.active_status is not None:
            self.active_status.setText("Stop requested")

    def _process_completed(self, code: int) -> None:
        if self.active_status is not None:
            self.active_status.setText("Complete" if code == 0 else f"Stopped with exit code {code}")
        if code == 0 and self.active_progress is not None:
            self.active_progress.setValue(100)
        if self.active_workflow == "training":
            samples = [(path, path.name) for path in core.sample_images(self.output_edit.text())]
            self._populate_images(self.sample_gallery, samples)
        else:
            self._refresh_artifacts()
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
        self.training_stop_button.setEnabled(running)
        self.coreml_stop_button.setEnabled(running)

    def _refresh_artifacts(self) -> None:
        self.artifacts_tree.clear()
        for artifact in core.coreml_artifacts(self.coreml_output_edit.text()):
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
                "Stop the active process and close the application?",
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
    app.setApplicationName("Clover Image Tiny LoRA Trainer")
    window = CloverTrainerWindow(selected_tab=args.tab, demo=args.demo)
    window.show()
    window.raise_()
    window.activateWindow()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
