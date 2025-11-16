"""
STXM Live Visualization Application v9
Displays live detector frames and STXM scan images from ZMQ data stream
Optimized for high-speed data acquisition and large detector frames

Enhanced Features:
- ROI (Region of Interest) selection with statistics
- Comprehensive export options (STXM, frames, ROI data)
- Pause/Resume acquisition
- Clear/Reset views
- Snapshot capture
- Colormap selection
- Crosshair cursor with pixel values
- Save/Load ROI positions
"""

import sys
import time
import numpy as np
import zmq
from collections import deque
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QThread, pyqtSignal


@dataclass
class ScanMetadata:
    """Scan metadata structure"""
    x_start: float
    x_stop: float
    x_num: int
    y_start: float
    y_stop: float
    y_num: int
    exposure_time_s: float
    detector_shape: tuple


class ZMQDataReceiver(QThread):
    """Background thread for receiving ZMQ data streams"""
    
    metadata_received = pyqtSignal(dict)
    frame_received = pyqtSignal(np.ndarray, int)
    status_update = pyqtSignal(str)
    
    def __init__(self, md_port=50001, data_port=50002):
        super().__init__()
        self.md_port = md_port
        self.data_port = data_port
        self.running = True
        self.paused = False
        
        # ZMQ context and sockets
        self.context = zmq.Context()
        self.md_socket = None
        self.data_socket = None
        
        # Frame counter
        self.frame_count = 0
        self.current_scan_frames = 0
        
    def run(self):
        """Main thread loop for receiving data"""
        try:
            # Setup sockets
            self.md_socket = self.context.socket(zmq.SUB)
            self.md_socket.connect(f"tcp://127.0.0.1:{self.md_port}")
            self.md_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.md_socket.setsockopt(zmq.RCVTIMEO, 100)
            
            self.data_socket = self.context.socket(zmq.SUB)
            self.data_socket.connect(f"tcp://127.0.0.1:{self.data_port}")
            self.data_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.data_socket.setsockopt(zmq.RCVTIMEO, 100)
            
            self.status_update.emit("Connected to data streams")
            
            # Poller for non-blocking receive
            poller = zmq.Poller()
            poller.register(self.md_socket, zmq.POLLIN)
            poller.register(self.data_socket, zmq.POLLIN)
            
            while self.running:
                try:
                    socks = dict(poller.poll(100))
                    
                    if self.paused:
                        continue
                    
                    # Check for metadata
                    if self.md_socket in socks:
                        md = self.md_socket.recv_json()
                        self.metadata_received.emit(md)
                        self.frame_count = 0
                        self.current_scan_frames = md['x_num'] * md['y_num']
                        self.status_update.emit(f"New scan: {md['x_num']}x{md['y_num']} points")
                    
                    # Check for frame data
                    if self.data_socket in socks:
                        md = self.data_socket.recv_json(flags=zmq.NOBLOCK)
                        frame = self.data_socket.recv(flags=zmq.NOBLOCK, copy=False)
                        
                        # Convert to numpy array
                        dtype = np.dtype(md['dtype'])
                        shape = tuple(md['shape'])
                        frame_array = np.frombuffer(frame, dtype=dtype).reshape(shape)
                        
                        self.frame_received.emit(frame_array.copy(), self.frame_count)
                        self.frame_count += 1
                        
                except zmq.Again:
                    continue
                except Exception as e:
                    self.status_update.emit(f"Error: {str(e)}")
                    
        except Exception as e:
            self.status_update.emit(f"Connection error: {str(e)}")
        finally:
            if self.md_socket:
                self.md_socket.close()
            if self.data_socket:
                self.data_socket.close()
    
    def pause(self):
        """Pause data reception"""
        self.paused = True
    
    def resume(self):
        """Resume data reception"""
        self.paused = False
    
    def stop(self):
        """Stop the receiver thread"""
        self.running = False


class STXMVisualizationApp(QtWidgets.QMainWindow):
    """Main application window with dual visualization"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STXM Live Visualization")
        self.resize(1700, 900)
        
        # Data storage
        self.current_metadata: Optional[ScanMetadata] = None
        self.stxm_data = None
        self.current_frame = None
        self.all_frames = []
        self.frame_buffer = deque(maxlen=10)
        self.snapshots = []
        
        # ROI
        self.roi = None
        self.roi_enabled = False
        self.roi_data_history = []
        
        # Crosshair
        self.vLine = None
        self.hLine = None
        self.crosshair_enabled = False
        
        # Performance tracking
        self.fps_counter = deque(maxlen=100)
        self.last_update_time = time.time()
        self.frame_count_total = 0
        
        # Downsampling factor
        self.downsample_factor = 1
        
        # Display update throttling (update display every Nth frame)
        self.display_update_interval = 1
        self.frames_since_display_update = 0
        
        # Available colormaps
        self.colormaps = ['viridis', 'grayscale']
        self.current_colormap = 'viridis'
        
        # Create custom grayscale colormap
        self.grayscale_cmap = pg.ColorMap(
            pos=np.array([0.0, 1.0]),
            color=np.array([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte)
        )
        
        # Setup UI
        self.setup_ui()
        
        # Start ZMQ receiver thread
        self.receiver = ZMQDataReceiver()
        self.receiver.metadata_received.connect(self.on_metadata_received)
        self.receiver.frame_received.connect(self.on_frame_received)
        self.receiver.status_update.connect(self.update_status)
        self.receiver.start()
        
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Status bar
        self.status_label = QtWidgets.QLabel("Waiting for data...")
        layout.addWidget(self.status_label)
        
        # === Control Panel 1: Display Settings ===
        control_group1 = QtWidgets.QGroupBox("Display Settings")
        control_layout1 = QtWidgets.QHBoxLayout()
        
        # Downsample control
        control_layout1.addWidget(QtWidgets.QLabel("Downsample:"))
        self.downsample_spin = QtWidgets.QSpinBox()
        self.downsample_spin.setMinimum(1)
        self.downsample_spin.setMaximum(10)
        self.downsample_spin.setValue(1)
        self.downsample_spin.valueChanged.connect(self.on_downsample_changed)
        control_layout1.addWidget(self.downsample_spin)
        
        # Display update interval
        control_layout1.addWidget(QtWidgets.QLabel("Update Every:"))
        self.update_interval_spin = QtWidgets.QSpinBox()
        self.update_interval_spin.setMinimum(1)
        self.update_interval_spin.setMaximum(20)
        self.update_interval_spin.setValue(1)
        self.update_interval_spin.setSuffix(" frames")
        self.update_interval_spin.valueChanged.connect(self.on_update_interval_changed)
        control_layout1.addWidget(self.update_interval_spin)
        
        # Colormap selection
        control_layout1.addWidget(QtWidgets.QLabel("Colormap:"))
        self.colormap_combo = QtWidgets.QComboBox()
        self.colormap_combo.addItems(self.colormaps)
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        control_layout1.addWidget(self.colormap_combo)
        
        # Auto-scale checkbox
        self.autoscale_check = QtWidgets.QCheckBox("Auto-scale")
        self.autoscale_check.setChecked(True)
        control_layout1.addWidget(self.autoscale_check)
        
        # Frame averaging
        self.avg_frames_check = QtWidgets.QCheckBox("Average frames")
        self.avg_frames_check.setChecked(False)
        control_layout1.addWidget(self.avg_frames_check)
        
        control_layout1.addStretch()
        control_group1.setLayout(control_layout1)
        layout.addWidget(control_group1)
        
        # === Control Panel 2: Tools ===
        control_group2 = QtWidgets.QGroupBox("Analysis Tools")
        control_layout2 = QtWidgets.QHBoxLayout()
        
        # ROI toggle
        self.roi_check = QtWidgets.QCheckBox("Enable ROI")
        self.roi_check.setChecked(False)
        self.roi_check.stateChanged.connect(self.on_roi_toggled)
        control_layout2.addWidget(self.roi_check)
        
        # Save/Load ROI
        self.save_roi_btn = QtWidgets.QPushButton("Save ROI")
        self.save_roi_btn.clicked.connect(self.save_roi_position)
        self.save_roi_btn.setEnabled(False)
        control_layout2.addWidget(self.save_roi_btn)
        
        self.load_roi_btn = QtWidgets.QPushButton("Load ROI")
        self.load_roi_btn.clicked.connect(self.load_roi_position)
        control_layout2.addWidget(self.load_roi_btn)
        
        # Crosshair toggle
        self.crosshair_check = QtWidgets.QCheckBox("Crosshair")
        self.crosshair_check.setChecked(False)
        self.crosshair_check.stateChanged.connect(self.on_crosshair_toggled)
        control_layout2.addWidget(self.crosshair_check)
        
        # Export ROI data
        self.export_roi_btn = QtWidgets.QPushButton("Export ROI Data")
        self.export_roi_btn.clicked.connect(self.export_roi_data)
        self.export_roi_btn.setEnabled(False)
        control_layout2.addWidget(self.export_roi_btn)
        
        control_layout2.addStretch()
        control_group2.setLayout(control_layout2)
        layout.addWidget(control_group2)
        
        # === Control Panel 3: Acquisition & Export ===
        control_group3 = QtWidgets.QGroupBox("Acquisition & Export")
        control_layout3 = QtWidgets.QHBoxLayout()
        
        # Pause/Resume button
        self.pause_btn = QtWidgets.QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setCheckable(True)
        control_layout3.addWidget(self.pause_btn)
        
        # Clear button
        self.clear_btn = QtWidgets.QPushButton("🗑 Clear Views")
        self.clear_btn.clicked.connect(self.clear_views)
        control_layout3.addWidget(self.clear_btn)
        
        # Snapshot button
        self.snapshot_btn = QtWidgets.QPushButton("📷 Snapshot")
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        control_layout3.addWidget(self.snapshot_btn)
        
        control_layout3.addWidget(QtWidgets.QLabel("|"))
        
        # Export buttons
        self.export_stxm_btn = QtWidgets.QPushButton("💾 Export STXM")
        self.export_stxm_btn.clicked.connect(self.export_stxm_data)
        control_layout3.addWidget(self.export_stxm_btn)
        
        self.export_frame_btn = QtWidgets.QPushButton("💾 Export Frame")
        self.export_frame_btn.clicked.connect(self.export_current_frame)
        control_layout3.addWidget(self.export_frame_btn)
        
        self.export_all_btn = QtWidgets.QPushButton("💾 Export All")
        self.export_all_btn.clicked.connect(self.export_all_frames)
        control_layout3.addWidget(self.export_all_btn)
        
        self.export_snapshots_btn = QtWidgets.QPushButton("💾 Export Snapshots")
        self.export_snapshots_btn.clicked.connect(self.export_snapshots)
        control_layout3.addWidget(self.export_snapshots_btn)
        
        control_layout3.addStretch()
        
        # FPS display
        self.fps_label = QtWidgets.QLabel("FPS: 0.0")
        control_layout3.addWidget(self.fps_label)
        
        control_group3.setLayout(control_layout3)
        layout.addWidget(control_group3)
        
        # Graphics layout for plots
        self.graphics_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_widget)
        
        # Detector frame view (left)
        self.frame_plot = self.graphics_widget.addPlot(title="Detector Frame")
        self.frame_img = pg.ImageItem()
        self.frame_plot.addItem(self.frame_img)
        self.frame_plot.setAspectLocked(True)
        
        # Add colorbar for frame
        self.frame_colorbar = pg.ColorBarItem(
            values=(0, 1),
            colorMap=pg.colormap.get('viridis'),
            width=15
        )
        self.frame_colorbar.setImageItem(self.frame_img)
        self.graphics_widget.addItem(self.frame_colorbar)
        
        # STXM image view (right)
        self.stxm_plot = self.graphics_widget.addPlot(title="STXM Image")
        self.stxm_img = pg.ImageItem()
        self.stxm_plot.addItem(self.stxm_img)
        self.stxm_plot.setAspectLocked(True)
        
        # Add colorbar for STXM
        self.stxm_colorbar = pg.ColorBarItem(
            values=(0, 1),
            colorMap=pg.colormap.get('viridis'),
            width=15
        )
        self.stxm_colorbar.setImageItem(self.stxm_img)
        self.graphics_widget.addItem(self.stxm_colorbar)
        
        # Info labels
        info_layout = QtWidgets.QHBoxLayout()
        self.roi_info_label = QtWidgets.QLabel("ROI: Not set")
        info_layout.addWidget(self.roi_info_label)
        self.pixel_info_label = QtWidgets.QLabel("Pixel: -")
        info_layout.addWidget(self.pixel_info_label)
        info_layout.addStretch()
        self.snapshot_info_label = QtWidgets.QLabel("Snapshots: 0")
        info_layout.addWidget(self.snapshot_info_label)
        layout.addLayout(info_layout)
        
    def on_downsample_changed(self, value):
        """Handle downsample factor change"""
        self.downsample_factor = value
        self.update_status(f"Downsample factor set to {value}")
    
    def on_update_interval_changed(self, value):
        """Handle display update interval change"""
        self.display_update_interval = value
        self.update_status(f"Display update interval set to every {value} frame(s)")
    
    def on_colormap_changed(self, colormap_name):
        """Handle colormap change"""
        self.current_colormap = colormap_name
        
        if colormap_name == 'grayscale':
            cmap = self.grayscale_cmap
        else:
            cmap = pg.colormap.get(colormap_name)
        
        self.frame_colorbar.setColorMap(cmap)
        self.stxm_colorbar.setColorMap(cmap)
        self.update_status(f"Colormap changed to {colormap_name}")
    
    def toggle_pause(self):
        """Toggle pause/resume"""
        if self.pause_btn.isChecked():
            self.receiver.pause()
            self.pause_btn.setText("▶ Resume")
            self.update_status("Acquisition PAUSED")
        else:
            self.receiver.resume()
            self.pause_btn.setText("⏸ Pause")
            self.update_status("Acquisition RESUMED")
    
    def clear_views(self):
        """Clear all views and reset data"""
        reply = QtWidgets.QMessageBox.question(
            self, 'Clear Views',
            'Clear all data and views?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.stxm_data = None
            self.current_frame = None
            self.all_frames = []
            self.frame_buffer.clear()
            self.frame_img.clear()
            self.stxm_img.clear()
            self.frame_count_total = 0
            self.fps_counter.clear()
            self.update_status("Views cleared")
    
    def take_snapshot(self):
        """Take snapshot of current frame and STXM data"""
        if self.current_frame is not None and self.stxm_data is not None:
            snapshot = {
                'timestamp': time.time(),
                'frame': self.current_frame.copy(),
                'stxm_data': self.stxm_data.copy(),
                'metadata': self.current_metadata
            }
            self.snapshots.append(snapshot)
            self.snapshot_info_label.setText(f"Snapshots: {len(self.snapshots)}")
            self.update_status(f"Snapshot captured (total: {len(self.snapshots)})")
        else:
            QtWidgets.QMessageBox.warning(self, "Snapshot Error", "No data to snapshot")
    
    def on_roi_toggled(self, state):
        """Handle ROI toggle"""
        self.roi_enabled = (state == QtCore.Qt.Checked)
        
        if self.roi_enabled:
            if self.roi is None and self.current_frame is not None:
                # Create ROI on detector frame
                h, w = self.current_frame.shape
                roi_size = min(h, w) // 4
                roi_x = w // 2 - roi_size // 2
                roi_y = h // 2 - roi_size // 2
                
                self.roi = pg.ROI([roi_x, roi_y], [roi_size, roi_size], pen='r')
                self.roi.addScaleHandle([1, 1], [0, 0])
                self.roi.addScaleHandle([0, 0], [1, 1])
                self.roi.sigRegionChanged.connect(self.on_roi_changed)
                self.frame_plot.addItem(self.roi)
                self.on_roi_changed()
            self.save_roi_btn.setEnabled(True)
            self.export_roi_btn.setEnabled(True)
        else:
            if self.roi is not None:
                self.frame_plot.removeItem(self.roi)
                self.roi = None
                self.roi_info_label.setText("ROI: Disabled")
            self.save_roi_btn.setEnabled(False)
            self.export_roi_btn.setEnabled(False)
    
    def on_roi_changed(self):
        """Handle ROI region change"""
        if self.roi is not None and self.current_frame is not None:
            # Get ROI bounds
            roi_slice, roi_data = self.roi.getArrayRegion(
                self.current_frame, self.frame_img, returnMappedCoords=True
            )
            
            # Calculate ROI statistics
            roi_mean = np.mean(roi_slice)
            roi_sum = np.sum(roi_slice)
            roi_max = np.max(roi_slice)
            roi_min = np.min(roi_slice)
            roi_std = np.std(roi_slice)
            
            # Store ROI data for export
            self.roi_data_history.append({
                'timestamp': time.time(),
                'mean': roi_mean,
                'sum': roi_sum,
                'max': roi_max,
                'min': roi_min,
                'std': roi_std,
                'shape': roi_slice.shape
            })
            
            self.roi_info_label.setText(
                f"ROI - Mean: {roi_mean:.2f}, Sum: {roi_sum:.2f}, "
                f"Max: {roi_max:.2f}, Min: {roi_min:.2f}, Std: {roi_std:.2f}, Shape: {roi_slice.shape}"
            )
    
    def save_roi_position(self):
        """Save ROI position to file"""
        if self.roi is None:
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save ROI Position", "roi_position.json", "JSON (*.json)"
        )
        
        if filename:
            roi_state = self.roi.saveState()
            with open(filename, 'w') as f:
                json.dump(roi_state, f, indent=2)
            self.update_status(f"ROI position saved to {filename}")
    
    def load_roi_position(self):
        """Load ROI position from file"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load ROI Position", "", "JSON (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    roi_state = json.load(f)
                
                if self.roi is None:
                    self.roi_check.setChecked(True)
                
                if self.roi is not None:
                    self.roi.setState(roi_state)
                    self.update_status(f"ROI position loaded from {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load ROI: {str(e)}")
    
    def on_crosshair_toggled(self, state):
        """Handle crosshair toggle"""
        self.crosshair_enabled = (state == QtCore.Qt.Checked)
        
        if self.crosshair_enabled:
            if self.vLine is None:
                self.vLine = pg.InfiniteLine(angle=90, movable=False, pen='g')
                self.hLine = pg.InfiniteLine(angle=0, movable=False, pen='g')
                self.frame_plot.addItem(self.vLine, ignoreBounds=True)
                self.frame_plot.addItem(self.hLine, ignoreBounds=True)
                self.frame_plot.scene().sigMouseMoved.connect(self.on_mouse_moved)
        else:
            if self.vLine is not None:
                self.frame_plot.removeItem(self.vLine)
                self.frame_plot.removeItem(self.hLine)
                self.vLine = None
                self.hLine = None
                self.pixel_info_label.setText("Pixel: -")
    
    def on_mouse_moved(self, pos):
        """Handle mouse movement for crosshair"""
        if self.crosshair_enabled and self.current_frame is not None:
            mousePoint = self.frame_plot.vb.mapSceneToView(pos)
            x, y = int(mousePoint.x()), int(mousePoint.y())
            
            if 0 <= x < self.current_frame.shape[1] and 0 <= y < self.current_frame.shape[0]:
                self.vLine.setPos(mousePoint.x())
                self.hLine.setPos(mousePoint.y())
                value = self.current_frame[y, x]
                self.pixel_info_label.setText(f"Pixel: ({x}, {y}) = {value:.2f}")
    
    def on_metadata_received(self, md_dict):
        """Handle new scan metadata"""
        self.current_metadata = ScanMetadata(**md_dict)
        
        # Initialize STXM data array
        self.stxm_data = np.zeros((self.current_metadata.y_num, self.current_metadata.x_num))
        self.stxm_img.setImage(self.stxm_data.T, autoLevels=True)
        
        # Clear frame storage
        self.all_frames = []
        self.roi_data_history = []
        
        # Setup axes for STXM plot
        self.stxm_img.setRect(QtCore.QRectF(
            self.current_metadata.x_start,
            self.current_metadata.y_start,
            self.current_metadata.x_stop - self.current_metadata.x_start,
            self.current_metadata.y_stop - self.current_metadata.y_start
        ))
        
        self.stxm_plot.setLabel('bottom', 'X Position')
        self.stxm_plot.setLabel('left', 'Y Position')
        
        info = (f"Scan: {self.current_metadata.x_num}x{self.current_metadata.y_num}, "
                f"Exposure: {self.current_metadata.exposure_time_s*1000:.1f}ms, "
                f"Detector: {self.current_metadata.detector_shape}")
        self.update_status(info)
        
    def on_frame_received(self, frame_data, frame_idx):
        """Handle new detector frame"""
        if self.current_metadata is None:
            return
        
        self.frame_count_total += 1
        self.current_frame = frame_data
        
        # Store frame for export
        self.all_frames.append(frame_data.copy())
        
        # Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_update_time
        if dt > 0:
            fps = 1.0 / dt
            self.fps_counter.append(fps)
            avg_fps = np.mean(self.fps_counter)
            self.fps_label.setText(f"FPS: {fps:.1f} (avg: {avg_fps:.1f}) | Frames: {self.frame_count_total}")
        self.last_update_time = current_time
        
        # Downsample for display if needed
        if self.downsample_factor > 1:
            h, w = frame_data.shape
            new_h = h // self.downsample_factor
            new_w = w // self.downsample_factor
            display_frame = frame_data[:new_h*self.downsample_factor, :new_w*self.downsample_factor]
            display_frame = display_frame.reshape(new_h, self.downsample_factor, 
                                                  new_w, self.downsample_factor).mean(axis=(1, 3))
        else:
            display_frame = frame_data
        
        # Update frame buffer for averaging
        self.frame_buffer.append(display_frame)
        
        # Display detector frame (averaged if enabled)
        if self.avg_frames_check.isChecked() and len(self.frame_buffer) > 1:
            avg_frame = np.mean(self.frame_buffer, axis=0)
            self.frame_img.setImage(avg_frame.T, autoLevels=self.autoscale_check.isChecked())
        else:
            self.frame_img.setImage(display_frame.T, autoLevels=self.autoscale_check.isChecked())
        
        # Update ROI info if enabled
        if self.roi_enabled and self.roi is not None:
            self.on_roi_changed()
        
        # Calculate STXM intensity (sum of all detector pixels)
        stxm_intensity = np.sum(frame_data)
        
        # Update STXM image
        total_points = self.current_metadata.x_num * self.current_metadata.y_num
        if frame_idx < total_points:
            # Convert linear index to 2D coordinates
            y_idx = frame_idx // self.current_metadata.x_num
            x_idx = frame_idx % self.current_metadata.x_num
            
            # Debug output
            print(f"Frame {frame_idx}: shape={frame_data.shape}, sum={stxm_intensity:.2f}, pos=({x_idx},{y_idx})")
            
            self.stxm_data[y_idx, x_idx] = stxm_intensity
            
            # Update display
            self.stxm_img.setImage(self.stxm_data.T, autoLevels=self.autoscale_check.isChecked())
    
    def export_stxm_data(self):
        """Export STXM data to file"""
        if self.stxm_data is None:
            QtWidgets.QMessageBox.warning(self, "Export Error", "No STXM data to export")
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export STXM Data", "stxm_data.npz", "NumPy Archive (*.npz);;CSV (*.csv);;Text (*.txt)"
        )
        
        if filename:
            try:
                if filename.endswith('.npz'):
                    np.savez(
                        filename,
                        stxm_data=self.stxm_data,
                        x_start=self.current_metadata.x_start,
                        x_stop=self.current_metadata.x_stop,
                        x_num=self.current_metadata.x_num,
                        y_start=self.current_metadata.y_start,
                        y_stop=self.current_metadata.y_stop,
                        y_num=self.current_metadata.y_num
                    )
                elif filename.endswith('.csv'):
                    np.savetxt(filename, self.stxm_data, delimiter=',')
                else:
                    np.savetxt(filename, self.stxm_data)
                
                self.update_status(f"STXM data exported to {filename}")
                QtWidgets.QMessageBox.information(self, "Export Success", f"STXM data saved to:\n{filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def export_current_frame(self):
        """Export current detector frame"""
        if self.current_frame is None:
            QtWidgets.QMessageBox.warning(self, "Export Error", "No frame to export")
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Current Frame", "detector_frame.npy", "NumPy Array (*.npy);;TIFF (*.tiff);;PNG (*.png)"
        )
        
        if filename:
            try:
                if filename.endswith('.npy'):
                    np.save(filename, self.current_frame)
                elif filename.endswith('.tiff') or filename.endswith('.tif'):
                    from PIL import Image
                    normalized = ((self.current_frame - self.current_frame.min()) / 
                                 (self.current_frame.max() - self.current_frame.min()) * 65535).astype(np.uint16)
                    Image.fromarray(normalized).save(filename)
                elif filename.endswith('.png'):
                    from PIL import Image
                    normalized = ((self.current_frame - self.current_frame.min()) / 
                                 (self.current_frame.max() - self.current_frame.min()) * 255).astype(np.uint8)
                    Image.fromarray(normalized).save(filename)
                
                self.update_status(f"Frame exported to {filename}")
                QtWidgets.QMessageBox.information(self, "Export Success", f"Frame saved to:\n{filename}")
            except ImportError:
                QtWidgets.QMessageBox.warning(self, "Export Warning", 
                    "PIL/Pillow not installed. Only .npy format available.\nInstall with: pip install Pillow")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def export_all_frames(self):
        """Export all collected frames"""
        if not self.all_frames:
            QtWidgets.QMessageBox.warning(self, "Export Error", "No frames to export")
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export All Frames", "all_frames.npz", "NumPy Archive (*.npz)"
        )
        
        if filename:
            try:
                frames_array = np.array(self.all_frames)
                np.savez_compressed(
                    filename,
                    frames=frames_array,
                    metadata={
                        'x_num': self.current_metadata.x_num,
                        'y_num': self.current_metadata.y_num,
                        'exposure_time_s': self.current_metadata.exposure_time_s,
                        'detector_shape': self.current_metadata.detector_shape
                    }
                )
                
                self.update_status(f"All frames ({len(self.all_frames)}) exported to {filename}")
                QtWidgets.QMessageBox.information(
                    self, "Export Success", 
                    f"Saved {len(self.all_frames)} frames to:\n{filename}\n"
                    f"Size: {frames_array.nbytes / 1024 / 1024:.1f} MB"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def export_snapshots(self):
        """Export all snapshots"""
        if not self.snapshots:
            QtWidgets.QMessageBox.warning(self, "Export Error", "No snapshots to export")
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Snapshots", "snapshots.npz", "NumPy Archive (*.npz)"
        )
        
        if filename:
            try:
                snapshot_data = {
                    f'snapshot_{i}_frame': snap['frame'] for i, snap in enumerate(self.snapshots)
                }
                snapshot_data.update({
                    f'snapshot_{i}_stxm': snap['stxm_data'] for i, snap in enumerate(self.snapshots)
                })
                snapshot_data['timestamps'] = [snap['timestamp'] for snap in self.snapshots]
                
                np.savez_compressed(filename, **snapshot_data)
                
                self.update_status(f"Snapshots ({len(self.snapshots)}) exported to {filename}")
                QtWidgets.QMessageBox.information(
                    self, "Export Success", 
                    f"Saved {len(self.snapshots)} snapshots to:\n{filename}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def export_roi_data(self):
        """Export ROI statistics history"""
        if not self.roi_data_history:
            QtWidgets.QMessageBox.warning(self, "Export Error", "No ROI data to export")
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export ROI Data", "roi_statistics.csv", "CSV (*.csv);;JSON (*.json)"
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(self.roi_data_history, f, indent=2)
                else:
                    # Export as CSV
                    import csv
                    with open(filename, 'w', newline='') as f:
                        if self.roi_data_history:
                            writer = csv.DictWriter(f, fieldnames=self.roi_data_history[0].keys())
                            writer.writeheader()
                            writer.writerows(self.roi_data_history)
                
                self.update_status(f"ROI data exported to {filename}")
                QtWidgets.QMessageBox.information(
                    self, "Export Success", 
                    f"Saved {len(self.roi_data_history)} ROI measurements to:\n{filename}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
        
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
        
    def closeEvent(self, event):
        """Handle application close"""
        self.receiver.stop()
        self.receiver.wait()
        event.accept()


def main():
    """Main entry point"""
    app = QtWidgets.QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtCore.Qt.black)
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    app.setPalette(palette)
    
    # Create and show main window
    window = STXMVisualizationApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":

    main()
