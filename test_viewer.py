"""
Comprehensive test suite for STXM Visualization Application
Tests all major functionality including UI, data processing, ROI, and export
"""

import pytest
import numpy as np
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from stxm_viewer import STXMVisualizationApp, ScanMetadata, ZMQDataReceiver


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def app(qtbot):
    """Create STXMVisualizationApp instance"""
    widget = STXMVisualizationApp()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def sample_metadata():
    """Create sample scan metadata"""
    return {
        'x_start': 0.0,
        'x_stop': 1.0,
        'x_num': 10,
        'y_start': 0.0,
        'y_stop': 1.0,
        'y_num': 10,
        'exposure_time_s': 0.01,
        'detector_shape': (200, 200)
    }


@pytest.fixture
def sample_frame():
    """Create sample detector frame"""
    return np.random.rand(200, 200).astype(np.float32)


# ============================================================================
# Initial State Tests
# ============================================================================

def test_initial_state(app):
    """Test application initial state"""
    assert app.current_metadata is None
    assert app.stxm_data is None
    assert app.current_frame is None
    assert len(app.all_frames) == 0
    assert app.frame_count_total == 0
    assert app.downsample_factor == 1
    assert app.roi_enabled is False


def test_ui_elements_exist(app):
    """Test that all UI elements are created"""
    assert app.downsample_spin is not None
    assert app.colormap_combo is not None
    assert app.autoscale_check is not None
    assert app.avg_frames_check is not None
    assert app.roi_check is not None
    assert app.pause_btn is not None
    assert app.clear_btn is not None
    assert app.snapshot_btn is not None
    assert app.export_stxm_btn is not None
    assert app.export_frame_btn is not None
    assert app.export_all_btn is not None


def test_initial_colormap(app):
    """Test initial colormap is viridis"""
    assert app.current_colormap == 'viridis'
    assert 'viridis' in app.colormaps
    assert 'grayscale' in app.colormaps


# ============================================================================
# Metadata Handling Tests
# ============================================================================

def test_metadata_received(app, sample_metadata):
    """Test metadata processing"""
    app.on_metadata_received(sample_metadata)
    
    assert app.current_metadata is not None
    assert app.current_metadata.x_num == 10
    assert app.current_metadata.y_num == 10
    assert app.stxm_data is not None
    assert app.stxm_data.shape == (10, 10)
    assert len(app.all_frames) == 0  # Should be cleared


def test_metadata_creates_stxm_array(app, sample_metadata):
    """Test that STXM array is properly initialized"""
    app.on_metadata_received(sample_metadata)
    
    assert app.stxm_data.shape == (sample_metadata['y_num'], sample_metadata['x_num'])
    assert np.all(app.stxm_data == 0)  # Should be zeros initially


def test_multiple_scans_reset_data(app, sample_metadata):
    """Test that new scan resets previous data"""
    # First scan
    app.on_metadata_received(sample_metadata)
    app.stxm_data[5, 5] = 100.0
    app.all_frames.append(np.ones((200, 200)))
    
    # Second scan with different size
    new_metadata = sample_metadata.copy()
    new_metadata['x_num'] = 15
    new_metadata['y_num'] = 15
    app.on_metadata_received(new_metadata)
    
    assert app.stxm_data.shape == (15, 15)
    assert len(app.all_frames) == 0
    assert np.all(app.stxm_data == 0)


# ============================================================================
# Frame Processing Tests
# ============================================================================

def test_frame_received(app, sample_metadata, sample_frame):
    """Test frame reception and processing"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    
    assert app.current_frame is not None
    assert app.frame_count_total == 1
    assert len(app.all_frames) == 1


def test_frame_storage(app, sample_metadata, sample_frame):
    """Test that frames are stored correctly"""
    app.on_metadata_received(sample_metadata)
    
    for i in range(5):
        app.on_frame_received(sample_frame.copy(), i)
    
    assert len(app.all_frames) == 5
    assert app.frame_count_total == 5


def test_stxm_intensity_calculation(app, sample_metadata, sample_frame):
    """Test STXM intensity calculation (sum of pixels)"""
    app.on_metadata_received(sample_metadata)
    
    # Create frame with known values
    test_frame = np.ones((200, 200)) * 2.0
    expected_sum = 200 * 200 * 2.0
    
    app.on_frame_received(test_frame, 0)
    
    # STXM data at position (0,0) should be the sum
    assert app.stxm_data[0, 0] == expected_sum


def test_stxm_position_mapping(app, sample_metadata):
    """Test correct mapping of frames to STXM positions"""
    app.on_metadata_received(sample_metadata)
    
    # Test specific positions
    test_cases = [
        (0, 0, 0),    # First frame -> (0, 0)
        (5, 0, 5),    # Frame 5 -> (0, 5)
        (10, 1, 0),   # Frame 10 -> (1, 0)
        (15, 1, 5),   # Frame 15 -> (1, 5)
    ]
    
    for frame_idx, expected_y, expected_x in test_cases:
        frame = np.ones((200, 200)) * frame_idx
        app.on_frame_received(frame, frame_idx)
        
        # Check that intensity is at correct position
        assert app.stxm_data[expected_y, expected_x] == np.sum(frame)


def test_frame_buffer_size(app, sample_metadata, sample_frame):
    """Test that frame buffer maintains max size"""
    app.on_metadata_received(sample_metadata)
    
    # Add more frames than buffer size (10)
    for i in range(20):
        app.on_frame_received(sample_frame.copy(), i)
    
    # Buffer should only keep last 10
    assert len(app.frame_buffer) == 10


# ============================================================================
# Downsampling Tests
# ============================================================================

def test_downsample_factor_change(app, qtbot):
    """Test downsample factor change"""
    app.downsample_spin.setValue(4)
    assert app.downsample_factor == 4


def test_downsample_applied_to_display(app, sample_metadata):
    """Test that downsampling is applied correctly"""
    app.on_metadata_received(sample_metadata)
    app.downsample_factor = 2
    
    # Create 200x200 frame
    frame = np.random.rand(200, 200).astype(np.float32)
    app.on_frame_received(frame, 0)
    
    # Downsampled frame should be 100x100
    # Note: actual downsampling is internal, just verify no crash
    assert app.current_frame.shape == (200, 200)


# ============================================================================
# Colormap Tests
# ============================================================================

def test_colormap_change_viridis(app, qtbot):
    """Test colormap change to viridis"""
    app.colormap_combo.setCurrentText('viridis')
    assert app.current_colormap == 'viridis'


def test_colormap_change_grayscale(app, qtbot):
    """Test colormap change to grayscale"""
    app.colormap_combo.setCurrentText('grayscale')
    assert app.current_colormap == 'grayscale'


def test_grayscale_colormap_exists(app):
    """Test that grayscale colormap is properly created"""
    assert app.grayscale_cmap is not None
    assert hasattr(app.grayscale_cmap, 'getColors')


# ============================================================================
# ROI Tests
# ============================================================================

def test_roi_toggle_on(app, qtbot, sample_metadata, sample_frame):
    """Test ROI enable"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    
    app.roi_check.setChecked(True)
    
    assert app.roi_enabled is True
    assert app.roi is not None
    assert app.save_roi_btn.isEnabled()
    assert app.export_roi_btn.isEnabled()


def test_roi_toggle_off(app, qtbot, sample_metadata, sample_frame):
    """Test ROI disable"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    
    app.roi_check.setChecked(True)
    app.roi_check.setChecked(False)
    
    assert app.roi_enabled is False
    assert app.roi is None
    assert not app.save_roi_btn.isEnabled()
    assert not app.export_roi_btn.isEnabled()


def test_roi_data_collection(app, sample_metadata, sample_frame):
    """Test that ROI data is collected"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    app.roi_check.setChecked(True)
    
    initial_count = len(app.roi_data_history)
    app.on_roi_changed()
    
    assert len(app.roi_data_history) > initial_count
    assert 'mean' in app.roi_data_history[-1]
    assert 'sum' in app.roi_data_history[-1]
    assert 'max' in app.roi_data_history[-1]
    assert 'min' in app.roi_data_history[-1]
    assert 'std' in app.roi_data_history[-1]


# ============================================================================
# Pause/Resume Tests
# ============================================================================

def test_pause_acquisition(app, qtbot):
    """Test pause functionality"""
    app.pause_btn.click()
    
    assert app.pause_btn.isChecked()
    assert app.receiver.paused is True
    assert "Resume" in app.pause_btn.text()


def test_resume_acquisition(app, qtbot):
    """Test resume functionality"""
    app.pause_btn.click()  # Pause
    app.pause_btn.click()  # Resume
    
    assert not app.pause_btn.isChecked()
    assert app.receiver.paused is False
    assert "Pause" in app.pause_btn.text()


# ============================================================================
# Snapshot Tests
# ============================================================================

def test_snapshot_capture(app, sample_metadata, sample_frame):
    """Test snapshot capture"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    
    initial_count = len(app.snapshots)
    app.take_snapshot()
    
    assert len(app.snapshots) == initial_count + 1
    assert 'timestamp' in app.snapshots[-1]
    assert 'frame' in app.snapshots[-1]
    assert 'stxm_data' in app.snapshots[-1]
    assert 'metadata' in app.snapshots[-1]


def test_snapshot_without_data(app, qtbot):
    """Test snapshot when no data available"""
    with patch.object(app, 'update_status') as mock_status:
        app.take_snapshot()
        # Should not add snapshot
        assert len(app.snapshots) == 0


def test_multiple_snapshots(app, sample_metadata, sample_frame):
    """Test multiple snapshots"""
    app.on_metadata_received(sample_metadata)
    
    for i in range(5):
        app.on_frame_received(sample_frame, i)
        app.take_snapshot()
    
    assert len(app.snapshots) == 5


# ============================================================================
# Clear Views Tests
# ============================================================================

def test_clear_views(app, qtbot, sample_metadata, sample_frame):
    """Test clear views functionality"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    app.all_frames.append(sample_frame)
    
    # Mock the message box to auto-accept
    with patch('PyQt5.QtWidgets.QMessageBox.question', return_value=QtWidgets.QMessageBox.Yes):
        app.clear_views()
    
    assert app.stxm_data is None
    assert app.current_frame is None
    assert len(app.all_frames) == 0
    assert app.frame_count_total == 0


# ============================================================================
# Export Tests
# ============================================================================

def test_export_stxm_npz(app, sample_metadata, sample_frame):
    """Test STXM export to NPZ"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName', return_value=(tmp_path, '')):
            with patch('PyQt5.QtWidgets.QMessageBox.information'):
                app.export_stxm_data()
        
        # Verify file was created and contains data
        data = np.load(tmp_path)
        assert 'stxm_data' in data
        assert 'x_num' in data
        assert data['x_num'] == 10
        data.close()  # Close the file before deletion
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


def test_export_frame_npy(app, sample_metadata, sample_frame):
    """Test frame export to NPY"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName', return_value=(tmp_path, '')):
            with patch('PyQt5.QtWidgets.QMessageBox.information'):
                app.export_current_frame()
        
        # Verify file was created
        loaded_frame = np.load(tmp_path)
        assert loaded_frame.shape == sample_frame.shape
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_export_all_frames(app, sample_metadata, sample_frame):
    """Test export all frames"""
    app.on_metadata_received(sample_metadata)
    
    for i in range(5):
        app.on_frame_received(sample_frame.copy(), i)
    
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName', return_value=(tmp_path, '')):
            with patch('PyQt5.QtWidgets.QMessageBox.information'):
                app.export_all_frames()
        
        data = np.load(tmp_path, allow_pickle=True)
        assert 'frames' in data
        assert data['frames'].shape[0] == 5
        data.close()  # Close the file before deletion
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


def test_export_snapshots(app, sample_metadata, sample_frame):
    """Test export snapshots"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    app.take_snapshot()
    app.take_snapshot()
    
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName', return_value=(tmp_path, '')):
            with patch('PyQt5.QtWidgets.QMessageBox.information'):
                app.export_snapshots()
        
        data = np.load(tmp_path, allow_pickle=True)
        assert 'snapshot_0_frame' in data
        assert 'snapshot_0_stxm' in data
        assert 'timestamps' in data
        data.close()  # Close the file before deletion
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


def test_export_roi_data_csv(app, sample_metadata, sample_frame):
    """Test ROI data export to CSV"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    app.roi_check.setChecked(True)
    app.on_roi_changed()
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName', return_value=(tmp_path, '')):
            with patch('PyQt5.QtWidgets.QMessageBox.information'):
                app.export_roi_data()
        
        # Verify file exists
        assert Path(tmp_path).exists()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_save_load_roi_position(app, sample_metadata, sample_frame):
    """Test save and load ROI position"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    app.roi_check.setChecked(True)
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save ROI
        with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName', return_value=(tmp_path, '')):
            app.save_roi_position()
        
        # Verify file exists
        assert Path(tmp_path).exists()
        
        # Load ROI
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(tmp_path, '')):
            app.load_roi_position()
        
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ============================================================================
# Crosshair Tests
# ============================================================================

def test_crosshair_toggle(app, sample_metadata, sample_frame):
    """Test crosshair enable/disable"""
    app.on_metadata_received(sample_metadata)
    app.on_frame_received(sample_frame, 0)
    
    app.crosshair_check.setChecked(True)
    assert app.crosshair_enabled is True
    assert app.vLine is not None
    assert app.hLine is not None
    
    app.crosshair_check.setChecked(False)
    assert app.crosshair_enabled is False
    assert app.vLine is None
    assert app.hLine is None


# ============================================================================
# FPS Calculation Tests
# ============================================================================

def test_fps_calculation(app, sample_metadata, sample_frame):
    """Test FPS calculation"""
    app.on_metadata_received(sample_metadata)
    
    import time
    for i in range(10):
        app.on_frame_received(sample_frame, i)
        time.sleep(0.01)  # Small delay
    
    assert len(app.fps_counter) > 0
    assert all(fps > 0 for fps in app.fps_counter)


# ============================================================================
# ZMQDataReceiver Tests
# ============================================================================

def test_zmq_receiver_initialization():
    """Test ZMQ receiver initialization"""
    receiver = ZMQDataReceiver()
    assert receiver.running is True
    assert receiver.paused is False
    assert receiver.md_port == 50001
    assert receiver.data_port == 50002


def test_zmq_receiver_pause_resume():
    """Test ZMQ receiver pause/resume"""
    receiver = ZMQDataReceiver()
    
    receiver.pause()
    assert receiver.paused is True
    
    receiver.resume()
    assert receiver.paused is False


def test_zmq_receiver_stop():
    """Test ZMQ receiver stop"""
    receiver = ZMQDataReceiver()
    receiver.stop()
    assert receiver.running is False


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

def test_export_without_data(app, qtbot):
    """Test export operations without data"""
    # These should show warnings, not crash
    with patch('PyQt5.QtWidgets.QMessageBox.warning'):
        app.export_stxm_data()
        app.export_current_frame()
        app.export_all_frames()
        app.export_snapshots()


def test_frame_received_without_metadata(app, sample_frame):
    """Test frame reception without metadata"""
    # Should not crash
    app.on_frame_received(sample_frame, 0)
    # Frame should not be processed
    assert app.current_frame is None


def test_large_frame_handling(app, sample_metadata):
    """Test handling of large frames (4000x4000)"""
    sample_metadata['detector_shape'] = (4000, 4000)
    app.on_metadata_received(sample_metadata)
    
    large_frame = np.random.rand(4000, 4000).astype(np.float32)
    app.on_frame_received(large_frame, 0)
    
    assert app.current_frame.shape == (4000, 4000)
    assert len(app.all_frames) == 1


def test_different_scan_sizes(app):
    """Test handling of different scan sizes"""
    test_sizes = [(5, 5), (10, 20), (100, 50), (3, 7)]
    
    for x_num, y_num in test_sizes:
        md = {
            'x_start': 0.0, 'x_stop': 1.0, 'x_num': x_num,
            'y_start': 0.0, 'y_stop': 1.0, 'y_num': y_num,
            'exposure_time_s': 0.01, 'detector_shape': (200, 200)
        }
        app.on_metadata_received(md)
        assert app.stxm_data.shape == (y_num, x_num)


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_scan_workflow(app, sample_metadata):
    """Test complete scan workflow"""
    # Setup scan
    app.on_metadata_received(sample_metadata)
    
    # Process all frames
    total_frames = sample_metadata['x_num'] * sample_metadata['y_num']
    for i in range(total_frames):
        frame = np.random.rand(200, 200).astype(np.float32)
        app.on_frame_received(frame, i)
    
    # Verify all data collected
    assert len(app.all_frames) == total_frames
    assert app.frame_count_total == total_frames
    assert np.all(app.stxm_data != 0)  # All positions filled


def test_workflow_with_pause(app, sample_metadata, sample_frame):
    """Test workflow with pause/resume"""
    app.on_metadata_received(sample_metadata)
    
    # Process some frames
    for i in range(5):
        app.on_frame_received(sample_frame, i)
    
    # Pause
    app.pause_btn.click()
    assert app.receiver.paused is True
    
    # Resume
    app.pause_btn.click()
    assert app.receiver.paused is False
    
    # Process more frames
    for i in range(5, 10):
        app.on_frame_received(sample_frame, i)
    
    assert len(app.all_frames) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])