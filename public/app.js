/**
 * Guitar Detection System - Main Application Script
 * Handles file selection, tab controls, and basic UI interactions
 */

document.addEventListener('DOMContentLoaded', function () {

    // === File Selection ===
    var videoInput = document.getElementById('video-input');
    var pathLabel = document.getElementById('path-label');
    var startBtn = document.getElementById('start-btn');

    videoInput.addEventListener('change', function () {
        if (videoInput.files.length > 0) {
            var name = videoInput.files[0].name;
            pathLabel.textContent = name;
            startBtn.disabled = false;
            document.getElementById('status-label').textContent = 'Video loaded - Ready to start';
        }
    });

    // === Start Detection (placeholder for web demo) ===
    startBtn.addEventListener('click', function () {
        document.getElementById('status-label').textContent = 'Detection requires the desktop application (Python + YOLO).';
        var tabDisplay = document.getElementById('tab-display');
        tabDisplay.value = 'Detection requires the desktop application.\n\n' +
            'The full pipeline (YOLOv8 + hand tracking + audio analysis)\n' +
            'runs locally. Clone the repo and run:\n\n' +
            '  python3 guitar_detector_gui.py\n\n' +
            'The browser tuner on the left works here though!\n' +
            'Click "Start Tuner" to try it.';
    });

    // === Tab Controls ===
    var saveTabBtn = document.getElementById('save-tab-btn');
    var clearTabBtn = document.getElementById('clear-tab-btn');
    var tabDisplay = document.getElementById('tab-display');

    clearTabBtn.addEventListener('click', function () {
        tabDisplay.value = 'Tab cleared. Start new detection to generate tabs.';
        saveTabBtn.disabled = true;
        clearTabBtn.disabled = true;
    });

    saveTabBtn.addEventListener('click', function () {
        var content = tabDisplay.value;
        var blob = new Blob([content], { type: 'text/plain' });
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'guitar_tab.txt';
        a.click();
        URL.revokeObjectURL(a.href);
    });
});
