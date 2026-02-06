#!/bin/bash
# Quick Start Guide for Guitar Tab Generation GUI

echo "🎸 Guitar Tab Generator - Quick Start"
echo "======================================"
echo ""

# Check if guitar_detector_gui.py exists
if [ ! -f "guitar_detector_gui.py" ]; then
    echo "❌ Error: guitar_detector_gui.py not found!"
    exit 1
fi

echo "✅ Found guitar_detector_gui.py"
echo ""

# Check Python
if command -v python3 &> /dev/null; then
    echo "✅ Python3 is installed"
    python3 --version
else
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo ""
echo "📋 Required Dependencies:"
echo "   - opencv-python (cv2)"
echo "   - numpy"
echo "   - ultralytics (YOLO)"
echo "   - cvzone"
echo "   - librosa"
echo "   - tkinter (usually pre-installed)"
echo ""

# Check if we should install dependencies
read -p "Would you like to install/update dependencies? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Installing dependencies..."
    pip3 install opencv-python numpy ultralytics cvzone librosa
    echo "✅ Dependencies installed!"
fi

echo ""
echo "🚀 Launching Guitar Detector GUI..."
echo ""
echo "Usage Tips:"
echo "  1. Click 'Browse Video File' to select your guitar video"
echo "  2. Optionally enable audio preprocessing for better results"
echo "  3. Click 'START DETECTION' to begin"
echo "  4. Watch tabs appear in real-time on the right panel"
echo "  5. Click 'Save Tab to File' when done"
echo ""
echo "Press Ctrl+C to exit"
echo ""

# Launch the GUI
python3 guitar_detector_gui.py
