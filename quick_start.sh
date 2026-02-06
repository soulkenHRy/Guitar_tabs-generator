#!/bin/bash
# Quick Start Script for Guitar Detection with Audio Preprocessing

echo "=================================="
echo "🎸 Guitar Detection Quick Start"
echo "=================================="
echo ""

# Check Python version
echo "1️⃣  Checking Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi
echo "✅ Python OK"
echo ""

# Check if ffmpeg is installed
echo "2️⃣  Checking ffmpeg..."
ffmpeg -version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ ffmpeg not found."
    echo ""
    echo "Please install ffmpeg:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS:         brew install ffmpeg"
    echo "  Windows:       Download from https://ffmpeg.org"
    exit 1
fi
echo "✅ ffmpeg OK"
echo ""

# Install core dependencies
echo "3️⃣  Installing core dependencies..."
pip3 install -q opencv-python numpy ultralytics cvzone librosa soundfile 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some packages may have failed to install"
else
    echo "✅ Core dependencies installed"
fi
echo ""

# Offer to install audio preprocessing
echo "4️⃣  Audio Preprocessing Setup"
echo ""
echo "Do you want to install audio preprocessing? (Recommended)"
echo "This enables automatic noise and vocal removal for better accuracy."
echo ""
echo "Choose an option:"
echo "  1) Demucs (Best quality, ~300MB, slower: 2-5 min per song)"
echo "  2) Spleeter (Good quality, ~200MB, faster: 30-60 sec per song)"
echo "  3) Both (Recommended)"
echo "  4) Skip (use original audio)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Installing Demucs..."
        pip3 install -q demucs torch torchaudio
        echo "✅ Demucs installed"
        ;;
    2)
        echo ""
        echo "Installing Spleeter..."
        pip3 install -q spleeter
        echo "✅ Spleeter installed"
        ;;
    3)
        echo ""
        echo "Installing both Demucs and Spleeter..."
        pip3 install -q demucs spleeter torch torchaudio
        echo "✅ Both tools installed"
        ;;
    4)
        echo ""
        echo "⚠️  Skipping audio preprocessing installation"
        ;;
    *)
        echo ""
        echo "Invalid choice. Skipping."
        ;;
esac
echo ""

# Run test
echo "5️⃣  Running diagnostic test..."
echo ""
python3 test_preprocessing.py
echo ""

# Show usage options
echo "=================================="
echo "🎉 Setup Complete!"
echo "=================================="
echo ""
echo "📚 Quick Usage Guide:"
echo ""
echo "GUI Mode (Easiest):"
echo "  python3 guitar_detector_gui.py"
echo ""
echo "Command Line (with video file):"
echo "  python3 guitar_complete_detector.py --video your_video.mp4"
echo ""
echo "Command Line (webcam):"
echo "  python3 guitar_complete_detector.py"
echo ""
echo "Save output video:"
echo "  python3 guitar_complete_detector.py --video input.mp4 --output result.mp4"
echo ""
echo "Skip preprocessing (faster):"
echo "  python3 guitar_complete_detector.py --video input.mp4 --no-preprocess"
echo ""
echo "📖 Documentation:"
echo "  - AUDIO_PREPROCESSING_README.md  (Quick start guide)"
echo "  - INSTALL.md                     (Full installation guide)"
echo "  - IMPLEMENTATION_SUMMARY.md      (Technical details)"
echo "  - WORKFLOW_DIAGRAM.md            (Visual diagrams)"
echo ""
echo "🧪 Test preprocessing:"
echo "  python3 test_preprocessing.py"
echo ""
echo "=================================="
