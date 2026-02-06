#!/usr/bin/env python3
"""
Audio Preprocessor for Guitar Detection
Uses Demucs or Spleeter to isolate guitar/instrument audio from background noise
"""

import os
import sys
import tempfile
import subprocess
import shutil
from pathlib import Path
import librosa
import soundfile as sf


class AudioPreprocessor:
    """
    Handles audio preprocessing to extract clean instrument audio from music files.
    Supports Demucs (preferred) and Spleeter (fallback).
    """
    
    def __init__(self, method='auto'):
        """
        Initialize the audio preprocessor.
        
        Args:
            method: 'demucs', 'spleeter', or 'auto' (auto-detect available tool)
        """
        self.method = method
        self.available_methods = self._check_available_methods()
        
        if method == 'auto':
            if 'demucs' in self.available_methods:
                self.method = 'demucs'
            elif 'spleeter' in self.available_methods:
                self.method = 'spleeter'
            else:
                self.method = None
        
        if self.method not in self.available_methods:
            print(f"⚠️  Warning: {self.method} not available. Available: {self.available_methods}")
    
    def _check_available_methods(self):
        """Check which audio separation tools are available."""
        available = []
        
        # Check for Demucs
        try:
            result = subprocess.run(['demucs', '--help'], 
                                   capture_output=True, 
                                   timeout=5)
            if result.returncode == 0:
                available.append('demucs')
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Check for Spleeter
        try:
            result = subprocess.run(['spleeter', '--help'], 
                                   capture_output=True, 
                                   timeout=5)
            if result.returncode == 0:
                available.append('spleeter')
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Check for Python imports
        try:
            import demucs
            if 'demucs' not in available:
                available.append('demucs')
        except ImportError:
            pass
        
        try:
            import spleeter
            if 'spleeter' not in available:
                available.append('spleeter')
        except ImportError:
            pass
        
        return available
    
    def is_available(self):
        """Check if any preprocessing method is available."""
        return len(self.available_methods) > 0 and self.method is not None
    
    def preprocess_audio(self, audio_path, output_path=None, progress_callback=None):
        """
        Preprocess audio file to isolate instrument/guitar audio.
        
        Args:
            audio_path: Path to input audio/video file
            output_path: Path to save cleaned audio (optional, will create temp if None)
            progress_callback: Function to call with progress updates
        
        Returns:
            Path to the processed audio file, or None if preprocessing failed
        """
        if not self.is_available():
            if progress_callback:
                progress_callback("⚠️  No audio preprocessing available - using original audio")
            return audio_path
        
        try:
            if self.method == 'demucs':
                return self._preprocess_with_demucs(audio_path, output_path, progress_callback)
            elif self.method == 'spleeter':
                return self._preprocess_with_spleeter(audio_path, output_path, progress_callback)
            else:
                if progress_callback:
                    progress_callback("⚠️  No preprocessing method available")
                return audio_path
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Preprocessing error: {e}")
                progress_callback("⚠️  Using original audio as fallback")
            print(f"Error during preprocessing: {e}")
            print("Using original audio as fallback")
            return audio_path
    
    def _preprocess_with_demucs(self, audio_path, output_path, progress_callback):
        """
        Use Demucs to separate audio sources.
        Demucs is more modern and produces better results.
        """
        if progress_callback:
            progress_callback("🎵 Initializing Demucs audio separation...")
        
        # Create temporary directory for Demucs output
        temp_dir = tempfile.mkdtemp(prefix='demucs_')
        
        try:
            # Try to use Demucs Python API first (better control)
            try:
                import torch
                import torchaudio
                from demucs.pretrained import get_model
                from demucs.apply import apply_model
                
                if progress_callback:
                    progress_callback("🎵 Loading Demucs model...")
                
                # Load the model (htdemucs is the best general model)
                model = get_model('htdemucs')
                model.eval()
                
                if progress_callback:
                    progress_callback("🎵 Loading audio file...")
                
                # Load audio
                wav, sr = torchaudio.load(audio_path)
                
                # Ensure stereo
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)
                
                # Resample if needed (Demucs expects 44.1kHz)
                if sr != model.samplerate:
                    if progress_callback:
                        progress_callback(f"🎵 Resampling from {sr}Hz to {model.samplerate}Hz...")
                    resampler = torchaudio.transforms.Resample(sr, model.samplerate)
                    wav = resampler(wav)
                    sr = model.samplerate
                
                if progress_callback:
                    progress_callback("🎵 Separating audio sources (this may take a while)...")
                
                # Apply model
                with torch.no_grad():
                    sources = apply_model(model, wav.unsqueeze(0), device='cpu')[0]
                
                # Sources: [drums, bass, other, vocals]
                # We want everything except vocals and drums for guitar detection
                # Index 0: drums, 1: bass, 2: other (includes guitar), 3: vocals
                
                if progress_callback:
                    progress_callback("🎵 Extracting instrumental audio...")
                
                # Combine bass + other (instruments including guitar)
                instrumental = sources[1] + sources[2]  # bass + other
                
                # Determine output path
                if output_path is None:
                    output_path = os.path.join(temp_dir, 'cleaned_audio.wav')
                
                # Save the instrumental audio
                torchaudio.save(output_path, instrumental.cpu(), sr)
                
                if progress_callback:
                    progress_callback(f"✅ Audio cleaned with Demucs: {output_path}")
                
                return output_path
                
            except ImportError:
                # Fall back to command-line Demucs
                if progress_callback:
                    progress_callback("🎵 Using Demucs CLI (Python API not available)...")
                
                # Run Demucs command-line with MP3 output (avoids torchcodec issue)
                cmd = [
                    'demucs',
                    '-n', 'htdemucs',  # Use the best model
                    '--two-stems', 'vocals',  # Separate vocals from everything else
                    '--mp3',  # Use MP3 format to avoid torchcodec dependency
                    '--mp3-bitrate', '320',  # High quality
                    '-o', temp_dir,
                    audio_path
                ]
                
                if progress_callback:
                    progress_callback("🎵 Running Demucs separation...")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"Demucs failed: {result.stderr}")
                
                # Find the output file (no_vocals.mp3 or no_vocals.wav)
                audio_name = Path(audio_path).stem
                model_output_dir = os.path.join(temp_dir, 'htdemucs', audio_name)
                
                # Try MP3 first, then WAV
                no_vocals_path = os.path.join(model_output_dir, 'no_vocals.mp3')
                if not os.path.exists(no_vocals_path):
                    no_vocals_path = os.path.join(model_output_dir, 'no_vocals.wav')
                
                if not os.path.exists(no_vocals_path):
                    raise Exception(f"Demucs output not found: {no_vocals_path}")
                
                # Convert to WAV using librosa and soundfile
                if progress_callback:
                    progress_callback("🎵 Converting to WAV format...")
                
                audio_data, sr = librosa.load(no_vocals_path, sr=None, mono=False)
                
                # Determine output path
                if output_path is None:
                    output_path = os.path.join(temp_dir, 'cleaned_audio.wav')
                
                # Save using soundfile (avoids torchcodec dependency)
                sf.write(output_path, audio_data.T if audio_data.ndim > 1 else audio_data, sr)
                
                if progress_callback:
                    progress_callback(f"✅ Audio cleaned with Demucs CLI: {output_path}")
                
                return output_path
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Demucs failed: {e}")
            raise
        
        finally:
            # Clean up temp directory if we created our own output
            if output_path and not output_path.startswith(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
    
    def _preprocess_with_spleeter(self, audio_path, output_path, progress_callback):
        """
        Use Spleeter to separate audio sources.
        Spleeter is faster but may produce slightly lower quality results.
        """
        if progress_callback:
            progress_callback("🎵 Initializing Spleeter audio separation...")
        
        # Create temporary directory for Spleeter output
        temp_dir = tempfile.mkdtemp(prefix='spleeter_')
        
        try:
            # Try to use Spleeter Python API
            try:
                from spleeter.separator import Separator
                
                if progress_callback:
                    progress_callback("🎵 Loading Spleeter model...")
                
                # Use 2stems model (vocals vs accompaniment)
                # This is faster and good enough for our purpose
                separator = Separator('spleeter:2stems')
                
                if progress_callback:
                    progress_callback("🎵 Separating audio sources...")
                
                # Separate the audio
                separator.separate_to_file(audio_path, temp_dir)
                
                # Find the accompaniment file
                audio_name = Path(audio_path).stem
                accompaniment_path = os.path.join(temp_dir, audio_name, 'accompaniment.wav')
                
                if not os.path.exists(accompaniment_path):
                    raise Exception(f"Spleeter output not found: {accompaniment_path}")
                
                # Determine output path
                if output_path is None:
                    output_path = os.path.join(temp_dir, 'cleaned_audio.wav')
                
                # Copy to output path
                shutil.copy(accompaniment_path, output_path)
                
                if progress_callback:
                    progress_callback(f"✅ Audio cleaned with Spleeter: {output_path}")
                
                return output_path
                
            except ImportError:
                # Fall back to command-line Spleeter
                if progress_callback:
                    progress_callback("🎵 Using Spleeter CLI...")
                
                # Run Spleeter command
                cmd = [
                    'spleeter',
                    'separate',
                    '-p', 'spleeter:2stems',
                    '-o', temp_dir,
                    audio_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"Spleeter failed: {result.stderr}")
                
                # Find the output file
                audio_name = Path(audio_path).stem
                accompaniment_path = os.path.join(temp_dir, audio_name, 'accompaniment.wav')
                
                if not os.path.exists(accompaniment_path):
                    raise Exception(f"Spleeter output not found: {accompaniment_path}")
                
                # Determine output path
                if output_path is None:
                    output_path = os.path.join(temp_dir, 'cleaned_audio.wav')
                
                # Copy to output path
                shutil.copy(accompaniment_path, output_path)
                
                if progress_callback:
                    progress_callback(f"✅ Audio cleaned with Spleeter CLI: {output_path}")
                
                return output_path
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Spleeter failed: {e}")
            raise
        
        finally:
            # Clean up temp directory if we created our own output
            if output_path and not output_path.startswith(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
    
    def extract_audio_from_video(self, video_path, output_path=None):
        """
        Extract audio from video file using ffmpeg.
        
        Args:
            video_path: Path to video file
            output_path: Path to save extracted audio (optional)
        
        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            temp_dir = tempfile.mkdtemp(prefix='audio_extract_')
            output_path = os.path.join(temp_dir, 'extracted_audio.wav')
        
        # Extract audio using ffmpeg
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '44100',  # 44.1kHz
            '-ac', '2',  # Stereo
            '-y',  # Overwrite
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Audio extraction failed: {result.stderr}")
        
        return output_path


def demo():
    """Demonstrate the audio preprocessor."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_preprocessor.py <audio_or_video_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    print("=" * 80)
    print("Audio Preprocessor Demo")
    print("=" * 80)
    
    preprocessor = AudioPreprocessor(method='auto')
    
    print(f"\nAvailable methods: {preprocessor.available_methods}")
    print(f"Selected method: {preprocessor.method}")
    
    if not preprocessor.is_available():
        print("\n❌ No preprocessing method available!")
        print("\nTo install Demucs (recommended):")
        print("  pip install demucs")
        print("\nTo install Spleeter:")
        print("  pip install spleeter")
        sys.exit(1)
    
    print(f"\n🎵 Processing: {input_file}")
    
    def progress(msg):
        print(f"  {msg}")
    
    output_file = preprocessor.preprocess_audio(input_file, progress_callback=progress)
    
    print(f"\n✅ Processed audio: {output_file}")
    print("\nYou can now use this cleaned audio for guitar detection!")


if __name__ == "__main__":
    demo()
