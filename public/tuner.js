/**
 * Fun Guitar - Browser-based Guitar Tuner
 * Uses Web Audio API for real-time pitch detection via autocorrelation
 */

class GuitarTuner {
    constructor() {
        this.audioContext = null;
        this.analyser = null;
        this.mediaStream = null;
        this.animationFrame = null;
        this.isRunning = false;

        // Standard tuning frequencies
        this.standardTuning = [
            { string: 1, note: 'E4', freq: 329.63 },
            { string: 2, note: 'B3', freq: 246.94 },
            { string: 3, note: 'G3', freq: 196.00 },
            { string: 4, note: 'D3', freq: 146.83 },
            { string: 5, note: 'A2', freq: 110.00 },
            { string: 6, note: 'E2', freq: 82.41 },
        ];

        // All chromatic note frequencies (A0 to C8)
        this.noteFrequencies = this._generateNoteFrequencies();

        // Tuner sensitivity
        this.minVolume = 0.01;
        this.bufferSize = 4096;

        // DOM elements
        this.noteDisplay = document.getElementById('tuner-note');
        this.freqDisplay = document.getElementById('tuner-frequency');
        this.needle = document.getElementById('tuner-needle');
        this.centsDisplay = document.getElementById('tuner-cents');
        this.stringInfo = document.getElementById('tuner-string-info');
        this.startBtn = document.getElementById('tuner-start');
        this.stopBtn = document.getElementById('tuner-stop');

        // Reference tone
        this.oscillator = null;
        this.gainNode = null;

        this._bindEvents();
    }

    _generateNoteFrequencies() {
        const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        const freqs = [];
        // Generate from C1 to B7
        for (let octave = 1; octave <= 7; octave++) {
            for (let i = 0; i < notes.length; i++) {
                const noteNum = (octave - 1) * 12 + i - 9; // relative to A4
                const freq = 440 * Math.pow(2, noteNum / 12);
                if (freq >= 60 && freq <= 1200) { // guitar range
                    freqs.push({
                        note: notes[i] + octave,
                        freq: freq
                    });
                }
            }
        }
        return freqs;
    }

    _bindEvents() {
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());

        // String reference buttons - play reference tone
        document.querySelectorAll('.string-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const freq = parseFloat(btn.dataset.freq);
                const note = btn.dataset.note;
                this._playReferenceTone(freq);
                
                // Highlight active
                document.querySelectorAll('.string-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                this.stringInfo.textContent = `Playing reference: ${note} (${freq} Hz) — click again to stop`;
            });
        });
    }

    async start() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                }
            });

            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = this.bufferSize * 2;
            this.analyser.smoothingTimeConstant = 0.8;
            
            source.connect(this.analyser);
            
            this.isRunning = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.stringInfo.textContent = 'Listening... play a string!';
            
            this._detect();
        } catch (err) {
            console.error('Microphone access error:', err);
            this.stringInfo.textContent = 'Error: Could not access microphone. Please allow mic access.';
        }
    }

    stop() {
        this.isRunning = false;
        
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(t => t.stop());
        }
        
        if (this.audioContext) {
            this.audioContext.close();
        }

        this._stopReferenceTone();
        
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.noteDisplay.textContent = '--';
        this.noteDisplay.className = 'tuner-note';
        this.freqDisplay.textContent = '-- Hz';
        this.needle.style.left = '50%';
        this.centsDisplay.textContent = '0 cents';
        this.stringInfo.textContent = 'Tuner stopped';
        document.querySelectorAll('.string-btn').forEach(b => b.classList.remove('active'));
    }

    _detect() {
        if (!this.isRunning) return;

        const buffer = new Float32Array(this.analyser.fftSize);
        this.analyser.getFloatTimeDomainData(buffer);

        // Check volume (RMS)
        let rms = 0;
        for (let i = 0; i < buffer.length; i++) {
            rms += buffer[i] * buffer[i];
        }
        rms = Math.sqrt(rms / buffer.length);

        if (rms > this.minVolume) {
            const freq = this._autoCorrelate(buffer, this.audioContext.sampleRate);
            
            if (freq > 60 && freq < 1200) {
                this._updateDisplay(freq);
            }
        }

        this.animationFrame = requestAnimationFrame(() => this._detect());
    }

    /**
     * Autocorrelation pitch detection algorithm
     * More accurate than simple FFT peak for musical instruments
     */
    _autoCorrelate(buffer, sampleRate) {
        const size = buffer.length;
        
        // Find the first positive-going zero crossing
        let start = 0;
        for (let i = 0; i < size / 2; i++) {
            if (buffer[i] < 0 && buffer[i + 1] >= 0) {
                start = i;
                break;
            }
        }

        // Autocorrelation
        const correlations = new Float32Array(size / 2);
        for (let lag = 0; lag < size / 2; lag++) {
            let sum = 0;
            for (let i = 0; i < size / 2; i++) {
                sum += buffer[i] * buffer[i + lag];
            }
            correlations[lag] = sum;
        }

        // Find the first peak after the initial decline
        let foundPeak = false;
        let peakLag = -1;
        let peakVal = -1;
        
        // Skip initial lag (always high correlation)
        let minLag = Math.floor(sampleRate / 1200); // highest freq we care about
        let maxLag = Math.floor(sampleRate / 60);   // lowest freq we care about

        // Find where correlation first dips
        let lastVal = correlations[minLag];
        for (let lag = minLag + 1; lag < maxLag && lag < correlations.length; lag++) {
            if (correlations[lag] < lastVal) {
                // Declining
                lastVal = correlations[lag];
            } else if (!foundPeak && correlations[lag] > lastVal) {
                // Found first uptick after decline - now find peak
                foundPeak = true;
            }
            
            if (foundPeak && correlations[lag] > peakVal) {
                peakVal = correlations[lag];
                peakLag = lag;
            }
            
            if (foundPeak && correlations[lag] < peakVal * 0.9) {
                // Past the peak
                break;
            }
            
            lastVal = correlations[lag];
        }

        if (peakLag < 0) return -1;

        // Parabolic interpolation for sub-sample accuracy
        let betterLag = peakLag;
        if (peakLag > 0 && peakLag < correlations.length - 1) {
            const y0 = correlations[peakLag - 1];
            const y1 = correlations[peakLag];
            const y2 = correlations[peakLag + 1];
            const shift = (y2 - y0) / (2 * (2 * y1 - y0 - y2));
            if (Math.abs(shift) < 1) {
                betterLag = peakLag + shift;
            }
        }

        return sampleRate / betterLag;
    }

    _updateDisplay(detectedFreq) {
        // Find closest note
        let closestNote = null;
        let minDiff = Infinity;
        
        for (const noteData of this.noteFrequencies) {
            const diff = Math.abs(noteData.freq - detectedFreq);
            if (diff < minDiff) {
                minDiff = diff;
                closestNote = noteData;
            }
        }

        if (!closestNote) return;

        // Calculate cents deviation
        const cents = 1200 * Math.log2(detectedFreq / closestNote.freq);
        const absCents = Math.abs(cents);

        // Update note display
        this.noteDisplay.textContent = closestNote.note;
        this.freqDisplay.textContent = `${detectedFreq.toFixed(1)} Hz`;

        // Update needle position (cents range: -50 to +50 mapped to 0-100%)
        const needlePos = 50 + (cents / 50) * 50;
        this.needle.style.left = `${Math.max(2, Math.min(98, needlePos))}%`;

        // Update cents display
        const sign = cents >= 0 ? '+' : '';
        this.centsDisplay.textContent = `${sign}${cents.toFixed(1)} cents`;

        // Color coding
        this.noteDisplay.className = 'tuner-note';
        if (absCents <= 5) {
            this.noteDisplay.classList.add('in-tune');
            this.stringInfo.textContent = '✓ In tune!';
        } else if (cents > 0) {
            this.noteDisplay.classList.add('sharp');
            this.stringInfo.textContent = `↓ Tune down — ${absCents.toFixed(0)} cents sharp`;
        } else {
            this.noteDisplay.classList.add('flat');
            this.stringInfo.textContent = `↑ Tune up — ${absCents.toFixed(0)} cents flat`;
        }

        // Highlight matching string
        document.querySelectorAll('.string-btn').forEach(btn => {
            const btnFreq = parseFloat(btn.dataset.freq);
            const btnCents = Math.abs(1200 * Math.log2(detectedFreq / btnFreq));
            if (btnCents < 100) { // within 100 cents = 1 semitone
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    _playReferenceTone(freq) {
        // Stop any existing tone
        this._stopReferenceTone();
        
        if (!this.audioContext || this.audioContext.state === 'closed') {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        this.oscillator = this.audioContext.createOscillator();
        this.gainNode = this.audioContext.createGain();
        
        this.oscillator.type = 'sine';
        this.oscillator.frequency.setValueAtTime(freq, this.audioContext.currentTime);
        
        this.gainNode.gain.setValueAtTime(0.15, this.audioContext.currentTime);
        // Fade out after 2 seconds
        this.gainNode.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + 2);
        
        this.oscillator.connect(this.gainNode);
        this.gainNode.connect(this.audioContext.destination);
        
        this.oscillator.start();
        this.oscillator.stop(this.audioContext.currentTime + 2);
        
        this.oscillator.onended = () => {
            this.oscillator = null;
            document.querySelectorAll('.string-btn').forEach(b => b.classList.remove('active'));
            if (this.isRunning) {
                this.stringInfo.textContent = 'Listening... play a string!';
            }
        };
    }

    _stopReferenceTone() {
        if (this.oscillator) {
            try {
                this.oscillator.stop();
            } catch (e) {
                // Already stopped
            }
            this.oscillator = null;
        }
    }
}

// Initialize tuner when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.guitarTuner = new GuitarTuner();
});
