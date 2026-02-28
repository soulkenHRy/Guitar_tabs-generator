/**
 * Guitar Detection System - Browser Guitar Tuner
 * Uses Web Audio API for real-time pitch detection via autocorrelation
 */

var GuitarTuner = (function () {

    var audioContext = null;
    var analyser = null;
    var mediaStream = null;
    var animationFrame = null;
    var isRunning = false;
    var oscillator = null;
    var gainNode = null;

    var minVolume = 0.01;

    // DOM elements
    var noteDisplay = document.getElementById('tuner-note');
    var freqDisplay = document.getElementById('tuner-frequency');
    var needle = document.getElementById('tuner-needle');
    var centsDisplay = document.getElementById('tuner-cents');
    var stringInfo = document.getElementById('tuner-string-info');
    var startBtn = document.getElementById('tuner-start');
    var stopBtn = document.getElementById('tuner-stop');

    // All note frequencies in guitar range
    var noteFrequencies = (function () {
        var notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        var freqs = [];
        for (var octave = 1; octave <= 7; octave++) {
            for (var i = 0; i < notes.length; i++) {
                var noteNum = (octave - 1) * 12 + i - 9;
                var freq = 440 * Math.pow(2, noteNum / 12);
                if (freq >= 60 && freq <= 1200) {
                    freqs.push({ note: notes[i] + octave, freq: freq });
                }
            }
        }
        return freqs;
    })();

    function autoCorrelate(buffer, sampleRate) {
        var size = buffer.length;

        // Find first positive-going zero crossing
        for (var i = 0; i < size / 2; i++) {
            if (buffer[i] < 0 && buffer[i + 1] >= 0) break;
        }

        // Autocorrelation
        var correlations = new Float32Array(size / 2);
        for (var lag = 0; lag < size / 2; lag++) {
            var sum = 0;
            for (var j = 0; j < size / 2; j++) {
                sum += buffer[j] * buffer[j + lag];
            }
            correlations[lag] = sum;
        }

        var minLag = Math.floor(sampleRate / 1200);
        var maxLag = Math.floor(sampleRate / 60);

        var foundPeak = false;
        var peakLag = -1;
        var peakVal = -1;
        var lastVal = correlations[minLag];

        for (var lag = minLag + 1; lag < maxLag && lag < correlations.length; lag++) {
            if (correlations[lag] < lastVal) {
                lastVal = correlations[lag];
            } else if (!foundPeak && correlations[lag] > lastVal) {
                foundPeak = true;
            }

            if (foundPeak && correlations[lag] > peakVal) {
                peakVal = correlations[lag];
                peakLag = lag;
            }

            if (foundPeak && correlations[lag] < peakVal * 0.9) break;

            lastVal = correlations[lag];
        }

        if (peakLag < 0) return -1;

        // Parabolic interpolation
        var betterLag = peakLag;
        if (peakLag > 0 && peakLag < correlations.length - 1) {
            var y0 = correlations[peakLag - 1];
            var y1 = correlations[peakLag];
            var y2 = correlations[peakLag + 1];
            var shift = (y2 - y0) / (2 * (2 * y1 - y0 - y2));
            if (Math.abs(shift) < 1) betterLag = peakLag + shift;
        }

        return sampleRate / betterLag;
    }

    function findClosestNote(freq) {
        var closest = null;
        var minDiff = Infinity;
        for (var i = 0; i < noteFrequencies.length; i++) {
            var diff = Math.abs(noteFrequencies[i].freq - freq);
            if (diff < minDiff) {
                minDiff = diff;
                closest = noteFrequencies[i];
            }
        }
        return closest;
    }

    function updateDisplay(detectedFreq) {
        var closest = findClosestNote(detectedFreq);
        if (!closest) return;

        var cents = 1200 * Math.log2(detectedFreq / closest.freq);
        var absCents = Math.abs(cents);

        noteDisplay.textContent = closest.note;
        freqDisplay.textContent = detectedFreq.toFixed(1) + ' Hz';

        // Needle position
        var needlePos = 50 + (cents / 50) * 50;
        needle.style.left = Math.max(2, Math.min(98, needlePos)) + '%';

        // Cents
        var sign = cents >= 0 ? '+' : '';
        centsDisplay.textContent = sign + cents.toFixed(1) + ' cents';

        // Color coding
        noteDisplay.className = 'tuner-note';
        if (absCents <= 5) {
            noteDisplay.classList.add('in-tune');
            stringInfo.textContent = 'In tune!';
        } else if (cents > 0) {
            noteDisplay.classList.add('sharp');
            stringInfo.textContent = 'Tune down - ' + absCents.toFixed(0) + ' cents sharp';
        } else {
            noteDisplay.classList.add('flat');
            stringInfo.textContent = 'Tune up - ' + absCents.toFixed(0) + ' cents flat';
        }

        // Highlight matching string button
        var buttons = document.querySelectorAll('.string-btn');
        for (var i = 0; i < buttons.length; i++) {
            var btnFreq = parseFloat(buttons[i].dataset.freq);
            var btnCents = Math.abs(1200 * Math.log2(detectedFreq / btnFreq));
            if (btnCents < 100) {
                buttons[i].classList.add('active');
            } else {
                buttons[i].classList.remove('active');
            }
        }
    }

    function detect() {
        if (!isRunning) return;

        var buffer = new Float32Array(analyser.fftSize);
        analyser.getFloatTimeDomainData(buffer);

        var rms = 0;
        for (var i = 0; i < buffer.length; i++) {
            rms += buffer[i] * buffer[i];
        }
        rms = Math.sqrt(rms / buffer.length);

        if (rms > minVolume) {
            var freq = autoCorrelate(buffer, audioContext.sampleRate);
            if (freq > 60 && freq < 1200) {
                updateDisplay(freq);
            }
        }

        animationFrame = requestAnimationFrame(detect);
    }

    function playReferenceTone(freq) {
        stopReferenceTone();

        if (!audioContext || audioContext.state === 'closed') {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        oscillator = audioContext.createOscillator();
        gainNode = audioContext.createGain();

        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(freq, audioContext.currentTime);
        gainNode.gain.setValueAtTime(0.15, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 2);

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.start();
        oscillator.stop(audioContext.currentTime + 2);

        oscillator.onended = function () {
            oscillator = null;
            var buttons = document.querySelectorAll('.string-btn');
            for (var i = 0; i < buttons.length; i++) buttons[i].classList.remove('active');
            if (isRunning) stringInfo.textContent = 'Listening... play a string!';
        };
    }

    function stopReferenceTone() {
        if (oscillator) {
            try { oscillator.stop(); } catch (e) { }
            oscillator = null;
        }
    }

    // Public methods
    return {
        start: function () {
            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();

                navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false
                    }
                }).then(function (stream) {
                    mediaStream = stream;
                    var source = audioContext.createMediaStreamSource(stream);
                    analyser = audioContext.createAnalyser();
                    analyser.fftSize = 8192;
                    analyser.smoothingTimeConstant = 0.8;
                    source.connect(analyser);

                    isRunning = true;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    stringInfo.textContent = 'Listening... play a string!';
                    detect();
                }).catch(function (err) {
                    console.error('Microphone error:', err);
                    stringInfo.textContent = 'Error: Could not access microphone.';
                });
            } catch (err) {
                stringInfo.textContent = 'Error: Web Audio not supported.';
            }
        },

        stop: function () {
            isRunning = false;
            if (animationFrame) cancelAnimationFrame(animationFrame);
            if (mediaStream) mediaStream.getTracks().forEach(function (t) { t.stop(); });
            if (audioContext) audioContext.close();
            stopReferenceTone();

            startBtn.disabled = false;
            stopBtn.disabled = true;
            noteDisplay.textContent = '--';
            noteDisplay.className = 'tuner-note';
            freqDisplay.textContent = '-- Hz';
            needle.style.left = '50%';
            centsDisplay.textContent = '0 cents';
            stringInfo.textContent = 'Tuner stopped';
            var buttons = document.querySelectorAll('.string-btn');
            for (var i = 0; i < buttons.length; i++) buttons[i].classList.remove('active');
        },

        playRef: playReferenceTone
    };

})();

// Bind tuner buttons
document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('tuner-start').addEventListener('click', GuitarTuner.start);
    document.getElementById('tuner-stop').addEventListener('click', GuitarTuner.stop);

    document.querySelectorAll('.string-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
            var freq = parseFloat(btn.dataset.freq);
            var note = btn.dataset.note;
            GuitarTuner.playRef(freq);
            document.querySelectorAll('.string-btn').forEach(function (b) { b.classList.remove('active'); });
            btn.classList.add('active');
            document.getElementById('tuner-string-info').textContent =
                'Playing reference: ' + note + ' (' + freq + ' Hz)';
        });
    });
});
