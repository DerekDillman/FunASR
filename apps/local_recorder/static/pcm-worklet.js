// Captures mic audio, resamples it from the AudioContext rate to 16 kHz,
// and posts Int16 PCM chunks (~120 ms) to the main thread.
const TARGET_RATE = 16000;
const CHUNK_SAMPLES = 1920; // 120 ms @ 16 kHz

class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.src = new Float32Array(0); // pending source-rate samples
    this.pos = 0;                   // fractional read position into src
    this.out = new Int16Array(CHUNK_SAMPLES);
    this.outLen = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    // Append new input.
    const merged = new Float32Array(this.src.length + channel.length);
    merged.set(this.src);
    merged.set(channel, this.src.length);
    this.src = merged;

    // Linear-interpolation resample to 16 kHz.
    let pos = this.pos;
    const src = this.src;
    while (Math.floor(pos) + 1 < src.length) {
      const i = Math.floor(pos);
      const frac = pos - i;
      let s = src[i] * (1 - frac) + src[i + 1] * frac;
      s = Math.max(-1, Math.min(1, s));
      this.out[this.outLen++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.outLen === CHUNK_SAMPLES) {
        this.port.postMessage(this.out.buffer.slice(0));
        this.outLen = 0;
      }
      pos += this.ratio;
    }

    // Drop consumed source samples, keep the fractional remainder.
    const consumed = Math.floor(pos);
    this.src = src.subarray(consumed).slice(0);
    this.pos = pos - consumed;
    return true;
  }
}

registerProcessor("pcm-capture", PCMCaptureProcessor);
