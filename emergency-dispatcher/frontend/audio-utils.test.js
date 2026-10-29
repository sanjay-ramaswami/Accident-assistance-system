const test = require("node:test");
const assert = require("node:assert/strict");
const {
  createLinearResampler,
  float32ToPcm16Buffer,
  isCurrentAudioSession,
  isCurrentSessionRequest,
  isValidPcm16ChunkByteLength,
  getPlaybackOutcome,
  relayAudioMetrics,
  getSafePlaybackDiagnostic,
  shouldShowProviderTimeout,
  getSocketCloseOutcome,
} = require("./audio-utils.js");

test("streaming resampling preserves phase across irregular capture buffers", () => {
  const input = Float32Array.from({ length: 12001 }, (_, i) => Math.sin(i / 31));
  const whole = createLinearResampler(44100, 24000).process(input);
  const streaming = createLinearResampler(44100, 24000);
  const parts = [
    streaming.process(input.slice(0, 4096)),
    streaming.process(input.slice(4096, 8192)),
    streaming.process(input.slice(8192)),
  ];
  const streamed = Float32Array.from(parts.flatMap((part) => Array.from(part)));
  assert.equal(streamed.length, whole.length);
  for (let i = 0; i < whole.length; i += 1) {
    assert.ok(Math.abs(streamed[i] - whole[i]) < 1e-6, `sample ${i} differs`);
  }
});

test("PCM conversion clamps samples and writes signed 16-bit little-endian", () => {
  const bytes = new Uint8Array(float32ToPcm16Buffer([-1, 0, 1, 2]));
  assert.deepEqual(Array.from(bytes), [0, 128, 0, 0, 255, 127, 255, 127]);
});

test("late socket events cannot be attributed to a later call session", () => {
  const oldSocket = {};
  const currentSocket = {};
  assert.equal(isCurrentAudioSession("call-2", "call-1", currentSocket, oldSocket), false);
  assert.equal(isCurrentAudioSession("call-2", "call-2", currentSocket, oldSocket), false);
  assert.equal(isCurrentAudioSession("call-2", "call-2", currentSocket, currentSocket), true);
});

test("late first-aid requests cannot overwrite a newer request or session", () => {
  assert.equal(isCurrentSessionRequest(4, 5, "call-1", "call-1"), false);
  assert.equal(isCurrentSessionRequest(5, 5, "call-1", "call-2"), false);
  assert.equal(isCurrentSessionRequest(5, 5, "call-2", "call-2"), true);
});

test("empty and odd-length PCM output frames are rejected", () => {
  assert.equal(isValidPcm16ChunkByteLength(0), false);
  assert.equal(isValidPcm16ChunkByteLength(1), false);
  assert.equal(isValidPcm16ChunkByteLength(2), true);
  assert.equal(isValidPcm16ChunkByteLength(1920), true);
});

test("playback diagnostics distinguish text-only, buffered, and started speech", () => {
  assert.equal(getPlaybackOutcome(false, 0, false), "no_response");
  assert.equal(getPlaybackOutcome(true, 0, false), "text_without_audio");
  assert.equal(getPlaybackOutcome(true, 1920, false), "not_played");
  assert.equal(getPlaybackOutcome(true, 1920, true), "played");
});

test("playback metrics relay actual worklet sample progress to the UI callback", () => {
  const calls = [];
  const processor = {
    onMetrics: (metrics) => calls.push(["internal", metrics.actualAudioPlayed]),
  };
  assert.equal(relayAudioMetrics(processor, (metrics) => calls.push(["ui", metrics.actualAudioPlayed])), true);
  processor.onMetrics({ actualAudioPlayed: 0.125 });
  assert.deepEqual(calls, [["internal", 0.125], ["ui", 0.125]]);
  assert.equal(relayAudioMetrics(null, () => {}), false);
});

test("playback diagnostics report only safe counters and browser audio state", () => {
  const diagnostic = getSafePlaybackDiagnostic({
    stage: "queued",
    contextState: "running",
    receivedChunks: 3,
    receivedBytes: 5760,
    decodedSamples: 2880,
    queuedChunks: 3,
    actualAudioPlayed: 0.08,
    bufferMs: 160,
    sampleRate: 48000,
    callerText: "PRIVATE_TRANSCRIPT",
    apiKey: "PRIVATE_KEY",
  });

  assert.match(diagnostic, /received_bytes=5760/);
  assert.match(diagnostic, /decoded_samples=2880/);
  assert.match(diagnostic, /context_rate_hz=48000/);
  assert.match(diagnostic, /played_samples=3840/);
  assert.match(diagnostic, /decoded_samples=2880/);
  assert.doesNotMatch(diagnostic, /PRIVATE/);
});

test("provider timeout notice only appears when no text or audio arrived", () => {
  assert.equal(shouldShowProviderTimeout(false, 0), true);
  assert.equal(shouldShowProviderTimeout(true, 0), false);
  assert.equal(shouldShowProviderTimeout(false, 1920), false);
});

test("abnormal provider socket closes report unavailable instead of disconnected", () => {
  const providerFailure = getSocketCloseOutcome(1011, false);
  assert.equal(providerFailure.status, "Unavailable");
  assert.equal(providerFailure.kind, "error");
  assert.match(providerFailure.playbackStatus, /WebSocket 1011/);
  assert.match(providerFailure.playbackStatus, /No dispatcher audio was confirmed/);

  assert.equal(getSocketCloseOutcome(1006, false).status, "Unavailable");
  assert.equal(getSocketCloseOutcome(1000, true).status, "Disconnected");
});
