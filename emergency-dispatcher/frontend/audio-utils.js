(function (root) {
  function createLinearResampler(sourceRate, targetRate) {
    let samples = new Float32Array(0);
    let nextSourcePosition = 0;

    return {
      process(input) {
        if (!input || !input.length) return new Float32Array(0);
        if (!sourceRate || !targetRate || sourceRate === targetRate) {
          return new Float32Array(input);
        }
        const joined = new Float32Array(samples.length + input.length);
        joined.set(samples);
        joined.set(input, samples.length);
        const ratio = sourceRate / targetRate;
        const output = [];
        while (nextSourcePosition + 1 < joined.length) {
          const leftIndex = Math.floor(nextSourcePosition);
          const fraction = nextSourcePosition - leftIndex;
          const left = joined[leftIndex];
          const right = joined[leftIndex + 1];
          output.push(left + (right - left) * fraction);
          nextSourcePosition += ratio;
        }

        // Retain samples needed by the next interpolation point and its phase.
        const consumed = Math.floor(nextSourcePosition);
        samples = joined.slice(consumed);
        nextSourcePosition -= consumed;
        return Float32Array.from(output);
      },
      reset() {
        samples = new Float32Array(0);
        nextSourcePosition = 0;
      },
    };
  }

  function float32ToPcm16Buffer(float32Samples) {
    const buffer = new ArrayBuffer(float32Samples.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < float32Samples.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, float32Samples[i]));
      view.setInt16(i * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
    }
    return buffer;
  }

  function isCurrentAudioSession(currentSessionId, eventSessionId, currentSocket, eventSocket) {
    return Boolean(currentSessionId && currentSessionId === eventSessionId && currentSocket && currentSocket === eventSocket);
  }

  function isCurrentSessionRequest(requestVersion, currentVersion, requestedSessionId, currentSessionId) {
    return requestVersion === currentVersion && requestedSessionId === currentSessionId;
  }

  function isValidPcm16ChunkByteLength(byteLength) {
    return Number.isSafeInteger(byteLength) && byteLength > 0 && byteLength % 2 === 0;
  }

  function getPlaybackOutcome(textReceived, audioBytesReceived, playbackStarted) {
    if (playbackStarted && audioBytesReceived) return "played";
    if (audioBytesReceived) return "not_played";
    if (textReceived) return "text_without_audio";
    return "no_response";
  }

  function relayAudioMetrics(audioProcessor, onMetrics) {
    if (!audioProcessor || typeof audioProcessor.onMetrics !== "function" || typeof onMetrics !== "function") return false;
    const internalHandler = audioProcessor.onMetrics;
    audioProcessor.onMetrics = (metrics) => {
      internalHandler.call(audioProcessor, metrics);
      onMetrics(metrics);
    };
    return true;
  }

  function getSafePlaybackDiagnostic({
    stage,
    contextState,
    receivedChunks = 0,
    receivedBytes = 0,
    decodedSamples = 0,
    queuedChunks = 0,
    actualAudioPlayed = 0,
    bufferMs = 0,
    sampleRate = 0,
  }) {
    const safeStage = /^[a-z_]{1,32}$/.test(String(stage || "")) ? stage : "unknown";
    const safeState = /^[a-z]{1,16}$/.test(String(contextState || "")) ? contextState : "unknown";
    const safeNumber = (value) => (Number.isFinite(Number(value)) && Number(value) >= 0 ? Number(value) : 0);
    const playedSampleCount = Math.floor(safeNumber(actualAudioPlayed) * safeNumber(sampleRate));
    return `audio_playback_diag stage=${safeStage} context=${safeState} received_chunks=${safeNumber(receivedChunks)} received_bytes=${safeNumber(receivedBytes)} decoded_samples=${safeNumber(decodedSamples)} queued_chunks=${safeNumber(queuedChunks)} played_s=${safeNumber(actualAudioPlayed).toFixed(3)} played_samples=${playedSampleCount} buffer_ms=${safeNumber(bufferMs).toFixed(1)} context_rate_hz=${safeNumber(sampleRate)}`;
  }

  function shouldShowProviderTimeout(textReceived, audioBytesReceived) {
    return !textReceived && !(Number(audioBytesReceived) > 0);
  }

  function getSocketCloseOutcome(code, wasClean) {
    if (code === 1011 || code === 1006 || wasClean === false) {
      return {
        status: "Unavailable",
        kind: "error",
        playbackStatus: `Live provider connection ended unexpectedly (WebSocket ${code}). No dispatcher audio was confirmed.`,
      };
    }
    return {
      status: "Disconnected",
      kind: "idle",
      playbackStatus: "Connection closed",
    };
  }

  const api = {
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
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.DispatcherAudioUtils = api;
})(typeof window !== "undefined" ? window : globalThis);
