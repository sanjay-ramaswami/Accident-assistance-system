(function () {
  const statusEl = document.getElementById("status");
  const bufferMetricEl = document.getElementById("buffer-metric");
  const sessionMetricEl = document.getElementById("session-metric");
  const dispatchMetricEl = document.getElementById("dispatch-metric");
  const playbackStatusEl = document.getElementById("playback-status");
  const callerStreamEl = document.getElementById("caller-stream");
  const agentStreamEl = document.getElementById("agent-stream");
  const providerTimeoutEl = document.getElementById("provider-timeout-status");
  const cleanedStreamEl = document.getElementById("cleaned-stream");
  const eventLogEl = document.getElementById("event-log");
  const snapshotGridEl = document.getElementById("snapshot-grid");
  const labelChipsEl = document.getElementById("label-chips");
  const severityHeroEl = document.getElementById("severity-hero");
  const severityScoreEl = document.getElementById("severity-score");
  const severitySubtextEl = document.getElementById("severity-subtext");
  const operatorReportEl = document.getElementById("operator-report");
  const takeoverBtn = document.getElementById("takeover-btn");
  const triagePanelEl = document.getElementById("triage-panel");
  const ticketPanelEl = document.getElementById("ticket-panel");
  const toolChainEl = document.getElementById("tool-chain");
  const toolLogEl = document.getElementById("tool-log");
  const memoryPanelEl = document.getElementById("memory-panel");
  const mapStatusEl = document.getElementById("map-status");
  const sessionReportStatusEl = document.getElementById("session-report-status");
  const retrySessionReportBtn = document.getElementById("retry-session-report");
  const emergencyAssessmentPanelEl = document.getElementById("emergency-assessment-panel");
  const servicePanelEl = document.getElementById("service-panel");
  const connectBtn = document.getElementById("connect-btn");
  const disconnectBtn = document.getElementById("disconnect-btn");
  const unlockAudioBtn = document.getElementById("unlock-audio-btn");
  const clearBtn = document.getElementById("clear-btn");
  const dispatchModeSelect = document.getElementById("dispatch-mode-select");
  const voiceProviderSelect = document.getElementById("voice-provider-select");
  const triageDemoForm = document.getElementById("triage-demo-form");
  const triageDemoText = document.getElementById("triage-demo-text");
  const triageDemoEngine = document.getElementById("triage-demo-engine");
  const triageDemoSubmit = document.getElementById("triage-demo-submit");
  const triageDemoStatus = document.getElementById("triage-demo-status");
  const triageDemoResult = document.getElementById("triage-demo-result");
  const complaintCodeForm = document.getElementById("complaint-code-form");
  const complaintCodeInput = document.getElementById("complaint-code-input");
  const complaintCodeSubmit = document.getElementById("complaint-code-submit");
  const complaintCodeStatus = document.getElementById("complaint-code-status");
  const complaintCodeResult = document.getElementById("complaint-code-result");
  const liveMlStatus = document.getElementById("live-ml-status");
  const liveMlResult = document.getElementById("live-ml-result");
  const firstAidForm = document.getElementById("first-aid-form");
  const firstAidText = document.getElementById("first-aid-text");
  const firstAidSubmit = document.getElementById("first-aid-submit");
  const firstAidStatus = document.getElementById("first-aid-status");
  const firstAidResult = document.getElementById("first-aid-result");
  const incidentDemoScenario = document.getElementById("incident-demo-scenario");
  const incidentDemoLocation = document.getElementById("incident-demo-location");
  const incidentDemoGeocode = document.getElementById("incident-demo-geocode");
  const incidentDemoFallback = document.getElementById("incident-demo-fallback");
  const mapCorrectionToggle = document.getElementById("map-correction-toggle");
  const incidentDemoRun = document.getElementById("incident-demo-run");
  const incidentDemoReset = document.getElementById("incident-demo-reset");
  const incidentDemoStatus = document.getElementById("incident-demo-status");

  let ws = null;
  let player = null;
  let sessionId = null;
  let sessionStartedAt = null;
  let sessionReportRevision = 0;
  let sessionReportSyncPromise = null;
  let queuedSessionReportStatus = null;
  let sessionReportFailureCount = 0;
  let sessionReportPaused = false;
  let finalSessionReportStatus = null;
  let emergencyAssessmentStatus = "Awaiting additional information";
  let currentLiveMlUpdate = { status: "awaiting", reason: "Complaint-code classification requires an explicitly stated supported code." };
  let reportSent = false;
  let userTranscriptChunks = [];
  let agentTranscriptChunks = [];
  let userTranscripts = [];
  let agentTranscripts = [];
  let pendingUserChunks = [];
  let pendingAgentChunks = [];
  let eventEntries = [];
  let maxBufferMs = 0;
  let currentBufferMs = 0;
  let sessionPollHandle = null;
  let syncHandle = null;
  let demoProgressHandle = null;
  let latestSession = null;
  let sessionDetailRequestVersion = 0;
  let firstAidRequestVersion = 0;
  let firstAidDebounceHandle = null;
  let lastFirstAidText = "";
  let map = null;
  let mapMarker = null;
  let demoMapLayer = null;
  let demoAmbulanceMarker = null;
  let demoMapLegend = null;
  let demoRouteLine = null;
  let mapCorrectionMode = false;
  let demoMapSessionId = null;
  let lastMapAnimationKey = null;
  let mapSessionId = null;
  let lastValidMapLocation = null;
  let interruptCount = 0;
  let lastAdaptiveAt = 0;
  let recoveryModeArmed = false;
  let assistantTurnActive = false;
  let assistantTextReceivedThisTurn = false;
  let assistantAudioBytesThisTurn = 0;
  let assistantPlaybackStartedThisTurn = false;
  let playbackWatchdogHandle = null;
  let playbackQueueDiagnostics = { receivedChunks: 0, receivedBytes: 0, decodedSamples: 0, queuedChunks: 0 };
  let responseTimeoutHandle = null;
  let ttsAudioChunksReceived = 0;
  let ttsAudioBytesReceived = 0;
  let audioPlayedSeconds = 0;
  let callStartedPerf = null;
  const voiceLatencyStages = new Set();
  let finalizedCallerTurnIndex = 0;
  let finalizedCallerTurnPerf = null;
  const callerTurnLatencyStages = new Set();
  const liveMlClassifier = window.DispatcherLiveMlUtils?.createLiveComplaintClassifier({
    fetchImpl: (...args) => fetch(...args),
    onUpdate: renderLiveMlUpdate,
  });
  let pendingTriageUpdate = null;
  let pendingLedgerUpdate = null;
  let pendingDispatchUpdate = null;
  let triageSyncTimeout = null;
  let userFinalizeHandle = null;
  let latestTriageContext = null;
  let latestLedgerContext = null;
  let latestDispatchContext = null;
  let lastTriageVersionSent = 0;
  let lastLedgerVersionSent = 0;
  let lastDispatchVersionSent = 0;
  let lastRescueIndexSeen = -1;
  let lastActiveTranscriptProvider = "gradium";
  let humanTakeoverEngaged = false;
  let stickySummaryText = null;
  let pendingHandoffResumePrompt = null;
  let currentAudioConfig = {
    pcm: false,
    pcm_input: false,
    sample_rate: 48000,
    channels: 1,
  };
  const INDIA_MAP_CENTER = [22.5, 79.0];
  const INDIA_MAP_ZOOM = 5;
  let pcmCaptureSource = null;
  let pcmCaptureNode = null;
  let pcmCaptureSink = null;
  let pcmCapturePending = [];
  let pcmResampler = null;
  let pcmDiagnostics = null;
  let pcmInputRate = 0;
  let pcmTargetRate = 0;
  const SHORT_FINALIZE_DELAY_MS = 180;
  const DEFAULT_FINALIZE_DELAY_MS = 450;
  const DISPATCH_MODE_STORAGE_KEY = "dispatchMode";
  const HANDOFF_RESUME_PROMPT = "What is your status now?";

  function normalizeDispatchMode(value) {
    return String(value || "slm").trim().toLowerCase() === "llm_only" ? "llm_only" : "slm";
  }

  function dispatchModeLabel(value) {
    return normalizeDispatchMode(value) === "llm_only" ? "LLM only" : "SLM + LLM";
  }

  async function loadIncidentDemoScenarios() {
    if (!incidentDemoScenario) return;
    try {
      const response = await fetch("/api/demo/scenarios");
      if (!response.ok) throw new Error(`Scenario list unavailable (${response.status})`);
      const scenarios = await response.json();
      incidentDemoScenario.innerHTML = scenarios.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.title)}</option>`).join("");
    } catch (error) {
      incidentDemoStatus.textContent = error.message || "Scenario list unavailable.";
      incidentDemoRun.disabled = true;
    }
  }

  function stopDemoProgressLoop() {
    if (demoProgressHandle) clearInterval(demoProgressHandle);
    demoProgressHandle = null;
  }

  function startDemoProgressLoop() {
    stopDemoProgressLoop();
    demoProgressHandle = setInterval(() => {
      refreshSessionDetail().then(() => {
        if (latestSession?.demo_simulation?.stage === "COMPLETED") stopDemoProgressLoop();
      }).catch((error) => {
        incidentDemoStatus.textContent = `Demo progress refresh failed: ${error.message || "session state unavailable"}`;
      });
    }, 1000);
  }

  async function resetIncidentDemo() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      incidentDemoStatus.textContent = "Stop the live call before resetting the demo view.";
      return;
    }
    stopDemoProgressLoop();
    if (sessionId && latestSession?.demo_simulation?.simulation_only) {
      try {
        const response = await fetch(`/api/demo/reset/${encodeURIComponent(sessionId)}`, { method: "POST" });
        if (!response.ok) throw new Error(`Demo reset failed (${response.status}).`);
      } catch (error) {
        incidentDemoStatus.textContent = error.message || "Demo reset failed; current simulation remains recorded.";
        return;
      }
    }
    stopBackgroundLoops();
    clearTranscriptBuffers();
    userTranscripts = [];
    agentTranscripts = [];
    eventEntries = [];
    latestSession = null;
    sessionId = null;
    sessionStartedAt = null;
    reportSent = false;
    currentLiveMlUpdate = { status: "awaiting", reason: "Complaint-code classification requires an explicitly stated supported code." };
    clearTranscriptBuffers();
    renderDialogueStreams();
    statusEl.textContent = "Idle";
    statusEl.className = "status idle";
    setPlaybackStatus("Waiting for provider connection");
    eventLogEl.innerHTML = "";
    resetDashboard();
    incidentDemoStatus.textContent = "Demo reset. No services contacted; run a scenario to start a fresh deterministic simulation.";
  }

  async function runIncidentDemoScenario() {
    if (!incidentDemoScenario || !incidentDemoRun) return;
    if (ws && ws.readyState === WebSocket.OPEN) {
      incidentDemoStatus.textContent = "Stop the live call before switching to a simulated incident.";
      return;
    }
    clearProviderTimeout();
    stopDemoProgressLoop();
    liveMlClassifier?.reset();
    renderLiveMlUpdate({ status: "awaiting", reason: "No supported complaint code was supplied in this synthetic transcript." });
    emergencyAssessmentStatus = "Updating assessment";
    incidentDemoRun.disabled = true;
    incidentDemoStatus.textContent = "Running synthetic scenario through local session rules…";
    try {
      const response = await fetch("/api/demo/incident", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario_id: incidentDemoScenario.value,
          location_text: incidentDemoLocation?.value.trim() || null,
          resolve_scenario_location: Boolean(incidentDemoGeocode?.checked),
          use_demo_fallback: Boolean(incidentDemoFallback?.checked),
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `Scenario failed (${response.status})`);
      sessionId = payload.session_id;
      clearMapState();
      sessionStartedAt = payload.incident_assessment?.generated_at || new Date().toISOString();
      sessionReportRevision = 1;
      reportSent = true;
      statusEl.textContent = "Simulated demo";
      statusEl.className = "status idle";
      sessionMetricEl.textContent = `SIMULATED · ${sessionId}`;
      dispatchMetricEl.textContent = "Simulation only";
      setPlaybackStatus("No microphone or audio stream used");
      incidentDemoStatus.textContent = `${payload.scenario.title} · ${payload.notice}`;
      await refreshSessionDetail();
      startDemoProgressLoop();
    } catch (error) {
      incidentDemoStatus.textContent = error.message || "Scenario could not be run.";
    } finally {
      incidentDemoRun.disabled = false;
    }
  }

  if (incidentDemoReset) incidentDemoReset.addEventListener("click", resetIncidentDemo);

  if (mapCorrectionToggle) mapCorrectionToggle.addEventListener("click", () => {
    mapCorrectionMode = !mapCorrectionMode;
    mapCorrectionToggle.textContent = mapCorrectionMode ? "Click map to set incident point · Cancel" : "Select incident point on map";
    mapStatusEl.textContent = mapCorrectionMode
      ? "Select the incident point. This is an operator-selected simulation point, not caller GPS."
      : mapStatusEl.textContent;
  });

  function recordVoiceLatency(stage) {
    if (callStartedPerf === null || voiceLatencyStages.has(stage)) return;
    voiceLatencyStages.add(stage);
    const elapsedMs = Math.max(0, performance.now() - callStartedPerf);
    logEvent(`voice_latency stage=${stage} elapsed_ms=${elapsedMs.toFixed(1)}`);
  }

  function recordCallerTurnLatency(stage) {
    if (finalizedCallerTurnPerf === null) return;
    const key = `${finalizedCallerTurnIndex}:${stage}`;
    if (callerTurnLatencyStages.has(key)) return;
    callerTurnLatencyStages.add(key);
    const elapsedMs = Math.max(0, performance.now() - finalizedCallerTurnPerf);
    logEvent(`voice_latency scope=caller_turn turn=${finalizedCallerTurnIndex} stage=${stage} elapsed_ms=${elapsedMs.toFixed(1)}`);
  }

  function renderLiveMlUpdate(update) {
    currentLiveMlUpdate = update || { status: "awaiting" };
    if (!liveMlStatus || !liveMlResult) {
      renderEmergencyAssessment(latestSession || {});
      return;
    }
    liveMlResult.hidden = true;
    if (update.status === "pending") {
      liveMlStatus.textContent = `Pending · checking explicitly stated code ${update.code}`;
      renderEmergencyAssessment(latestSession || {});
      return;
    }
    if (update.status === "awaiting") {
      liveMlStatus.textContent = update.reason ? `Not applicable — complaint code unavailable · ${update.reason}` : "Not applicable — complaint code unavailable";
      renderEmergencyAssessment(latestSession || {});
      return;
    }
    if (update.status === "error") {
      liveMlStatus.textContent = `Error · ${update.message}`;
      renderEmergencyAssessment(latestSession || {});
      return;
    }
    if (update.status === "unavailable") {
      liveMlStatus.textContent = `Unavailable · ${update.message}`;
      renderEmergencyAssessment(latestSession || {});
      return;
    }
    const payload = update.payload;
    const score = Number(payload.uncalibrated_model_score);
    liveMlStatus.textContent = "Available · human review required";
    liveMlResult.textContent = [
      `Complaint text (caller transcript, verbatim): ${update.text}`,
      `Verified training code (caller explicitly stated): ${update.code}`,
      `Predicted broad clinical-functional category: ${payload.prediction.chapter}`,
      `Training coverage: ${payload.training_coverage}`,
      `Uncalibrated model score (not confidence): ${Number.isFinite(score) ? score.toFixed(4) : "Not provided"}`,
      `Timestamp: ${update.timestamp}`,
      "Human review: required",
      "This model classifies complaint codes only. It does not diagnose, predict severity, or authorize dispatch.",
    ].join("\n");
    liveMlResult.hidden = false;
    renderEmergencyAssessment(latestSession || {});
  }

  function getSelectedDispatchMode() {
    return normalizeDispatchMode(dispatchModeSelect?.value);
  }

  function setStatus(text, kind) {
    statusEl.textContent = text;
    statusEl.className = `status ${kind}`;
  }

  function setPlaybackStatus(text, kind = "subtle") {
    playbackStatusEl.textContent = text;
    playbackStatusEl.className = `metric ${kind}`;
  }

  function clearProviderTimeout() {
    if (providerTimeoutEl) providerTimeoutEl.hidden = true;
  }

  function showProviderTimeout(playerSessionId) {
    if (sessionId !== playerSessionId || !providerTimeoutEl) return;
    if (!window.DispatcherAudioUtils.shouldShowProviderTimeout(assistantTextReceivedThisTurn, assistantAudioBytesThisTurn)) return;
    if (providerTimeoutEl) providerTimeoutEl.hidden = false;
    setPlaybackStatus("No live provider response was confirmed; no simulated reply was inserted", "error");
    logEvent(`provider response timeout turn_active=${assistantTurnActive}; no client-generated reply`, "error");
  }

  function scheduleProviderTimeout(playerSessionId, restart = false) {
    if (responseTimeoutHandle && !restart) return;
    if (responseTimeoutHandle) clearTimeout(responseTimeoutHandle);
    responseTimeoutHandle = setTimeout(() => {
      responseTimeoutHandle = null;
      showProviderTimeout(playerSessionId);
    }, 15000);
  }

  function updateAudioUnlockState(audioPlayer = player) {
    const context = audioPlayer?.audioProcessor?.audioContext;
    const suspended = Boolean(sessionId && context && context.state !== "running");
    unlockAudioBtn.hidden = !suspended;
    if (suspended) setPlaybackStatus("Browser audio is suspended; enable playback", "error");
    return suspended;
  }

  unlockAudioBtn.addEventListener("click", async () => {
    const requestedSession = sessionId;
    const context = player?.audioProcessor?.audioContext;
    if (!context) {
      setPlaybackStatus("Audio output is unavailable", "error");
      return;
    }
    try {
      await context.resume();
      if (sessionId !== requestedSession) return;
      if (context.state === "running") {
        unlockAudioBtn.hidden = true;
        setPlaybackStatus("Playback enabled; waiting for dispatcher audio");
      } else {
        setPlaybackStatus("Browser did not resume audio; check the output device", "error");
      }
    } catch (_) {
      if (sessionId === requestedSession) {
        setPlaybackStatus("Browser could not resume audio; check the output device", "error");
      }
    }
  });

  function setSessionReportStatus(text) {
    if (sessionReportStatusEl) sessionReportStatusEl.textContent = text;
  }

  function now() {
    return new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function titleCase(value) {
    return String(value || "")
      .replaceAll("_", " ")
      .toLowerCase()
      .replace(/\b\w/g, (match) => match.toUpperCase());
  }

  function latestTranscriptText(items, count = 3) {
    return (items || [])
      .slice(-count)
      .map((item) => item.text)
      .filter(Boolean)
      .join(" ")
      .trim();
  }

  function humanizeField(field) {
    const aliases = {
      victim_breathing: "Breathing status",
      victim_bleeding: "Bleeding status",
      victim_conscious: "Consciousness",
      victim_count: "Victim count",
      caller_safe: "Caller safety",
      issue_type: "Incident type",
      priority: "Priority",
      location: "Location",
    };
    return aliases[field] || titleCase(field);
  }

  function severityBand(score) {
    if (score >= 86) return "red";
    if (score >= 61) return "orange";
    if (score >= 31) return "amber";
    return "green";
  }

  function severityHeadline(score) {
    if (score >= 86) return "Immediate Human Review Recommended";
    if (score >= 61) return "Human Review Requested";
    if (score >= 31) return "Human Review Recommended";
    return "Autonomous Handling";
  }

  function appendEntry(target, klass, who, text) {
    const entry = document.createElement("article");
    entry.className = `entry ${klass}`;
    entry.innerHTML = `
      <div class="meta">
        <span class="who">${escapeHtml(who)}</span>
        <span>${now()}</span>
      </div>
      <div class="body"></div>
    `;
    entry.querySelector(".body").textContent = text;
    target.appendChild(entry);
    target.scrollTop = target.scrollHeight;
  }

  function joinTranscriptChunks(items) {
    return items
      .map((item) => item.text)
      .filter(Boolean)
      .join(" ")
      .replace(/\s+([,.;:!?])/g, "$1")
      .replace(/\s{2,}/g, " ")
      .trim();
  }

  function cleanCallerTranscript(rawText) {
    if (!rawText) return "";
    let cleaned = ` ${rawText} `;
    cleaned = cleaned.replace(/\b(can you hear me\??\s*)+/gi, " ");
    cleaned = cleaned.replace(/\b(uh|um|erm|hmm|ah)\b/gi, " ");
    cleaned = cleaned.replace(/\b(\w+)(\s+\1\b)+/gi, "$1");
    cleaned = cleaned.replace(/\s+([,.;:!?])/g, "$1");
    cleaned = cleaned.replace(/\s{2,}/g, " ").trim();

    const sentences = cleaned
      .split(/(?<=[.!?])\s+/)
      .map((item) => item.trim())
      .filter(Boolean);
    const deduped = [];
    for (const sentence of sentences) {
      if (!deduped.length || deduped[deduped.length - 1].toLowerCase() !== sentence.toLowerCase()) {
        deduped.push(sentence);
      }
    }
    return deduped.join(" ").trim();
  }

  function renderDialogueStreams() {
    const callerRaw = joinTranscriptChunks(userTranscriptChunks);
    const agentRaw = joinTranscriptChunks(agentTranscriptChunks);
    const callerCleaned = cleanCallerTranscript(callerRaw);

    callerStreamEl.textContent = callerRaw || "Waiting for caller audio.";
    agentStreamEl.textContent = agentRaw || "Waiting for dispatcher audio.";
    cleanedStreamEl.textContent = callerCleaned || "Waiting for cleaned caller text.";
  }

  function logEvent(text, kind = "event") {
    appendEntry(eventLogEl, kind, kind === "error" ? "Error" : "Event", text);
    eventEntries.push({
      at: new Date().toISOString(),
      kind,
      text,
    });
  }

  function clearTranscriptBuffers() {
    if (userFinalizeHandle) {
      clearTimeout(userFinalizeHandle);
      userFinalizeHandle = null;
    }
    pendingUserChunks = [];
    pendingAgentChunks = [];
  }

  function finalizePendingUser(reason = "silence") {
    if (userFinalizeHandle) {
      clearTimeout(userFinalizeHandle);
      userFinalizeHandle = null;
    }
    if (!pendingUserChunks.length) return;
    const text = joinTranscriptChunks(pendingUserChunks);
    pendingUserChunks = [];
    if (!text) return;
    userTranscripts.push({
      at: new Date().toISOString(),
      text,
      reason,
    });
    finalizedCallerTurnIndex += 1;
    finalizedCallerTurnPerf = performance.now();
    recordCallerTurnLatency("transcript_finalized");
    emergencyAssessmentStatus = "Processing transcript";
    renderEmergencyAssessment(latestSession || {});
    if (ws && ws.readyState === WebSocket.OPEN && liveMlClassifier) {
      // Keep the model request off the microphone and audio processing path.
      void liveMlClassifier.finalize(text, sessionId);
    }
    scheduleTriageSync();
  }

  function finalizePendingAgent(reason = "turn_end") {
    if (!pendingAgentChunks.length) return;
    const text = joinTranscriptChunks(pendingAgentChunks);
    pendingAgentChunks = [];
    if (!text) return;
    agentTranscripts.push({
      at: new Date().toISOString(),
      text,
      reason,
    });
    scheduleTriageSync();
  }

  function scheduleUserFinalize(latestText) {
    if (userFinalizeHandle) {
      clearTimeout(userFinalizeHandle);
    }
    const pendingText = joinTranscriptChunks(pendingUserChunks);
    const wordCount = pendingText ? pendingText.split(/\s+/).filter(Boolean).length : 0;
    const endsWithBoundary = /[.!?,]$/.test((latestText || "").trim());
    const delay = endsWithBoundary && wordCount >= 3 ? SHORT_FINALIZE_DELAY_MS : DEFAULT_FINALIZE_DELAY_MS;
    userFinalizeHandle = setTimeout(() => {
      userFinalizeHandle = null;
      finalizePendingUser("silence");
    }, delay);
  }

  function logTranscript(text, isUser) {
    if (!text || !String(text).trim()) return;
    if (!isUser && !humanTakeoverEngaged && pendingHandoffResumePrompt) {
      pendingHandoffResumePrompt = null;
    }
    const item = {
      at: new Date().toISOString(),
      text,
    };
    if (isUser) {
      userTranscriptChunks.push(item);
      pendingUserChunks.push(item);
      scheduleUserFinalize(text);
      queueFirstAidFromCaller();
    } else {
      agentTranscriptChunks.push(item);
      pendingAgentChunks.push(item);
    }
    renderDialogueStreams();
  }

  function resetFirstAidPanel() {
    firstAidRequestVersion += 1;
    if (firstAidDebounceHandle) clearTimeout(firstAidDebounceHandle);
    firstAidDebounceHandle = null;
    lastFirstAidText = "";
    if (firstAidText) firstAidText.value = "";
    if (firstAidStatus) firstAidStatus.textContent = "Ready · local first-aid rules available.";
    if (firstAidResult) firstAidResult.innerHTML = "";
  }

  function renderFirstAidGuidance(guidance) {
    const escalation = guidance.emergency_escalation || {};
    const callNow = Boolean(escalation.recommended);
    const escalationHeading = escalation.level === "contact_local_poison_center_or_urgent_medical_service_now"
      ? "CONTACT YOUR LOCAL POISON CENTER OR URGENT MEDICAL SERVICE NOW"
      : escalation.level === "emergency_services_if_seizure_red_flags"
        ? "CALL EMERGENCY SERVICES IF SEIZURE RED FLAGS APPLY"
        : callNow
          ? "CALL LOCAL EMERGENCY SERVICES NOW — THIS APP DOES NOT PLACE THE CALL"
          : "URGENT MEDICAL ASSESSMENT IF RED FLAGS ARE PRESENT";
    const renderList = (heading, items) => {
      const safeItems = Array.isArray(items) ? items : [];
      if (!safeItems.length) return "";
      return `<h3>${escapeHtml(heading)}</h3><ul>${safeItems.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
    };
    const sources = Array.isArray(guidance.sources) ? guidance.sources : [];
    const relatedGuidance = Array.isArray(guidance.related_guidance) ? guidance.related_guidance : [];
    const relatedMarkup = relatedGuidance.map((related) => `
      <section class="first-aid-related">
        <h3>Also detected: ${escapeHtml(related.scenario?.label || "Additional finding")}</h3>
        ${renderList("Immediate actions", related.immediate_actions)}
        ${renderList("Avoid", related.actions_to_avoid)}
        ${renderList("Follow-up questions", related.critical_follow_up_questions)}
      </section>
    `).join("");
    const sourceMarkup = sources.map((source) => {
      const url = /^https:\/\//i.test(String(source.url || "")) ? source.url : "";
      const title = escapeHtml(`${source.organization || "Source"}: ${source.title || "Reference"}`);
      const link = url
        ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${title}</a>`
        : title;
      return `<p class="first-aid-source">${link} · checked ${escapeHtml(source.verified_on || "date unavailable")}</p>`;
    }).join("");
    const flags = (guidance.safety_flags || []).map((item) => item.replaceAll("_", " "));
    firstAidResult.innerHTML = `
      <article class="first-aid-guidance">
        <div class="first-aid-escalation ${callNow ? "urgent" : "assessment"}">${escapeHtml(escalationHeading)}<br>${escapeHtml(escalation.message || "Follow local emergency guidance.")}</div>
        <strong>${escapeHtml(guidance.detected_scenario?.label || "Unknown emergency")} · ${escapeHtml(String(guidance.severity || "unknown").toUpperCase())}</strong>
        ${renderList("Immediate actions", guidance.immediate_actions)}
        ${renderList("Avoid", guidance.actions_to_avoid)}
        ${renderList("Follow-up questions", guidance.critical_follow_up_questions)}
        ${relatedMarkup}
        ${flags.length ? `<p><strong>Safety flags:</strong> ${escapeHtml(flags.join(" · "))}</p>` : ""}
        ${sourceMarkup}
        <p class="first-aid-limitations">${escapeHtml(guidance.limitations || "Advisory only; follow local emergency services.")} · Local rules ${escapeHtml(guidance.engine?.version || "")}; not live AI.</p>
      </article>
    `;
  }

  async function requestFirstAidGuidance(text, expectedVersion = null, force = false) {
    const normalized = String(text || "").trim();
    if (!normalized) return;
    if (expectedVersion !== null && expectedVersion !== firstAidRequestVersion) return;
    if (!force && normalized === lastFirstAidText) return;
    const requestVersion = expectedVersion === null ? ++firstAidRequestVersion : expectedVersion;
    const requestSessionId = sessionId;
    lastFirstAidText = normalized;
    if (firstAidStatus) firstAidStatus.textContent = "Analyzing locally · deterministic first-aid rules · not live AI.";
    try {
      const response = await fetch("/api/first-aid", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: normalized }),
      });
      const payload = await response.json();
      if (!window.DispatcherAudioUtils.isCurrentSessionRequest(requestVersion, firstAidRequestVersion, requestSessionId, sessionId)) return;
      if (!response.ok) throw new Error(payload.detail || `Guidance request failed (${response.status})`);
      renderFirstAidGuidance(payload);
      const needsHuman = payload.human_review_required ? " · Human review required" : "";
      firstAidStatus.textContent = `LOCAL FIRST-AID RULES · ${payload.engine?.version || "version unknown"} · advisory only${needsHuman}`;
    } catch (error) {
      if (!window.DispatcherAudioUtils.isCurrentSessionRequest(requestVersion, firstAidRequestVersion, requestSessionId, sessionId)) return;
      firstAidStatus.textContent = `Local guidance unavailable · ${error?.message || "request failed"}`;
    }
  }

  function queueFirstAidFromCaller() {
    const transcript = joinTranscriptChunks(userTranscriptChunks);
    if (!transcript || transcript === lastFirstAidText) return;
    if (firstAidDebounceHandle) clearTimeout(firstAidDebounceHandle);
    const queuedVersion = ++firstAidRequestVersion;
    firstAidDebounceHandle = setTimeout(() => {
      firstAidDebounceHandle = null;
      requestFirstAidGuidance(transcript, queuedVersion);
    }, 650);
  }

  function getWsUrl() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/ws`;
  }

  function currentDispatchContextForWire() {
    if (!latestDispatchContext) return null;
    return { ...latestDispatchContext };
  }

  function currentLedgerContextForWire() {
    if (!latestLedgerContext) return null;
    return { ...latestLedgerContext };
  }

  function raiseAdaptivePatience(reason) {
    const nowMs = Date.now();
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (nowMs - lastAdaptiveAt < 4000) return;
    interruptCount += 1;
    lastAdaptiveAt = nowMs;
    recoveryModeArmed = true;
    ws.send(
      JSON.stringify({
        type: "config",
        reason,
        interrupt_count: interruptCount,
        interruption_recovery: true,
        triage_context: latestTriageContext,
        ledger_context: currentLedgerContextForWire(),
        dispatch_context: currentDispatchContextForWire(),
        human_takeover_active: humanTakeoverEngaged,
        handoff_recovery_prompt: pendingHandoffResumePrompt,
      })
    );
    logEvent(`adaptive patience raised after interruption (${interruptCount})`);
  }

  function clearRecoveryMode() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !recoveryModeArmed) return;
    recoveryModeArmed = false;
    interruptCount = Math.max(0, interruptCount - 1);
    ws.send(
      JSON.stringify({
        type: "config",
        reason: "recovery_complete",
        interrupt_count: interruptCount,
        interruption_recovery: false,
        triage_context: latestTriageContext,
        ledger_context: currentLedgerContextForWire(),
        dispatch_context: currentDispatchContextForWire(),
        human_takeover_active: humanTakeoverEngaged,
        handoff_recovery_prompt: pendingHandoffResumePrompt,
      })
    );
  }

  function sendTriageConfig(reason = "triage_update") {
    const hasControlUpdate = humanTakeoverEngaged || Boolean(pendingHandoffResumePrompt);
    if (!ws || ws.readyState !== WebSocket.OPEN || (!latestTriageContext && !latestLedgerContext && !latestDispatchContext && !hasControlUpdate)) return;
    const dispatchContext = currentDispatchContextForWire();
    ws.send(
      JSON.stringify({
        type: "config",
        reason,
        interrupt_count: interruptCount,
        interruption_recovery: recoveryModeArmed,
        triage_context: latestTriageContext,
        ledger_context: currentLedgerContextForWire(),
        dispatch_context: dispatchContext,
        human_takeover_active: humanTakeoverEngaged,
        handoff_recovery_prompt: pendingHandoffResumePrompt,
      })
    );
    if (latestDispatchContext?.announcement_line) {
      latestDispatchContext = { ...latestDispatchContext, announcement_line: null };
    }
  }

  function scheduleTriageSync() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (triageSyncTimeout) {
      clearTimeout(triageSyncTimeout);
    }
    triageSyncTimeout = setTimeout(() => {
      triageSyncTimeout = null;
      syncSessionReport("live").catch(() => {});
    }, 150);
  }

  function initMap() {
    if (!window.L || map) return;
    map = window.L.map("map", {
      zoomControl: false,
      attributionControl: true,
    }).setView(INDIA_MAP_CENTER, INDIA_MAP_ZOOM);
    map.on("click", async (event) => {
      if (!mapCorrectionMode) return;
      if (!sessionId || !latestSession?.demo_simulation?.simulation_only) {
        mapStatusEl.textContent = "Map point selection is available only during a demo simulation.";
        mapCorrectionMode = false;
        return;
      }
      mapCorrectionMode = false;
      if (mapCorrectionToggle) mapCorrectionToggle.textContent = "Select incident point on map";
      mapStatusEl.textContent = "Saving operator-selected demo point…";
      try {
        const response = await fetch(`/api/demo/location/${encodeURIComponent(sessionId)}`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ lat: event.latlng.lat, lon: event.latlng.lng }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || `Map correction failed (${response.status}).`);
        await refreshSessionDetail();
        mapStatusEl.textContent = "Operator-selected point saved · simulation only · verify the caller's exact location.";
      } catch (error) {
        mapStatusEl.textContent = error.message || "Could not save the map correction.";
      }
    });

    window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);

    window.L.control
      .zoom({
        position: "bottomright",
      })
      .addTo(map);
    invalidateMapSizeSoon();
  }

  function invalidateMapSizeSoon() {
    if (!map) return;
    window.requestAnimationFrame(() => {
      window.setTimeout(() => {
        if (map) {
          map.invalidateSize(false);
        }
      }, 80);
    });
  }

  function clearMapState() {
    if (mapMarker && map) {
      map.removeLayer(mapMarker);
      mapMarker = null;
    }
    lastValidMapLocation = null;
    mapSessionId = sessionId;
    lastMapAnimationKey = null;
    if (demoMapLayer && map) map.removeLayer(demoMapLayer);
    demoMapLayer = null;
    demoRouteLine = null;
    demoAmbulanceMarker = null;
    demoMapSessionId = null;
    if (demoMapLegend && map) map.removeControl(demoMapLegend);
    demoMapLegend = null;
    if (map) {
      map.setView(INDIA_MAP_CENTER, INDIA_MAP_ZOOM);
    }
    mapStatusEl.textContent = "Waiting for a geocoded location.";
  }

  function renderDemoSimulationMap(simulation) {
    if (!map || !simulation?.simulation_only) return;
    const route = simulation.route || {};
    const routeCoordinates = Array.isArray(route.coordinates) ? route.coordinates : [];
    if (demoMapSessionId !== simulation.session_id) {
      if (demoMapLayer) map.removeLayer(demoMapLayer);
      demoMapLayer = window.L.layerGroup().addTo(map);
      demoRouteLine = null;
      demoAmbulanceMarker = null;
      demoMapSessionId = simulation.session_id;
      for (const unit of (simulation.fleet || [])) {
        if (unit.id === simulation.selected_unit_id) continue;
        window.L.circleMarker([unit.lat, unit.lon], { radius: 6, color: "#64748b", fillColor: "#cbd5e1", fillOpacity: 0.9 }).bindPopup(`${escapeHtml(unit.call_sign)} · simulated fleet · available`).addTo(demoMapLayer);
      }
      if (demoMapLegend && map) map.removeControl(demoMapLegend);
      demoMapLegend = window.L.control({ position: "topright" });
      demoMapLegend.onAdd = () => {
        const element = window.L.DomUtil.create("div", "demo-map-legend");
        element.textContent = "Incident pin · simulated ambulance · dashed simulated route";
        return element;
      };
      demoMapLegend.addTo(map);
    }
    const routeKey = JSON.stringify(routeCoordinates);
    if (demoRouteLine?.routeKey !== routeKey) {
      if (demoRouteLine) demoMapLayer.removeLayer(demoRouteLine);
      demoRouteLine = routeCoordinates.length >= 2
        ? window.L.polyline(routeCoordinates, { color: "#2563eb", weight: 4, opacity: 0.8, dashArray: "8 8" }).addTo(demoMapLayer)
        : null;
      if (demoRouteLine) demoRouteLine.routeKey = routeKey;
    }
    const position = simulation.ambulance_position || routeCoordinates[0];
    if (position?.length === 2) {
      if (!demoAmbulanceMarker) {
        demoAmbulanceMarker = window.L.marker(position, {
          icon: window.L.divIcon({ className: "demo-ambulance-icon", html: "🚑", iconSize: [30, 30], iconAnchor: [15, 15] }),
        }).addTo(demoMapLayer);
      } else demoAmbulanceMarker.setLatLng(position);
      demoAmbulanceMarker.bindPopup(`${escapeHtml(simulation.selected_call_sign || "Demo Ambulance")} · ${escapeHtml(String(simulation.ambulance_status || "assigned"))} · SIMULATION ONLY`);
    }
  }

  function updateMap(state) {
    initMap();
    if (mapSessionId !== sessionId) {
      if (mapMarker && map) map.removeLayer(mapMarker);
      mapMarker = null;
      lastValidMapLocation = null;
      lastMapAnimationKey = null;
      mapSessionId = sessionId;
      if (map) map.setView(INDIA_MAP_CENTER, INDIA_MAP_ZOOM);
    }
    const { lat, lon, location, mapSource, mapConfidence, mapReason, mapCandidates, locationNeedsConfirmation, dispatchServices } = state;
    if (mapCorrectionToggle) {
      mapCorrectionToggle.disabled = !state.demoSimulation;
      if (!state.demoSimulation) mapCorrectionMode = false;
    }
    const candidateCoordinates = window.DispatcherLocationUtils.getMapCoordinates(lat, lon);
    const candidate = {
      lat,
      lon,
      location,
      source: mapSource,
      confidence: mapConfidence,
      reason: mapReason,
      candidates: mapCandidates,
      locationNeedsConfirmation,
      revision: state.mapRevision,
    };
    lastValidMapLocation = window.DispatcherLocationUtils.retainMapLocation(
      lastValidMapLocation,
      candidate,
      sessionId
    );
    if (!lastValidMapLocation || !map) {
      const candidateText = window.DispatcherLocationUtils.getUnresolvedMapStatus({
        location,
        mapReason,
        mapSource,
        mapCandidates,
        locationNeedsConfirmation,
      });
      const locationFollowup = state.incidentAssessmentLocation?.clarification;
      const geocoderError = state.incidentAssessmentLocation?.error_category;
      mapStatusEl.textContent = geocoderError
        ? `${candidateText} Geocoder: ${geocoderError}. ${locationFollowup || "Please provide a city and nearby landmark."}`
        : candidateText;
      return;
    }

    const retained = lastValidMapLocation;
    const coordinates = retained.coordinates || candidateCoordinates;
    const animationKey = `${sessionId}:${retained.lat}:${retained.lon}:${retained.confidence}:${retained.location}`;
    if (!mapMarker) {
      mapMarker = window.L.marker(coordinates).addTo(map);
    } else {
      mapMarker.setLatLng(coordinates);
    }
    const mapPresentation = window.DispatcherLocationUtils.getMapPresentation({
      location: retained.location,
      confidence: retained.confidence,
      reason: retained.reason,
      source: retained.source,
    });
    const originalPhrase = state.demoSimulation?.location?.original_phrase || state.incidentAssessmentLocation?.original_phrase || "Not available";
    const locationSource = retained.source || "unknown source";
    const locationQuality = retained.confidence || "unverified";
    const confirmation = state.demoSimulation?.location?.operator_confirmed
      ? "operator-selected"
      : state.demoSimulation?.location?.source === "demo_configuration"
        ? "simulated fallback"
        : state.demoSimulation?.location?.verified_from_caller
          ? "caller-confirmed"
          : "approximate or pending confirmation";
    mapMarker.bindPopup(`${escapeHtml(mapPresentation.markerLabel)}<br>Original phrase: ${escapeHtml(originalPhrase)}<br>${coordinates[0].toFixed(5)}, ${coordinates[1].toFixed(5)}<br>Source: ${escapeHtml(locationSource)} · quality: ${escapeHtml(locationQuality)} · ${escapeHtml(confirmation)}`).openPopup();
    if (lastMapAnimationKey !== animationKey) {
      lastMapAnimationKey = animationKey;
      map.flyTo(coordinates, mapPresentation.zoom, { animate: true, duration: 0.85 });
    } else {
      map.setView(coordinates, Math.min(mapPresentation.zoom, 14));
    }
    renderDemoSimulationMap(state.demoSimulation);
    invalidateMapSizeSoon();
    if (retained.retained) {
      const currentCandidateStatus = window.DispatcherLocationUtils.getUnresolvedMapStatus({
        location,
        mapReason,
        mapSource,
        mapCandidates,
        locationNeedsConfirmation,
      });
      const reason = retained.retentionReason === "lower_confidence"
        ? "A higher-confidence pin is retained while this candidate is reviewed."
        : retained.retentionReason === "stale_update"
          ? "A stale location update was ignored."
          : "The current update omitted coordinates; the prior pin is retained.";
      mapStatusEl.textContent = `${currentCandidateStatus} ${reason}`;
    } else {
      mapStatusEl.textContent = state.demoSimulation
        ? `${state.demoSimulation.location?.display_label || "DEMO LOCATION — NOT VERIFIED FROM CALLER"} · ${state.demoSimulation.route?.label || "SIMULATED ROUTE — NOT ROAD NAVIGATION"} · ${state.demoSimulation.stage}`
        : mapPresentation.status;
    }
  }

  function flushPcmPending(forcePad = false) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const chunkSamples = currentAudioConfig.chunk_samples || 1920;
    while (pcmCapturePending.length >= chunkSamples) {
      const chunk = new Float32Array(pcmCapturePending.slice(0, chunkSamples));
      pcmCapturePending = pcmCapturePending.slice(chunkSamples);
      const pcm = window.DispatcherAudioUtils.float32ToPcm16Buffer(chunk);
      recordVoiceLatency("outbound_audio_send");
      ws.send(pcm);
      if (pcmDiagnostics) {
        pcmDiagnostics.chunks += 1;
        pcmDiagnostics.bytes += pcm.byteLength;
        if (pcmDiagnostics.chunks === 1 || pcmDiagnostics.chunks % 100 === 0) {
          logEvent(`microphone_uplink stage=browser_to_server at=${new Date().toISOString()} session=${sessionId} chunks=${pcmDiagnostics.chunks} bytes=${pcmDiagnostics.bytes} sample_rate_hz=${pcmTargetRate} channels=1 format=s16le frame_ms=${(chunkSamples / pcmTargetRate * 1000).toFixed(1)}`);
        }
      }
    }
    if (forcePad && pcmCapturePending.length) {
      const padded = new Float32Array(chunkSamples);
      padded.set(pcmCapturePending.slice(0, chunkSamples));
      pcmCapturePending = [];
      const pcm = window.DispatcherAudioUtils.float32ToPcm16Buffer(padded);
      recordVoiceLatency("outbound_audio_send");
      ws.send(pcm);
      if (pcmDiagnostics) {
        pcmDiagnostics.chunks += 1;
        pcmDiagnostics.bytes += pcm.byteLength;
        logEvent(`microphone_uplink stage=browser_to_server at=${new Date().toISOString()} session=${sessionId} chunks=${pcmDiagnostics.chunks} bytes=${pcmDiagnostics.bytes} sample_rate_hz=${pcmTargetRate} channels=1 format=s16le frame_ms=${(chunkSamples / pcmTargetRate * 1000).toFixed(1)} padded=true`);
      }
    }
  }

  function stopPcmCapture() {
    if (pcmDiagnostics && pcmDiagnostics.samples) {
      const durationS = pcmDiagnostics.samples / pcmInputRate;
      const rms = Math.sqrt(pcmDiagnostics.sumSquares / pcmDiagnostics.samples);
      logEvent(
        `microphone PCM summary: ${pcmDiagnostics.chunks} chunks, ${pcmDiagnostics.bytes} bytes, ${durationS.toFixed(2)} s, ${pcmInputRate} -> ${pcmTargetRate} Hz, mono s16le, RMS ${rms.toFixed(4)}, peak ${pcmDiagnostics.peak.toFixed(4)}`
      );
    }
    pcmDiagnostics = null;
    pcmResampler = null;
    pcmCapturePending = [];
    if (pcmCaptureNode) {
      pcmCaptureNode.disconnect();
      pcmCaptureNode.onaudioprocess = null;
      pcmCaptureNode = null;
    }
    if (pcmCaptureSource) {
      pcmCaptureSource.disconnect();
      pcmCaptureSource = null;
    }
    if (pcmCaptureSink) {
      pcmCaptureSink.disconnect();
      pcmCaptureSink = null;
    }
  }

  function startPcmCapture(playerInstance, audioConfig) {
    stopPcmCapture();
    if (!audioConfig.pcm_input) return;
    const audioProcessor = playerInstance?.audioProcessor;
    const audioContext = audioProcessor?.audioContext;
    const mediaStream = audioProcessor?.mediaStream;
    if (!audioContext || !mediaStream) {
      logEvent("pcm microphone uplink unavailable", "error");
      return;
    }

    const targetRate = audioConfig.sample_rate || audioContext.sampleRate;
    pcmInputRate = audioContext.sampleRate;
    pcmTargetRate = targetRate;
    pcmResampler = window.DispatcherAudioUtils.createLinearResampler(audioContext.sampleRate, targetRate);
    pcmDiagnostics = { chunks: 0, bytes: 0, sumSquares: 0, samples: 0, captureFrames: 0, peak: 0 };

    pcmCaptureSource = audioContext.createMediaStreamSource(mediaStream);
    pcmCaptureNode = audioContext.createScriptProcessor(4096, 1, 1);
    pcmCaptureSink = audioContext.createGain();
    pcmCaptureSink.gain.value = 0;

    pcmCaptureNode.onaudioprocess = (event) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const input = event.inputBuffer.getChannelData(0);
      if (!input || !input.length) return;
      if (pcmDiagnostics) {
        pcmDiagnostics.captureFrames += 1;
        if (pcmDiagnostics.captureFrames === 1 || pcmDiagnostics.captureFrames % 100 === 0) {
          logEvent(`microphone_capture stage=frame at=${new Date().toISOString()} session=${sessionId} state=active frames=${pcmDiagnostics.captureFrames} samples=${pcmDiagnostics.samples} input_rate_hz=${pcmInputRate} channels=1 format=float32`);
        }
      }
      recordVoiceLatency("microphone_capture");
      let frameSquares = 0;
      for (let i = 0; i < input.length; i += 1) {
        frameSquares += input[i] * input[i];
        if (pcmDiagnostics) pcmDiagnostics.peak = Math.max(pcmDiagnostics.peak, Math.abs(input[i]));
      }
      if (pcmDiagnostics) {
        pcmDiagnostics.sumSquares += frameSquares;
        pcmDiagnostics.samples += input.length;
      }
      const resampled = pcmResampler.process(input);
      for (let i = 0; i < resampled.length; i += 1) {
        pcmCapturePending.push(resampled[i]);
      }
      flushPcmPending(false);
    };

    pcmCaptureSource.connect(pcmCaptureNode);
    pcmCaptureNode.connect(pcmCaptureSink);
    pcmCaptureSink.connect(audioContext.destination);
    logEvent(
      `pcm microphone format: ${audioContext.sampleRate} Hz input -> ${targetRate} Hz, mono, signed 16-bit little-endian; resampling state retained across capture buffers`
    );
  }

  async function createPlayer(audioConfig, playerSessionId) {
    currentAudioConfig = audioConfig;
    const handlePlaybackMetrics = (metrics) => {
      const bufferMs = Math.round(metrics.bufferMs || 0);
      currentBufferMs = bufferMs;
      maxBufferMs = Math.max(maxBufferMs, bufferMs);
      bufferMetricEl.textContent = `${bufferMs} ms`;
      const playedSeconds = Number(metrics.actualAudioPlayed || 0);
      if (playedSeconds > audioPlayedSeconds) {
        const firstPlaybackProgress = !assistantPlaybackStartedThisTurn;
        audioPlayedSeconds = playedSeconds;
        assistantPlaybackStartedThisTurn = true;
        setPlaybackStatus("Dispatcher audio playback started", "live");
        if (firstPlaybackProgress) {
          recordVoiceLatency("actual_playback_start");
          recordCallerTurnLatency("actual_playback_start");
          logEvent("dispatcher audio playback started");
        }
        if (playbackWatchdogHandle) {
          clearTimeout(playbackWatchdogHandle);
          playbackWatchdogHandle = null;
        }
        logEvent(window.DispatcherAudioUtils.getSafePlaybackDiagnostic({
          stage: "playing",
          contextState: syncedPlayer.audioProcessor?.audioContext?.state,
          receivedChunks: playbackQueueDiagnostics.receivedChunks,
          receivedBytes: playbackQueueDiagnostics.receivedBytes,
          decodedSamples: playbackQueueDiagnostics.decodedSamples,
          queuedChunks: playbackQueueDiagnostics.queuedChunks,
          actualAudioPlayed: playedSeconds,
          bufferMs,
          sampleRate: syncedPlayer.audioProcessor?.audioContext?.sampleRate,
        }));
      }
      updateAudioUnlockState(syncedPlayer);
    };
    const syncedPlayer = new window.SyncedAudioPlayer({
      basePath: "/static/js",
      pcmOutput: Boolean(audioConfig.pcm),
      onEncodedAudio: (audioData) => {
        if (!audioConfig.pcm_input && ws && ws.readyState === WebSocket.OPEN) {
          recordVoiceLatency("outbound_audio_send");
          ws.send(audioData);
        }
      },
      onMetrics: handlePlaybackMetrics,
      onText: ({ text, isUser }) => {
        if (sessionId !== playerSessionId) return;
        if (isUser) {
          clearProviderTimeout();
          assistantTextReceivedThisTurn = false;
          assistantAudioBytesThisTurn = 0;
          assistantPlaybackStartedThisTurn = false;
          if (responseTimeoutHandle) {
            clearTimeout(responseTimeoutHandle);
            responseTimeoutHandle = null;
          }
        } else {
          assistantTextReceivedThisTurn = true;
          clearProviderTimeout();
          setPlaybackStatus("Dispatcher response text received; waiting for audio");
          if (responseTimeoutHandle) {
            clearTimeout(responseTimeoutHandle);
            responseTimeoutHandle = null;
          }
        }
        logEvent(isUser ? "caller transcript event received from primary provider" : "dispatcher response text event received");
        logTranscript(text, isUser);
      },
      onEvent: (eventName) => {
        if (sessionId !== playerSessionId) return;
        logEvent(eventName);
        if (["push_to_llm", "llm_started", "first_word", "first_tts_audio"].includes(eventName)) {
          assistantTurnActive = true;
        }
        if (eventName === "push_to_llm") {
          setPlaybackStatus("Generating dispatcher response");
          scheduleProviderTimeout(playerSessionId);
        }
        if (eventName === "llm_started") {
          recordVoiceLatency("first_provider_response");
          recordCallerTurnLatency("first_provider_response");
          assistantTextReceivedThisTurn = false;
          assistantAudioBytesThisTurn = 0;
          assistantPlaybackStartedThisTurn = false;
          playbackQueueDiagnostics = { receivedChunks: 0, receivedBytes: 0, decodedSamples: 0, queuedChunks: 0 };
          if (playbackWatchdogHandle) {
            clearTimeout(playbackWatchdogHandle);
            playbackWatchdogHandle = null;
          }
          clearProviderTimeout();
          setPlaybackStatus("Generating dispatcher response");
          scheduleProviderTimeout(playerSessionId, true);
        } else if (eventName === "first_tts_audio") {
          setPlaybackStatus("Provider reports speech generation; waiting for audio bytes");
        } else if (eventName === "end_of_turn") {
          const playbackOutcome = window.DispatcherAudioUtils.getPlaybackOutcome(
            assistantTextReceivedThisTurn,
            assistantAudioBytesThisTurn,
            assistantPlaybackStartedThisTurn
          );
          if (playbackOutcome === "text_without_audio") {
            setPlaybackStatus("Dispatcher text returned without audio", "error");
            logEvent("dispatcher response had text but no audio bytes", "error");
          } else if (playbackOutcome === "not_played") {
            setPlaybackStatus("Audio bytes arrived; waiting for browser playback confirmation", "live");
            updateAudioUnlockState(syncedPlayer);
          } else if (playbackOutcome === "played") {
            setPlaybackStatus("Dispatcher audio turn finished", "live");
          }
        }
        if (eventName === "previous_llm_gen") {
          raiseAdaptivePatience("caller_interrupt");
        }
      },
      onError: (error) => {
        const errorName = error?.name || "AudioError";
        const contextState = syncedPlayer.audioProcessor?.audioContext?.state || "unavailable";
        logEvent(`audio stream or playback failed stage=player error_type=${errorName} context=${contextState}`, "error");
        setPlaybackStatus("Audio stream failed; check provider status and browser output", "error");
      },
      onEndOfTurn: () => {
        if (sessionId !== playerSessionId) return;
        assistantTurnActive = false;
        finalizePendingAgent("turn_end");
        logEvent("assistant turn finished");
        clearRecoveryMode();
        if (pendingTriageUpdate) {
          const queuedUpdate = pendingTriageUpdate;
          pendingTriageUpdate = null;
          pushTriageContext(queuedUpdate);
        }
        if (pendingLedgerUpdate) {
          const queuedUpdate = pendingLedgerUpdate;
          pendingLedgerUpdate = null;
          pushLedgerContext(queuedUpdate);
        }
        if (pendingDispatchUpdate) {
          const queuedDispatch = pendingDispatchUpdate;
          pendingDispatchUpdate = null;
          pushDispatchContext(queuedDispatch);
        }
      },
    });
    if (audioConfig.pcm_input) {
      logEvent(`microphone_permission stage=requested at=${new Date().toISOString()} session=${playerSessionId}`);
    }
    await syncedPlayer.start();
    if (audioConfig.pcm_input) {
      const inputTracks = syncedPlayer.audioProcessor?.mediaStream?.getAudioTracks?.() || [];
      logEvent(`microphone_permission stage=granted at=${new Date().toISOString()} session=${playerSessionId} audio_tracks=${inputTracks.length} live_tracks=${inputTracks.filter((track) => track.readyState === "live" && track.enabled).length}`);
    }
    // SyncedAudioPlayer uses AudioProcessor metrics internally but does not
    // forward them to the onMetrics option. Relay worklet playback counters so
    // the UI can distinguish received bytes from samples actually played.
    if (!window.DispatcherAudioUtils.relayAudioMetrics(syncedPlayer.audioProcessor, handlePlaybackMetrics)) {
      throw new Error("Browser audio playback metrics are unavailable.");
    }
    updateAudioUnlockState(syncedPlayer);
    const audioContext = syncedPlayer.audioProcessor?.audioContext;
    if (audioContext) {
      if (audioContext.sampleRate !== 48000) {
        throw new Error(`Browser audio output requires 48000 Hz; context is ${audioContext.sampleRate} Hz.`);
      }
      logEvent(window.DispatcherAudioUtils.getSafePlaybackDiagnostic({
        stage: "ready",
        contextState: audioContext.state,
        sampleRate: audioContext.sampleRate,
      }));
      audioContext.addEventListener("statechange", () => {
        if (sessionId !== playerSessionId) return;
        logEvent(window.DispatcherAudioUtils.getSafePlaybackDiagnostic({
          stage: "context_state",
          contextState: audioContext.state,
          receivedChunks: playbackQueueDiagnostics.receivedChunks,
          receivedBytes: playbackQueueDiagnostics.receivedBytes,
          decodedSamples: playbackQueueDiagnostics.decodedSamples,
          queuedChunks: playbackQueueDiagnostics.queuedChunks,
          actualAudioPlayed: audioPlayedSeconds,
          sampleRate: audioContext.sampleRate,
        }));
        updateAudioUnlockState(syncedPlayer);
      });
      const worklet = syncedPlayer.audioProcessor?.outputWorklet;
      if (worklet) {
        worklet.addEventListener("processorerror", (error) => {
          if (sessionId !== playerSessionId) return;
          const errorName = error?.error?.name || "AudioWorkletError";
          logEvent(`audio playback failed stage=worklet error_type=${errorName} context=${audioContext.state}`, "error");
          setPlaybackStatus("Browser audio worklet failed; dispatcher audio could not play", "error");
        });
      }
    }
    if (audioConfig.pcm_input) {
      startPcmCapture(syncedPlayer, audioConfig);
    }
    return syncedPlayer;
  }

  function buildSessionReport(status) {
    return {
      session_id: sessionId,
      report_revision: ++sessionReportRevision,
      ...window.DispatcherSessionReportUtils.sessionStatusFields(status),
      triage_mode: getSelectedDispatchMode(),
      started_at: sessionStartedAt,
      transcripts: {
        user: userTranscripts,
        agent: agentTranscripts,
      },
      events: eventEntries,
      metrics: {
        max_buffer_ms: maxBufferMs,
      },
      page: {
        href: window.location.href,
      },
    };
  }

  function pushTriageContext(update) {
    if (!update || !update.prompt_context) return;
    latestTriageContext = update.prompt_context;
    const version = Number(update.version || update.prompt_context.version || 0);
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!version || version <= lastTriageVersionSent) return;
    if (assistantTurnActive) {
      pendingTriageUpdate = update;
      return;
    }
    lastTriageVersionSent = version;
    sendTriageConfig("triage_update");
    logEvent(`triage context updated (${version})`);
  }

  function pushLedgerContext(update) {
    if (!update || !update.prompt_context) return;
    latestLedgerContext = update.prompt_context;
    const version = Number(update.version || update.prompt_context.version || 0);
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!version || version <= lastLedgerVersionSent) return;
    if (assistantTurnActive) {
      pendingLedgerUpdate = update;
      return;
    }
    lastLedgerVersionSent = version;
    sendTriageConfig("ledger_update");
    logEvent(`shared ledger updated (${version})`);
  }

  function pushDispatchContext(update) {
    if (!update || !update.prompt_context) return;
    latestDispatchContext = update.prompt_context;
    const version = Number(update.version || update.prompt_context.version || 0);
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!version || version <= lastDispatchVersionSent) return;
    if (assistantTurnActive) {
      pendingDispatchUpdate = update;
      return;
    }
    lastDispatchVersionSent = version;
    sendTriageConfig("dispatch_update");
    logEvent(`dispatch operations updated (${version})`);
  }

  function syncSessionReport(status = "live") {
    if (sessionReportPaused) return Promise.resolve(null);
    if (sessionReportSyncPromise) {
      queuedSessionReportStatus = status;
      return sessionReportSyncPromise;
    }
    const requestedSessionId = sessionId;
    sessionReportSyncPromise = performSessionReportSync(status, requestedSessionId).finally(() => {
      sessionReportSyncPromise = null;
      const queuedStatus = queuedSessionReportStatus;
      queuedSessionReportStatus = null;
      if (queuedStatus && sessionId) syncSessionReport(queuedStatus).catch(() => {});
    });
    return sessionReportSyncPromise;
  }

  async function performSessionReportSync(status, requestedSessionId) {
    if (!sessionId) return;
    setSessionReportStatus("Saving session report…");
    if (status === "live") {
      emergencyAssessmentStatus = "Updating assessment";
      renderEmergencyAssessment(latestSession || {});
    }
    const reportPayload = buildSessionReport(status);
    if (reportPayload.session_id !== requestedSessionId) return;
    try {
      const result = await window.DispatcherSessionReportUtils.postSessionReport({
        fetchImpl: (...args) => fetch(...args),
        url: "/api/session-report",
        payload: reportPayload,
        timeoutMs: 12000,
        retries: 1,
        onAttempt: ({ attempt, requestBytes }) => {
          logEvent(`session_report attempt=${attempt} bytes=${requestBytes} keepalive=false`);
        },
      });
      if (requestedSessionId !== sessionId) return result.payload;
      const responsePayload = result.payload;
      if (responsePayload.incident_assessment) {
        emergencyAssessmentStatus = "Assessment updated";
        if (latestSession?.session_id === requestedSessionId) {
          latestSession = { ...latestSession, incident_assessment: responsePayload.incident_assessment };
        } else {
          latestSession = { session_id: requestedSessionId, incident_assessment: responsePayload.incident_assessment };
        }
        renderEmergencyAssessment(latestSession);
      }
      sessionReportFailureCount = 0;
      if (!finalSessionReportStatus) sessionReportPaused = false;
      if (retrySessionReportBtn) retrySessionReportBtn.hidden = true;
      setSessionReportStatus(
        responsePayload.triage_meta?.runtime_provider === "unavailable"
          ? "Session report saved · live triage unavailable."
          : "Session report saved."
      );
      if (responsePayload?.triage_update?.prompt_context) {
        pushTriageContext(responsePayload.triage_update);
      }
      if (responsePayload?.ledger_update?.prompt_context) {
        pushLedgerContext(responsePayload.ledger_update);
      }
      if (responsePayload?.dispatch_update?.prompt_context) {
        pushDispatchContext(responsePayload.dispatch_update);
      }
      if (responsePayload?.stt_rescue_event) {
        const rescueIndex = Number(responsePayload.stt_rescue_event.user_index ?? -1);
        if (rescueIndex > lastRescueIndexSeen) {
          lastRescueIndexSeen = rescueIndex;
          logEvent(
            `stt rescue ${responsePayload.stt_rescue_event.selected_provider}: ${responsePayload.stt_rescue_event.corrected_transcript}`
          );
        }
      }
      return responsePayload;
    } catch (error) {
      if (requestedSessionId !== sessionId) return null;
      const reason = window.DispatcherSessionReportUtils.describeSessionReportError(error);
      sessionReportFailureCount += 1;
      if (sessionReportFailureCount >= 3) {
        sessionReportPaused = true;
        if (retrySessionReportBtn) retrySessionReportBtn.hidden = false;
      }
      setSessionReportStatus(
        sessionReportPaused
          ? `Session report not saved · ${reason}; automatic sync paused after 3 failed requests. Retry manually.`
          : `Session report not saved · ${reason}.`
      );
      emergencyAssessmentStatus = `Assessment update failed · ${reason}`;
      renderEmergencyAssessment(latestSession || {});
      const missingField = error?.missingField ? ` missing=${error.missingField}` : "";
      logEvent(`session_report failed kind=${error?.kind || "unknown"} status=${error?.status || "none"}${missingField}`, "error");
      throw error;
    }
  }

  function flushSessionReport(status) {
    if (!sessionId || reportSent) return;
    reportSent = true;
    const requestedSessionId = sessionId;
    finalSessionReportStatus = status;
    sessionReportPaused = true;
    queuedSessionReportStatus = null;
    setSessionReportStatus("Saving final session report…");

    const sendFinalReport = () => {
      if (requestedSessionId !== sessionId) return;
      const payload = JSON.stringify(buildSessionReport(status));
      const payloadBytes = window.DispatcherSessionReportUtils.utf8Length(payload);
      const finalBeaconLimit = 60 * 1024;
      const blob = new Blob([payload], { type: "application/json" });
      if (status === "unloaded" && payloadBytes <= finalBeaconLimit && navigator.sendBeacon) {
        if (navigator.sendBeacon("/api/session-report", blob)) {
          setSessionReportStatus("Final session report queued · save not yet confirmed.");
          return;
        }
      }
      if (status === "unloaded" && payloadBytes > finalBeaconLimit) {
        setSessionReportStatus("Final unload report not queued · browser beacon limit exceeded; prior live report saves may be incomplete.");
        return;
      }
      window.DispatcherSessionReportUtils.postSessionReport({
        fetchImpl: (...args) => fetch(...args),
        url: "/api/session-report",
        payload: JSON.parse(payload),
        timeoutMs: 12000,
        retries: 1,
        onAttempt: ({ attempt, requestBytes }) => logEvent(`session_report final attempt=${attempt} bytes=${requestBytes} keepalive=false`),
      })
        .then(({ payload: responsePayload }) => {
          if (requestedSessionId !== sessionId) return;
          setSessionReportStatus(responsePayload.status === "ok" || responsePayload.status === "saved" ? "Final session report saved." : "Final report response received.");
        })
        .catch((error) => {
          if (requestedSessionId !== sessionId) return;
          setSessionReportStatus(`Final session report not saved · ${window.DispatcherSessionReportUtils.describeSessionReportError(error)}.`);
        });
    };

    if (status !== "unloaded" && sessionReportSyncPromise) {
      sessionReportSyncPromise.then(sendFinalReport, sendFinalReport);
    } else {
      sendFinalReport();
    }
  }

  function setMemoryPanel(session) {
    if (!session) {
      memoryPanelEl.innerHTML = `
        <article class="entry event">
          <div class="body">Start a call to populate the stored session state.</div>
        </article>
      `;
      return;
    }
    const server = session.server || {};
    const client = session.client || {};
    const turns = client.transcripts || {};
    const userTurns = (turns.user || []).length;
    const agentTurns = (turns.agent || []).length;
    memoryPanelEl.innerHTML = `
      <div class="kv-list">
        <div class="kv-row"><span>Session</span><strong>${escapeHtml(session.session_id || "n/a")}</strong></div>
        <div class="kv-row"><span>Server</span><strong>${escapeHtml(server.status || "waiting")}</strong></div>
        <div class="kv-row"><span>Client</span><strong>${escapeHtml(client.status || "live")}</strong></div>
        <div class="kv-row"><span>Turns</span><strong>${userTurns} caller / ${agentTurns} dispatcher</strong></div>
        <div class="kv-row"><span>Updated</span><strong>${escapeHtml(session.updated_at || "n/a")}</strong></div>
      </div>
    `;
  }

  function computeSeverity(session, state) {
    const triageMeta = session.triage_meta || {};
    const merged = session.merged_triage_state || {};
    const handoff = session.handoff_packet || null;
    const callerText = latestTranscriptText(session?.client?.transcripts?.user, 5).toLowerCase();
    const summaryText = String(state.ticketCall?.result?.caller_summary || "").toLowerCase();
    const combinedText = `${callerText} ${summaryText}`;
    const drivers = [];

    let score =
      {
        CRITICAL: 86,
        HIGH: 66,
        MEDIUM: 42,
        LOW: 18,
      }[String(state.priority || merged.priority || "").toUpperCase()] || 8;

    const issueType = String(state.issueType || merged.issue_type || "").toUpperCase();
    if (issueType === "FIRE") {
      score += 8;
      drivers.push("fire risk");
    } else if (issueType === "POLICE" || issueType === "HAZMAT") {
      score += 10;
      drivers.push("active danger");
    } else if (issueType === "MEDICAL") {
      score += 7;
      drivers.push("medical risk");
    } else if (issueType === "TRAFFIC") {
      score += 5;
      drivers.push("traffic injury");
    }

    const keywordDrivers = [
      ["smoke", 12, "smoke reported"],
      ["fire", 14, "fire reported"],
      ["trapped", 12, "people trapped"],
      ["bleeding", 12, "bleeding risk"],
      ["not breathing", 22, "breathing failure"],
      ["breathing", 8, "breathing uncertainty"],
      ["shot", 20, "violence reported"],
      ["shooting", 20, "violence reported"],
      ["child", 14, "child involved"],
      ["daughter", 14, "child involved"],
      ["son", 14, "child involved"],
      ["asthma", 12, "breathing vulnerability"],
      ["water", 18, "water hazard"],
      ["drowning", 24, "drowning risk"],
    ];

    for (const [term, weight, label] of keywordDrivers) {
      if (combinedText.includes(term)) {
        score += weight;
        if (!drivers.includes(label)) drivers.push(label);
      }
    }

    if (handoff || merged.escalate_to_human || triageMeta.autonomy_allowed === false) {
      score = Math.max(score, 72);
      if (!drivers.includes("human handoff advised")) drivers.push("human handoff advised");
    }

    const missing = triageMeta.missing_or_unknown_fields || [];
    if (missing.length && score >= 60) {
      score += Math.min(10, missing.length * 2);
      drivers.push(`${missing.length} unresolved fields`);
    }

    if (!state.hasPin && score >= 50) {
      score += 8;
      drivers.push("location not pinned");
    }

    if (!state.demoSimulation && state.hasPin) {
      drivers.push("location pinned");
    }

    score = Math.max(0, Math.min(100, Math.round(score)));
    const band = severityBand(score);
    const headline = humanTakeoverEngaged ? "Human Takeover Engaged" : severityHeadline(score);
    const subtext = humanTakeoverEngaged
      ? "Operator has manually taken the call surface. Backend automation remains visible for reference."
      : drivers.slice(0, 3).join(" · ") || "Waiting for stable call context.";

    return {
      score,
      band,
      headline,
      subtext,
      drivers,
    };
  }

  function isLowSignalSummary(text) {
    const normalized = String(text || "").trim();
    if (!normalized) return true;
    const compact = normalized.toLowerCase();
    const confirmations = (compact.match(/\b(?:yes|no|maybe|okay|ok|correct|sorry)\b/g) || []).length;
    const uncertainty = (compact.match(/\b(?:i don't know|dont know|not sure)\b/g) || []).length;
    const sentenceCount = normalized
      .split(/[.!?]+/)
      .map((part) => part.trim())
      .filter(Boolean).length;
    if (/(?:\byes\b[,. ]*){2,}|(?:\bno\b[,. ]*){2,}/i.test(compact)) return true;
    if (confirmations + uncertainty >= 4 && normalized.length >= 60) return true;
    if (sentenceCount >= 5 && confirmations >= 2) return true;
    return false;
  }

  function selectStickySummary(handoffBrief, ticket, state) {
    const candidates = [
      handoffBrief.one_line,
      handoffBrief.caller_summary,
      ticket.caller_summary,
    ]
      .map((item) => String(item || "").trim())
      .filter((item) => window.DispatcherSummaryUtils.isEnglishSummary(item));

    const candidate = candidates.find((item) => !isLowSignalSummary(item));
    const summary = candidate || window.DispatcherSummaryUtils.buildEnglishOperatorSummary({
      summaries: [],
      issueType: state.issueType,
      location: state.location,
    });
    if (window.DispatcherSummaryUtils.isEnglishSummary(summary)) stickySummaryText = summary;
    return stickySummaryText || summary;
  }

  function buildOperatorBrief(session, state, severity) {
    const triageMeta = session.triage_meta || {};
    const merged = session.merged_triage_state || {};
    const handoff = session.handoff_packet || null;
    const ticket = state.ticketCall?.result || {};
    const handoffBrief = state.handoffBriefCall?.result || {};
    const summary = state.demoSimulation
      ? `SIMULATION ONLY · ${state.issueType !== "Pending" ? titleCase(state.issueType) : "Synthetic incident"} · configured demo location: ${state.demoSimulation.location?.display_label || "unverified demo location"}.`
      : selectStickySummary(handoffBrief, ticket, state);
    const confirmed = [];
    if (!state.demoSimulation && state.location && state.location !== "Pending") {
      confirmed.push(`Location: ${state.location}`);
    }
    if (!state.demoSimulation && state.hasPin) {
      confirmed.push(`Pinned coordinates: ${state.lat}, ${state.lon}`);
    }
    if (state.issueType && state.issueType !== "Pending") {
      confirmed.push(`Issue: ${titleCase(state.issueType)}`);
    }
    if (state.priority && state.priority !== "Pending") {
      confirmed.push(`Priority: ${state.priority}`);
    }
    if (!state.demoSimulation && state.humanMonitoring && state.monitorName) {
      confirmed.push(`Human monitoring: ${state.monitorName} is watching this call`);
    }
    const serviceItems = state.demoSimulation ? [] : Object.values(state.dispatchServices || {}).filter((item) => item && item.needed);
    if (serviceItems.length) {
      confirmed.push(
        "Services: " +
          serviceItems
            .map((item) => `${item.label} ${String(item.status_label || item.status || "").toLowerCase()} (${item.eta_text})`)
            .join(" · ")
      );
    }

    if (!state.demoSimulation && !confirmed.length && Array.isArray(handoffBrief.confirmed_facts)) {
      confirmed.push(...handoffBrief.confirmed_facts);
    }

    const unknowns = window.DispatcherSummaryUtils.selectOperatorUnknowns({
      triageMissing: triageMeta.missing_or_unknown_fields || [],
      assessmentMissing: session.incident_assessment?.missing_information || [],
      handoffUnknowns: Array.isArray(handoffBrief.unknowns) ? handoffBrief.unknowns : [],
    }).map(humanizeField);
    let nextMove = "A human operator can review this packet and take over using the console.";
    if (humanTakeoverEngaged) {
      nextMove = "Human operator is on the line. Press hand back when you want the agent to resume with a status check.";
    } else
    if (!state.hasPin) {
      nextMove = "Get one landmark or road sign. If the place name sounds unstable, ask the caller to spell it.";
    } else if (state.demoSimulation) {
      nextMove = "Configured demo location and simulated fleet only; no caller location verification or live operator monitoring.";
    } else if (state.humanMonitoring && state.monitorName) {
      nextMove = `${state.monitorName} is watching the call while services continue moving. Keep the caller steady and do not stack questions.`;
    } else if (unknowns.includes("Breathing status")) {
      nextMove = "Confirm breathing with a yes/no question before adding more instructions.";
    } else if (unknowns.includes("Victim count")) {
      nextMove = "Confirm how many people are involved before widening the response.";
    } else if (unknowns.includes("Caller safety")) {
      nextMove = "Confirm whether the caller is currently safe where they are standing.";
    } else if (severity.score >= 86) {
      nextMove = "Immediate human review is recommended. Keep instructions short and use the console takeover control if an operator is available.";
    } else if (handoffBrief.next_question_goal && handoffBrief.next_question_goal !== "hold steady") {
      nextMove = handoffBrief.next_question_goal;
    }

    return {
      summary,
      confirmed,
      unknowns,
      nextMove,
      escalationReason: merged.escalation_reason || handoff?.escalation_reason || "None yet",
      recommendedMode:
        severity.score >= 86
          ? "Holding pattern"
          : severity.score >= 61
            ? "Pre-arrival priority"
            : severity.score >= 31
              ? "Single clear question"
              : "Normal triage",
    };
  }

  function renderOperatorDesk(session, state) {
    const severity = computeSeverity(session, state);
    const brief = buildOperatorBrief(session, state, severity);
    const severityClass = `severity-${severity.band}`;

    severityHeroEl.className = `severity-hero ${severityClass}`;
    severityScoreEl.textContent = `${severity.score} / 100`;
    severitySubtextEl.textContent = severity.subtext;
    const buttonLabel = humanTakeoverEngaged
      ? "Hand Back To Agent"
      : severity.score >= 86
        ? "Take Over Now"
        : severity.score >= 61
          ? "Request Human"
          : severity.score >= 31
            ? "Human Review"
            : "Human Standby";
    takeoverBtn.textContent = buttonLabel;
    takeoverBtn.className = `button takeover-button ${severityClass}${humanTakeoverEngaged ? " engaged" : ""}`;
    takeoverBtn.disabled = !sessionId;

    operatorReportEl.innerHTML = `
      <div class="report-grid">
        <article class="report-card">
          <div class="section-label">Operator Status</div>
          <strong>${escapeHtml(severity.headline)}</strong>
          <p>Heuristic advisory signal only; not clinically validated severity.</p>
          <p>${escapeHtml(brief.nextMove)}</p>
          <p><span class="chip-label">Mode</span> ${escapeHtml(brief.recommendedMode)}</p>
          <p><span class="chip-label">Escalation</span> ${escapeHtml(brief.escalationReason)}</p>
        </article>
        <article class="report-card">
          <div class="section-label">Live Summary</div>
          <strong>${escapeHtml(brief.summary)}</strong>
          <p>English operator summary; caller and dispatcher transcripts remain verbatim.</p>
        </article>
        <article class="report-card">
          <div class="section-label">Confirmed Facts</div>
          ${
            brief.confirmed.length
              ? `<ul>${brief.confirmed.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
              : `<p>No confirmed facts yet.</p>`
          }
        </article>
        <article class="report-card">
          <div class="section-label">Still Unknown</div>
          ${
            brief.unknowns.length
              ? `<ul>${brief.unknowns.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
              : `<p>No major gaps right now.</p>`
          }
        </article>
      </div>
    `;

    dispatchMetricEl.textContent = state.demoSimulation
      ? `SIMULATION · ${state.demoSimulation.stage}`
      : humanTakeoverEngaged ? "Human takeover engaged" : severity.headline;
  }

  function renderServicePanel(state) {
    const simulation = state.demoSimulation;
    if (simulation?.simulation_only) {
      const timeline = (simulation.timeline || []).map((item) => `<li><time>${escapeHtml(item.at || "")}</time> ${escapeHtml(item.stage)} · simulated</li>`).join("");
      servicePanelEl.innerHTML = `
        <aside class="simulation-banner">SIMULATION MODE · NO EMERGENCY SERVICES CONTACTED · DEMO FLEET ONLY</aside>
        <article class="service-card service-card-dispatched">
          <div class="service-top"><strong>${escapeHtml(simulation.selected_call_sign || (simulation.assignment_status === "no_unit_available" ? "No simulated ambulance available" : "Demo ambulance"))}</strong><span class="service-state">${escapeHtml(simulation.stage || "RECEIVED")}</span></div>
          <p>${escapeHtml(simulation.ambulance_status || "assigned")} · route progress ${Math.round(Number(simulation.route_progress || 0) * 100)}%</p>
          <small>${escapeHtml(simulation.route?.label || "SIMULATED ROUTE — NOT ROAD NAVIGATION")} · ${escapeHtml(String(simulation.route?.distance_km ?? "—"))} km straight-line · illustrative ETA ${escapeHtml(String(simulation.route?.eta_minutes ?? "—"))} min</small>
          ${simulation.assignment_status === "no_unit_available" ? "<small>No simulated unit is available. Reset an active demo or wait for completion, then retry.</small>" : ""}
          <small>${escapeHtml(simulation.location?.display_label || "DEMO LOCATION — NOT VERIFIED FROM CALLER")}</small>
          <strong>Simulation timeline</strong><ol class="demo-simulation-timeline">${timeline}</ol>
        </article>`;
      return;
    }
    const services = Object.values(state.dispatchServices || {}).filter((item) => item && item.needed);
    if (!services.length) {
      servicePanelEl.innerHTML = `
        <aside class="simulation-banner">SIMULATED DISPATCH ONLY · No emergency services are contacted.</aside>
        <article class="service-card service-card-pending">
          <div class="service-top">
            <strong>Services</strong>
            <span class="service-state">Pending</span>
          </div>
          <p>No service movement yet.</p>
          <small>The server will identify police, ambulance, or fire once the hard facts are stable enough.</small>
        </article>
      `;
      return;
    }
    servicePanelEl.innerHTML = '<aside class="simulation-banner">SIMULATED DISPATCH ONLY · Statuses and ETAs are illustrative; no emergency services are contacted.</aside>' + services
      .map(
        (service) => `
          <article class="service-card service-card-${escapeHtml(String(service.status || "pending").toLowerCase())}">
            <div class="service-top">
              <strong>${escapeHtml(service.label || "Service")}</strong>
              <span class="service-state">${escapeHtml(service.status_label || titleCase(service.status || "pending"))}</span>
            </div>
            <p>${escapeHtml(service.detail || "No update yet.")}</p>
            <small>${escapeHtml(`${service.unit_label || "Nearest unit"} · ${service.eta_text || "Pending"}`)}</small>
            ${
              service.base_name || service.base_address || service.distance_km != null
                ? `<small>${escapeHtml(
                    [
                      service.base_name,
                      service.base_address,
                      service.distance_km != null ? `${service.distance_km} km straight-line` : null,
                    ]
                      .filter(Boolean)
                      .join(" · ")
                  )}</small>`
                : ""
            }
          </article>
        `
      )
      .join("");
  }

  function summarizeToolCall(toolCall) {
    const { name, args = {}, result = {}, error } = toolCall;
    if (error) return error;

    if (name === "resolve_location_note") {
      return `${result.location_status || "pending"} · ${result.normalized_note || result.sub_location || args.location_note || "pending"}`;
    }
    if (name === "nearby_context") {
      return `${result.resolved ? "context ready" : "pending"} · ${result.summary || result.normalized_address || args.address_text || "pending"}`;
    }
    if (name === "checklist_by_incident") {
      return `${result.next_question_goal || "holding"} · ${(result.missing_fields || []).length} gaps`;
    }
    if (name === "build_handoff_brief") {
      return `${result.status || "draft"} · ${result.one_line || "brief pending"}`;
    }
    if (name === "plan_response_services") {
      return `${(result.needed_services || []).join(", ") || "no services"} · ${result.priority || "pending"}`;
    }
    if (name === "lookup_response_bases") {
      return `${result.resolved ? "bases found" : "no bases"} · ${result.summary || "pending"}`;
    }
    if (name === "simulate_dispatch_services") {
      return `SIMULATION ONLY · ${result.summary || "no service movement"}${result.announcement_line ? ` · ${result.announcement_line}` : ""}`;
    }
    if (name === "assign_dispatch_priority") {
      return `${result.priority || "pending"} · ${result.dispatch_lane || "no lane"}`;
    }
    if (name === "create_incident_ticket") {
      return `LOCAL DEMO TICKET ONLY · ${result.status || "created"} · ${result.priority || args.priority || "pending"} · ${result.address || args.address || "location pending"}`;
    }
    if (name === "validate_address") {
      return `${result.candidate_only ? "candidate" : result.valid ? "valid" : "invalid"} · ${result.normalized_address || args.address_text || "pending"}`;
    }
    if (name === "lookup_address") {
      return `${result.candidate_only ? "candidate" : result.found ? "matched" : "unmatched"} · ${result.normalized_address || args.address_text || "pending"}`;
    }
    return "completed";
  }

  function toolStageDefinitions() {
    return [
      {
        name: "resolve_location_note",
        title: "Location Note",
        eyebrow: "Step 1",
        formatter(toolCall) {
          const result = toolCall?.result || {};
          if (!toolCall) return { status: "pending", summary: "Landmarks, entrances, and floor notes will land here.", meta: "Best-effort clue capture." };
          if (toolCall.error) return { status: "error", summary: toolCall.error, meta: "Location note resolution failed." };
          return {
            status: result.usable_for_dispatch ? "complete" : "watch",
            summary: result.normalized_note || result.sub_location || toolCall.args?.location_note || "No note captured yet.",
            meta: `${titleCase(result.note_type || "none")} · ${titleCase(result.location_status || "missing")}`,
          };
        },
      },
      {
        name: "checklist_by_incident",
        title: "Question Checklist",
        eyebrow: "Step 2",
        formatter(toolCall) {
          const result = toolCall?.result || {};
          if (!toolCall) return { status: "pending", summary: "Gradbot will queue the next hard fact here.", meta: "Highest-priority missing fact." };
          if (toolCall.error) return { status: "error", summary: toolCall.error, meta: "Checklist generation failed." };
          return {
            status: (result.missing_fields || []).length ? "watch" : "complete",
            summary: result.next_question_goal || "Hold steady",
            meta: `${(result.missing_fields || []).length} gaps · ${result.location_search_allowed ? "search allowed" : "search guarded"}`,
          };
        },
      },
      {
        names: ["validate_address", "lookup_address"],
        title: "Location Check",
        eyebrow: "Step 3",
        formatter(toolCall) {
          const result = toolCall?.result || {};
          if (!toolCall) return { status: "pending", summary: "A searchable road, junction, or named place will be checked here.", meta: "Pin and readback." };
          if (toolCall.error) return { status: "error", summary: toolCall.error, meta: "Location check failed." };
          if (result.candidate_only) {
            return {
              status: "watch",
              summary: result.normalized_address || toolCall.args?.address_text || "Candidate location pending confirmation.",
              meta: "Needs caller confirmation",
            };
          }
          const resolved = Boolean(result.valid ?? result.found ?? result.geocoded);
          return {
            status: resolved ? "complete" : "watch",
            summary: result.normalized_address || toolCall.args?.address_text || "No stable location yet.",
            meta:
              toolCall.name === "validate_address"
                ? `${result.valid ? "valid" : "guarded"} · ${titleCase(result.reason || result.source || "pending")}`
                : `${result.found ? "matched" : "guarded"} · ${titleCase(result.reason || result.source || "pending")}`,
          };
        },
      },
      {
        name: "nearby_context",
        title: "Nearby Context",
        eyebrow: "Step 4",
        formatter(toolCall) {
          const result = toolCall?.result || {};
          if (!toolCall) return { status: "pending", summary: "Context appears once a stable place or address lands.", meta: "Roads, landmarks, and area hints." };
          if (toolCall.error) return { status: "error", summary: toolCall.error, meta: "Nearby context failed." };
          return {
            status: result.resolved ? "complete" : "watch",
            summary: result.summary || result.normalized_address || "No context yet.",
            meta: result.resolved ? `${(result.major_roads || []).length} roads · ${(result.landmarks || []).length} landmarks` : titleCase(result.reason || "pending"),
          };
        },
      },
      {
        name: "build_handoff_brief",
        title: "Handoff Brief",
        eyebrow: "Step 5",
        formatter(toolCall) {
          const result = toolCall?.result || {};
          if (!toolCall) return { status: "pending", summary: "A compact operator brief will appear when facts are stable.", meta: "Ready for operations." };
          if (toolCall.error) return { status: "error", summary: toolCall.error, meta: "Brief build failed." };
          return {
            status: result.status === "ready" ? "complete" : result.status ? "watch" : "pending",
            summary: result.one_line || "Facts still being gathered",
            meta: titleCase(result.status || "gathering"),
          };
        },
      },
      {
        name: "lookup_response_bases",
        title: "Nearest Bases",
        eyebrow: "Step 6",
        formatter(toolCall) {
          const result = toolCall?.result || {};
          if (!toolCall) return { status: "pending", summary: "Nearest hospital, police, and fire bases will appear here after a pin lands.", meta: "Grounded dispatch references." };
          if (toolCall.error) return { status: "error", summary: toolCall.error, meta: "Nearest-base lookup failed." };
          return {
            status: result.resolved ? "complete" : "watch",
            summary: result.summary || "No nearby bases found.",
            meta: titleCase(result.source || "pending"),
          };
        },
      },
      {
        name: "plan_response_services",
        title: "Service Plan",
        eyebrow: "Step 7",
        formatter(toolCall) {
          const result = toolCall?.result || {};
          if (!toolCall) return { status: "pending", summary: "Services will be selected once the issue is grounded.", meta: "Police, ambulance, fire." };
          if (toolCall.error) return { status: "error", summary: toolCall.error, meta: "Service plan failed." };
          return {
            status: (result.needed_services || []).length ? "complete" : "watch",
            summary: (result.needed_services || []).map((item) => titleCase(item)).join(" · ") || "No services selected",
            meta: `${result.priority || "Pending"} · ${result.human_monitoring ? "Alex monitoring" : "autonomous"}`,
          };
        },
      },
      {
        name: "simulate_dispatch_services",
        title: "Dispatch Simulation",
        eyebrow: "Step 8",
        formatter(toolCall) {
          const result = toolCall?.result || {};
          if (!toolCall) return { status: "pending", summary: "No live units are connected; illustrative status changes appear here.", meta: "SIMULATION ONLY · No responders contacted." };
          if (toolCall.error) return { status: "error", summary: toolCall.error, meta: "SIMULATION FAILED · No responders contacted." };
          return {
            status: result.dispatchable ? "complete" : "watch",
            summary: result.summary || "No service movement yet.",
            meta: `SIMULATION ONLY · ${result.announcement_line || (result.human_monitoring ? "Human monitor state" : "Waiting for dispatchable facts")} · No responders contacted.`,
          };
        },
      },
    ];
  }

  function renderToolChain(session) {
    const toolCalls = session.tool_calls || [];
    const stages = toolStageDefinitions().map((stage) => {
      const toolCall = stage.names ? latestToolCallAny(toolCalls, stage.names) : latestToolCall(toolCalls, stage.name);
      return { ...stage, toolCall, view: stage.formatter(toolCall) };
    });
    const activeCount = stages.filter((stage) => stage.toolCall && !stage.toolCall.error).length;

    toolChainEl.innerHTML = `
      <div class="tool-chain-shell">
        <div class="tool-chain-head">
          <div>
            <div class="section-label">Gradbot At Work</div>
            <strong>Deterministic Assist Chain</strong>
          </div>
          <span class="tool-chain-badge">${escapeHtml(String(activeCount))} / ${escapeHtml(String(stages.length))} active</span>
        </div>
        <div class="tool-stage-stack">
          ${stages
            .map(
              (stage) => `
                <article class="tool-stage tool-stage-${escapeHtml(stage.view.status)}">
                  <div class="tool-stage-index">${escapeHtml(stage.eyebrow)}</div>
                  <div class="tool-stage-copy">
                    <div class="tool-stage-top">
                      <strong>${escapeHtml(stage.title)}</strong>
                      <span class="tool-stage-state">${escapeHtml(titleCase(stage.view.status))}</span>
                    </div>
                    <p>${escapeHtml(stage.view.summary)}</p>
                    <small>${escapeHtml(stage.view.meta)}</small>
                  </div>
                </article>
              `
            )
            .join("")}
        </div>
      </div>
    `;
  }

  function renderTriagePanel(session) {
    const latestTurn = [...(session.triage_turns || [])].reverse()[0];
    const merged = session.merged_triage_state || {};
    const triageMeta = session.triage_meta || {};
    const serverMode = normalizeDispatchMode(session.server?.live_dispatch_mode || session.server?.triage_engine);
    const overrides = session.rule_overrides || [];
    const handoff = session.handoff_packet;
    const rescueEvents = session.stt_rescue_events || [];
    const latestRescue = rescueEvents.length ? rescueEvents[rescueEvents.length - 1] : null;
    const sttMeta = session.stt_rescue_meta || {};
    if (serverMode === "llm_only") {
      triagePanelEl.innerHTML = `
        <div class="kv-list">
          <div class="kv-row"><span>Mode</span><strong>${escapeHtml(dispatchModeLabel(serverMode))}</strong></div>
          <div class="kv-row"><span>STT Active</span><strong>${escapeHtml(sttMeta.active_provider || "gradium")}</strong></div>
          <div class="kv-row"><span>STT Compare Mode</span><strong>${escapeHtml((sttMeta.continuous_compare ?? false) ? "continuous" : "rescue-only")}</strong></div>
          <div class="kv-row"><span>Shadow Active</span><strong>${escapeHtml(String(sttMeta.shadow_active ?? false))}</strong></div>
          <div class="kv-row"><span>Shadow Elapsed</span><strong>${escapeHtml(String(sttMeta.shadow_elapsed_s ?? 0))} s</strong></div>
        </div>
        <div class="ticket-summary">
          <div class="section-label">LLM Only Path</div>
          <p>Live rescue compare stays on, but structured triage inference is bypassed so you can judge the raw Gradium voice path.</p>
        </div>
        <div class="ticket-summary">
          <div class="section-label">STT Rescue</div>
          ${
            latestRescue
              ? `<pre>${escapeHtml(JSON.stringify(latestRescue, null, 2))}</pre>`
              : `<p>No rescue events yet.</p>`
          }
        </div>
      `;
      return;
    }
    const sharedLedger = session.shared_ledger || null;
    if (serverMode === "slm" && sharedLedger) {
      const hardFacts = sharedLedger.hard_facts || {};
      const softState = sharedLedger.soft_state || {};
      const priority = sharedLedger.priority || {};
      const locationGate = sharedLedger.location_gate || {};
      const provenance = sharedLedger.provenance || {};
      const confirmedFacts = [];
      if ((hardFacts.issue_cues || []).length) {
        confirmedFacts.push(`Issue cues: ${(hardFacts.issue_cues || []).map(titleCase).join(", ")}`);
      }
      if (hardFacts.location_candidate || locationGate.candidate_text) {
        confirmedFacts.push(`Location candidate: ${hardFacts.location_candidate || locationGate.candidate_text}`);
      }
      if (hardFacts.sub_location) {
        confirmedFacts.push(`Sub-location: ${hardFacts.sub_location}`);
      }
      if (hardFacts.victim_count_confirmed !== "unknown") {
        confirmedFacts.push(`Confirmed people count: ${hardFacts.victim_count_confirmed}`);
      }
      ["child_present", "bleeding_status", "breathing_status", "consciousness_status", "trapped_status"].forEach((field) => {
        if (hardFacts[field] && hardFacts[field] !== "unknown") {
          confirmedFacts.push(`${humanizeField(field)}: ${hardFacts[field]}`);
        }
      });
      const softGuesses = [];
      if (softState.people_count_best_guess != null) {
        softGuesses.push(`People count estimate: ${softState.people_count_best_guess}`);
      }
      if (softState.caller_role && softState.caller_role !== "unknown") {
        softGuesses.push(`Caller role: ${softState.caller_role}`);
      }
      (softState.notes || []).forEach((note) => softGuesses.push(note));
      triagePanelEl.innerHTML = `
        <div class="kv-list">
          <div class="kv-row"><span>Mode</span><strong>${escapeHtml(dispatchModeLabel(serverMode))}</strong></div>
          <div class="kv-row"><span>SLM Status</span><strong>${escapeHtml(provenance.slm_job_state || "idle")}</strong></div>
          <div class="kv-row"><span>Ledger Version</span><strong>${escapeHtml(String(sharedLedger.version || 0))}</strong></div>
          <div class="kv-row"><span>Last SLM Run</span><strong>${escapeHtml(provenance.last_slm_run_at || "pending")}</strong></div>
          <div class="kv-row"><span>Next Question</span><strong>${escapeHtml(priority.next_question_goal || "hold steady")}</strong></div>
          <div class="kv-row"><span>Question Style</span><strong>${escapeHtml(priority.question_style || "short_open")}</strong></div>
          <div class="kv-row"><span>Location Gate</span><strong>${escapeHtml(locationGate.search_allowed ? "search allowed" : "search guarded")}</strong></div>
          <div class="kv-row"><span>Location Next Step</span><strong>${escapeHtml(priority.location_followup_kind || "none")}</strong></div>
        </div>
        <div class="ticket-summary">
          <div class="section-label">Confirmed Hard Facts</div>
          ${
            confirmedFacts.length
              ? `<ul>${confirmedFacts.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
              : `<p>No confirmed hard facts yet.</p>`
          }
        </div>
        <div class="ticket-summary">
          <div class="section-label">Soft Estimates</div>
          ${
            softGuesses.length
              ? `<ul>${softGuesses.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
              : `<p>No soft estimates yet.</p>`
          }
        </div>
        <div class="ticket-summary">
          <div class="section-label">Still Missing</div>
          ${
            (priority.missing_fields || []).length
              ? `<p>${escapeHtml((priority.missing_fields || []).map(humanizeField).join(", "))}</p>`
              : `<p>No major hard-fact gaps right now.</p>`
          }
        </div>
        <div class="ticket-summary">
          <div class="section-label">Location State</div>
          <p>${escapeHtml(
            [
              locationGate.search_reason || "pending",
              locationGate.candidate_text || null,
              locationGate.candidate_only ? "candidate only" : null,
              locationGate.needs_confirmation ? "needs confirmation" : null,
              locationGate.confirmed ? "confirmed" : null,
            ]
              .filter(Boolean)
              .join(" · ")
          )}</p>
        </div>
        <div class="ticket-summary">
          <div class="section-label">Location Follow-Up</div>
          <p>${escapeHtml(priority.location_followup_prompt || "No location follow-up queued.")}</p>
        </div>
      `;
      return;
    }
    if (!latestTurn && !Object.keys(merged).length) {
      triagePanelEl.innerHTML = `
        <article class="entry event">
          <div class="body">No triage state yet.</div>
        </article>
      `;
      return;
    }
    triagePanelEl.innerHTML = `
      <div class="kv-list">
        <div class="kv-row"><span>Engine</span><strong>${escapeHtml(triageMeta.selected_engine || latestTurn?.selected_engine || "pending")}</strong></div>
        <div class="kv-row"><span>Runtime</span><strong>${escapeHtml(triageMeta.runtime_provider || latestTurn?.runtime_provider || "pending")}</strong></div>
        <div class="kv-row"><span>Version</span><strong>${escapeHtml(String(triageMeta.version || latestTurn?.version || 0))}</strong></div>
        <div class="kv-row"><span>Dispatchable</span><strong>${escapeHtml(String(triageMeta.dispatchable ?? latestTurn?.dispatchable ?? false))}</strong></div>
        <div class="kv-row"><span>Autonomy Allowed</span><strong>${escapeHtml(String(triageMeta.autonomy_allowed ?? latestTurn?.autonomy_allowed ?? false))}</strong></div>
        <div class="kv-row"><span>STT Active</span><strong>${escapeHtml(sttMeta.active_provider || "gradium")}</strong></div>
        <div class="kv-row"><span>STT Compare Mode</span><strong>${escapeHtml((sttMeta.continuous_compare ?? false) ? "continuous" : "rescue-only")}</strong></div>
        <div class="kv-row"><span>STT Elapsed</span><strong>${escapeHtml(String(sttMeta.shadow_elapsed_s ?? 0))} s</strong></div>
      </div>
      <div class="ticket-summary">
        <div class="section-label">Latest Triage Delta</div>
        <pre>${escapeHtml(JSON.stringify(latestTurn?.latest_triage_delta || {}, null, 2))}</pre>
      </div>
      <div class="ticket-summary">
        <div class="section-label">Merged Triage State</div>
        <pre>${escapeHtml(JSON.stringify(merged, null, 2))}</pre>
      </div>
      <div class="ticket-summary">
        <div class="section-label">Missing / Unknown Fields</div>
        <p>${escapeHtml((triageMeta.missing_or_unknown_fields || latestTurn?.missing_or_unknown_fields || []).join(", ") || "None")}</p>
      </div>
      <div class="ticket-summary">
        <div class="section-label">Rule Overrides</div>
        ${
          overrides.length
            ? `<pre>${escapeHtml(JSON.stringify(overrides, null, 2))}</pre>`
            : `<p>No hard overrides triggered.</p>`
        }
      </div>
      <div class="ticket-summary">
        <div class="section-label">STT Rescue</div>
        ${
          latestRescue
            ? `<pre>${escapeHtml(JSON.stringify(latestRescue, null, 2))}</pre>`
            : `<p>No rescue events yet.</p>`
        }
      </div>
      <div class="ticket-summary">
        <div class="section-label">STT Provider Scores</div>
        ${
          sttMeta.provider_stats
            ? `<pre>${escapeHtml(JSON.stringify(sttMeta.provider_stats, null, 2))}</pre>`
            : `<p>No provider scores yet.</p>`
        }
      </div>
      <div class="ticket-summary">
        <div class="section-label">STT Switches</div>
        ${
          sttMeta.switch_events && sttMeta.switch_events.length
            ? `<pre>${escapeHtml(JSON.stringify(sttMeta.switch_events, null, 2))}</pre>`
            : `<p>No provider switch yet.</p>`
        }
      </div>
      <div class="ticket-summary">
        <div class="section-label">Handoff Packet</div>
        ${
          handoff
            ? `<pre>${escapeHtml(JSON.stringify(handoff, null, 2))}</pre>`
            : `<p>No handoff packet yet.</p>`
        }
      </div>
    `;
  }

  function latestToolCall(toolCalls, name) {
    return [...toolCalls].reverse().find((item) => item.name === name);
  }

  function latestToolCallAny(toolCalls, names) {
    const allowed = new Set(names);
    return [...toolCalls].reverse().find((item) => allowed.has(item.name));
  }

  function transcriptTurns(session, role) {
    return session?.client?.transcripts?.[role] || [];
  }

  function combinedTranscriptText(session, role) {
    return transcriptTurns(session, role)
      .map((item) => item.text)
      .filter(Boolean)
      .join(" ")
      .trim();
  }

  function inferTranscriptIssue(session) {
    const text = `${combinedTranscriptText(session, "user")} ${combinedTranscriptText(session, "agent")}`.toLowerCase();
    if ((text.includes("daughter") || text.includes("son") || text.includes("child")) && (text.includes("lost") || text.includes("cannot find") || text.includes("can't find") || text.includes("missing"))) {
      return "Missing child";
    }
    if (text.includes("not breathing") || text.includes("unconscious")) return "Medical emergency";
    if (text.includes("fire") || text.includes("smoke")) return "Fire";
    if (text.includes("bleeding") || text.includes("injured")) return "Medical emergency";
    if (text.includes("crash") || text.includes("accident") || text.includes("collision")) return "Traffic incident";
    if (text.includes("weapon") || text.includes("gun") || text.includes("knife") || text.includes("hostage")) return "Police incident";
    return "Pending";
  }

  function inferTranscriptPriority(session, issueLabel) {
    const text = `${combinedTranscriptText(session, "user")} ${combinedTranscriptText(session, "agent")}`.toLowerCase();
    if (issueLabel === "Missing child") return "HIGH";
    if (text.includes("not breathing") || text.includes("unconscious") || text.includes("fire") || text.includes("hostage")) return "CRITICAL";
    if (text.includes("bleeding") || text.includes("trapped") || text.includes("weapon")) return "HIGH";
    if (issueLabel !== "Pending") return "MEDIUM";
    return "Pending";
  }

  function dispatchLaneForPriority(priority) {
    const value = String(priority || "").toUpperCase();
    return {
      CRITICAL: "lights_and_sirens",
      HIGH: "urgent",
      MEDIUM: "standard",
      LOW: "queued",
    }[value] || "Pending";
  }

  function lastSubstantiveLocationClue(session) {
    const userTurns = transcriptTurns(session, "user");
    const locationPattern = /\b(station|bahnhof|park|platz|playground|campus|hospital|metro|street|strasse|straße|road|junction|entrance|berlin)\b/i;
    for (let index = userTurns.length - 1; index >= 0; index -= 1) {
      const text = String(userTurns[index]?.text || "").trim();
      if (text.length >= 6 && locationPattern.test(text)) {
        return text.replace(/\s+/g, " ");
      }
    }
    return null;
  }

  function lastConfirmedAgentLocation(session) {
    const agentTurns = transcriptTurns(session, "agent");
    const userTurns = transcriptTurns(session, "user");
    const yesPattern = /^(yes|yeah|yep|correct|that'?s right)\b/i;
    for (let index = agentTurns.length - 1; index >= 0; index -= 1) {
      const agentTurn = agentTurns[index];
      const agentText = String(agentTurn?.text || "").trim();
      let candidate = null;
      let match = agentText.match(/Did you mean (.+?)\?/i);
      if (match) {
        candidate = match[1].trim();
      } else {
        match = agentText.match(/^(.+?)\.\s*Is that right\?/i);
        if (match) candidate = match[1].trim();
      }
      if (!candidate) continue;
      const agentAt = agentTurn?.at ? Date.parse(agentTurn.at) : NaN;
      for (let userIndex = userTurns.length - 1; userIndex >= 0; userIndex -= 1) {
        const userTurn = userTurns[userIndex];
        const userText = String(userTurn?.text || "").trim();
        if (!yesPattern.test(userText)) continue;
        const userAt = userTurn?.at ? Date.parse(userTurn.at) : NaN;
        if (
          Number.isFinite(agentAt) &&
          Number.isFinite(userAt) &&
          Math.abs(userAt - agentAt) <= 2000
        ) {
          return candidate;
        }
        if (Number.isFinite(agentAt) && Number.isFinite(userAt) && userAt >= agentAt && userAt - agentAt <= 12000) {
          return candidate;
        }
      }
    }
    return null;
  }

  function transcriptLocationFallback(session) {
    const confirmed = lastConfirmedAgentLocation(session);
    if (confirmed) {
      return { text: confirmed, confidence: "confirmed by caller", source: "dialogue_confirmation" };
    }
    const clue = lastSubstantiveLocationClue(session);
    if (clue) {
      return { text: clue, confidence: "spoken clue", source: "caller_transcript" };
    }
    return { text: null, confidence: null, source: null };
  }

  function deriveDashboardState(session) {
    const serverMode = normalizeDispatchMode(session.server?.live_dispatch_mode || session.server?.triage_engine);
    const toolCalls = session.tool_calls || [];
    const dispatchServicesState = session.dispatch_services || {};
    const demoSimulation = session.demo_simulation?.simulation_only ? session.demo_simulation : null;
    const assessmentLocation = session.incident_assessment?.location || {};
    const sharedLedger = session.shared_ledger || {};
    const hardFacts = sharedLedger.hard_facts || {};
    const locationGate = sharedLedger.location_gate || {};
    const resolveNoteCall = latestToolCall(toolCalls, "resolve_location_note");
    const nearbyContextCall = latestToolCall(toolCalls, "nearby_context");
    const checklistCall = latestToolCall(toolCalls, "checklist_by_incident");
    const handoffBriefCall = latestToolCall(toolCalls, "build_handoff_brief");
    const ticketCall = latestToolCall(toolCalls, "create_incident_ticket");
    const validateCall = latestToolCall(toolCalls, "validate_address");
    const lookupCall = latestToolCall(toolCalls, "lookup_address");
    const validatedLocation = validateCall?.result?.candidate_only ? null : validateCall?.result || null;
    const lookedUpLocation = lookupCall?.result?.candidate_only ? null : lookupCall?.result || null;
    const candidateLocationResult =
      validateCall?.result?.candidate_only ? validateCall?.result : lookupCall?.result?.candidate_only ? lookupCall?.result : null;
    const approximateMapLocation =
      lookupCall?.result?.map_approximation && lookupCall.result.candidate_only
        ? lookupCall.result
        : null;
    const ticketResult = ticketCall?.result || {};
    const ticketArgs = ticketCall?.args || {};
    const transcriptIssue = inferTranscriptIssue(session);
    const transcriptLocation = transcriptLocationFallback(session);

    const priority =
      ticketResult.priority ||
      inferTranscriptPriority(session, transcriptIssue) ||
      "Pending";
    const issueType =
      ticketResult.issue_type ||
      ticketArgs.issue_type ||
      ((hardFacts.issue_cues || []).length ? (hardFacts.issue_cues || []).map(titleCase).join(", ") : null) ||
      transcriptIssue ||
      "Pending";
    const location = demoSimulation?.location?.label || assessmentLocation.candidate ||
      ticketResult.address ||
      nearbyContextCall?.result?.normalized_address ||
      validatedLocation?.normalized_address ||
      lookedUpLocation?.normalized_address ||
      approximateMapLocation?.normalized_address ||
      hardFacts.location_candidate ||
      locationGate.candidate_text ||
      hardFacts.location_note ||
      resolveNoteCall?.result?.anchor_location ||
      transcriptLocation.text ||
      resolveNoteCall?.result?.normalized_note ||
      candidateLocationResult?.normalized_address ||
      validateCall?.args?.address_text ||
      lookupCall?.args?.address_text ||
      "Pending";
    const lat =
      nearbyContextCall?.result?.lat ??
      validatedLocation?.lat ??
      lookedUpLocation?.lat ??
      approximateMapLocation?.lat ??
      demoSimulation?.location?.lat ??
      assessmentLocation.latitude ??
      null;
    const lon =
      nearbyContextCall?.result?.lon ??
      validatedLocation?.lon ??
      lookedUpLocation?.lon ??
      approximateMapLocation?.lon ??
      demoSimulation?.location?.lon ??
      assessmentLocation.longitude ??
      null;
    const ticketStatus = ticketResult.status || "Not created";
    const dispatchLane = dispatchLaneForPriority(priority);
    const toolCount = toolCalls.length;
    const mapCandidates =
      nearbyContextCall?.result?.candidates ||
      lookedUpLocation?.candidates ||
      validatedLocation?.candidates ||
      [];
    const mapSource =
      nearbyContextCall?.result?.reason ||
      validatedLocation?.source ||
      lookedUpLocation?.source ||
      approximateMapLocation?.source ||
      demoSimulation?.location?.source || assessmentLocation.source ||
      (serverMode === "slm" ? locationGate.search_reason : null) ||
      candidateLocationResult?.source ||
      transcriptLocation.source ||
      null;
    const mapConfidence =
      validatedLocation?.confidence ||
      lookedUpLocation?.confidence ||
      approximateMapLocation?.confidence ||
      (demoSimulation?.location?.confidence ? `${demoSimulation.location.confidence}${demoSimulation.location.source === "demo_configuration" ? " · demo" : ""}` : null) ||
      assessmentLocation.confidence ||
      (serverMode === "slm" ? (locationGate.confirmed ? "confirmed" : locationGate.candidate_only ? "needs confirmation" : null) : null) ||
      (candidateLocationResult ? "needs confirmation" : null) ||
      transcriptLocation.confidence ||
      null;
    const mapReason =
      assessmentLocation.error_category ||
      validatedLocation?.reason ||
      lookedUpLocation?.reason ||
      approximateMapLocation?.reason ||
      (demoSimulation?.location?.resolution_quality || assessmentLocation.resolution_quality || (demoSimulation ? "demo_location_not_verified_from_caller" : null)) ||
      (resolveNoteCall?.result?.requires_city_clarification ? "ambiguous_location_without_city" : null) ||
      candidateLocationResult?.reason ||
      null;
    const hasPin = lat != null && lon != null;

    const labels = [
      ["Issue", issueType !== "Pending" ? issueType : null],
      ["Lane", dispatchLane !== "Pending" ? dispatchLane : null],
      ["Next", checklistCall?.result?.next_question_goal],
      ["Handoff", handoffBriefCall?.result?.status],
      ["Geo", mapConfidence || (hasPin ? "pinned" : "unresolved")],
    ].filter(([, value]) => value);

    return {
      priority,
      issueType,
      location,
      lat,
      lon,
      ticketStatus,
      dispatchLane,
      toolCount,
      mapCandidates,
      mapSource,
      mapConfidence,
      mapReason,
      mapRevision: Number(session.client?.report_revision || 0),
      locationNeedsConfirmation: Boolean(candidateLocationResult || resolveNoteCall?.result?.requires_city_clarification),
      hasPin,
      labels,
      dispatchServices: demoSimulation ? {} : dispatchServicesState.services || {},
      demoSimulation,
      incidentAssessmentLocation: assessmentLocation,
      dispatchSummary: dispatchServicesState.summary || null,
      humanMonitoring: demoSimulation ? false : Boolean(dispatchServicesState.human_monitoring),
      monitorName: demoSimulation ? null : dispatchServicesState.monitor_name || null,
      seriousEmergency: Boolean(dispatchServicesState.serious_emergency),
      dispatchAnnouncement: dispatchServicesState.announcement_line || null,
      serverMode,
      sharedLedger,
      resolveNoteCall,
      nearbyContextCall,
      checklistCall,
      handoffBriefCall,
      ticketCall,
      toolCalls,
      serverStatus: session.server?.status || "waiting",
    };
  }

  function renderSnapshot(session) {
    const state = deriveDashboardState(session);
    dispatchMetricEl.textContent = state.ticketStatus;
    snapshotGridEl.innerHTML = `
      <article class="snapshot-card priority-${String(state.priority).toLowerCase()}">
        <span class="snapshot-label">Priority</span>
        <strong>${escapeHtml(state.priority)}</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Issue</span>
        <strong>${escapeHtml(state.issueType)}</strong>
      </article>
      <article class="snapshot-card location-card">
        <span class="snapshot-label">Location</span>
        <strong>${escapeHtml(state.location)}</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Pin</span>
        <strong>${escapeHtml(state.hasPin ? "Landed" : "Unresolved")}</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Geo Confidence</span>
        <strong>${escapeHtml(state.mapConfidence || "Pending")}</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Dispatch Lane</span>
        <strong>${escapeHtml(state.dispatchLane)}</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Demo Ticket</span>
        <strong>${escapeHtml(state.ticketStatus)}</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Tool Calls</span>
        <strong>${escapeHtml(String(state.toolCount))}</strong>
      </article>
    `;
    updateMap(state);
    renderToolChain(session);
    renderServicePanel(state);

    if (!state.labels.length) {
      labelChipsEl.innerHTML = `<span class="chip pending">Waiting for labels</span>`;
    } else {
      labelChipsEl.innerHTML = state.labels
        .map(
          ([label, value]) =>
            `<span class="chip"><span class="chip-label">${escapeHtml(label)}</span>${escapeHtml(value)}</span>`
        )
        .join("");
    }

    if (!state.ticketCall) {
      ticketPanelEl.innerHTML = `
        <article class="entry event">
          <div class="body">No incident ticket yet.</div>
        </article>
      `;
    } else {
      const ticket = state.ticketCall.result || {};
      const notes = (ticket.notes || []).map((note) => `<li>${escapeHtml(note)}</li>`).join("");
      ticketPanelEl.innerHTML = `
        <aside class="simulation-banner">LOCAL DEMO TICKET ONLY · This record is not sent to a dispatch service.</aside>
        <div class="kv-list">
          <div class="kv-row"><span>Status</span><strong>${escapeHtml(ticket.status || "pending")}</strong></div>
          <div class="kv-row"><span>Priority</span><strong>${escapeHtml(ticket.priority || "pending")}</strong></div>
          <div class="kv-row"><span>Issue Type</span><strong>${escapeHtml(ticket.issue_type || "pending")}</strong></div>
          <div class="kv-row"><span>Address</span><strong>${escapeHtml(ticket.address || "pending")}</strong></div>
          <div class="kv-row"><span>Schema</span><strong>${escapeHtml(ticket.schema_version || "n/a")}</strong></div>
        </div>
        <div class="ticket-summary">
          <div class="section-label">Caller Summary</div>
          <p>${escapeHtml(ticket.caller_summary || "Not available yet.")}</p>
        </div>
        <div class="ticket-summary">
          <div class="section-label">Notes</div>
          ${notes ? `<ul class="note-list">${notes}</ul>` : `<p>No notes yet.</p>`}
        </div>
      `;
    }

    if (!state.toolCalls.length) {
      toolLogEl.innerHTML = `
        <article class="entry event">
          <div class="body">No tool calls yet.</div>
        </article>
      `;
      return;
    }

    toolLogEl.innerHTML = [...state.toolCalls]
      .reverse()
      .map((toolCall) => {
        const result = toolCall.result || {};
        const error = toolCall.error;
        return `
          <article class="tool-entry ${error ? "error" : "ok"}">
            <div class="tool-head">
              <div>
                <strong>${escapeHtml(toolCall.name)}</strong>
                <p>${escapeHtml(summarizeToolCall(toolCall))}</p>
              </div>
              <span class="tool-at">${escapeHtml(toolCall.at || "")}</span>
            </div>
            <details>
              <summary>Inspect payload</summary>
              <div class="tool-payload">
                <div>
                  <div class="section-label">Args</div>
                  <pre>${escapeHtml(JSON.stringify(toolCall.args || {}, null, 2))}</pre>
                </div>
                <div>
                  <div class="section-label">${error ? "Error" : "Result"}</div>
                  <pre>${escapeHtml(JSON.stringify(error || result, null, 2))}</pre>
                </div>
              </div>
            </details>
          </article>
        `;
      })
      .join("");
  }

  function renderEmergencyAssessment(session) {
    if (!emergencyAssessmentPanelEl) return;
    const assessment = session.incident_assessment;
    const natural = assessment?.natural_language_classification || session.natural_language_classification;
    let naturalSummary;
    if (!natural || natural.status === "awaiting") {
      naturalSummary = "Awaiting finalized caller description.";
    } else if (natural.status === "pending") {
      naturalSummary = "Processing finalized caller description asynchronously; rule-based safety guidance continues independently.";
    } else if (natural.status === "unavailable") {
      const fallbackCategory = natural.rule_based_fallback?.category || assessment?.incident_type?.value;
      naturalSummary = "MODEL UNAVAILABLE · " + (natural.uncertainty_reason || "prototype artifact unavailable") + "." + (fallbackCategory && fallbackCategory !== "Unknown" ? " Rule-based demo fallback — not an ML prediction: " + escapeHtml(fallbackCategory) + "." : " Rule-based assessment remains separate.");
    } else if (natural.status === "insufficient_input") {
      const fallbackCategory = natural.rule_based_fallback?.category || assessment?.incident_type?.value;
      naturalSummary = "INSUFFICIENT INPUT · no incident content detected." + (fallbackCategory && fallbackCategory !== "Unknown" ? " Rule-based demo fallback — not an ML prediction: " + escapeHtml(fallbackCategory) + "." : " Awaiting a relevant finalized caller description.");
    } else if (natural.status === "error") {
      const fallbackCategory = natural.rule_based_fallback?.category || assessment?.incident_type?.value;
      naturalSummary = "MODEL ERROR · " + escapeHtml(natural.uncertainty_reason || "inference failed") + "." + (fallbackCategory && fallbackCategory !== "Unknown" ? " Rule-based demo fallback — not an ML prediction: " + escapeHtml(fallbackCategory) + "." : " Existing rule-based assessment remains separate.");
    } else if (natural.status === "classified") {
      const score = Number(natural.class_scores?.[natural.predicted_category]);
      const scoreLabel = Number.isFinite(score) ? ` · uncalibrated model score ${score.toFixed(4)} (not confidence)` : "";
      const prototypeLabel = natural.prototype_only ? "Synthetic-data prototype only · not clinically validated" : "Advisory classification";
      naturalSummary = `ML PREDICTION COMPLETE · ${escapeHtml(prototypeLabel)} · model ${escapeHtml(natural.model_id || "unavailable")} ${escapeHtml(natural.model_version || "")} · category: ${escapeHtml(natural.predicted_category || "not returned")}${scoreLabel} · human review required.`;
    } else if (natural.status === "abstained") {
      naturalSummary = `Uncertain — ${escapeHtml(natural.uncertainty_reason || "insufficient supported information")}.`;
    } else {
      naturalSummary = `Classification ${escapeHtml(natural.status || "unavailable")} — ${escapeHtml(natural.uncertainty_reason || "no result returned")}.`;
    }
    const ml = currentLiveMlUpdate || { status: "awaiting" };
    let mlSummary;
    if (ml.status === "success") {
      const prediction = ml.payload?.prediction || {};
      const score = Number(ml.payload?.uncalibrated_model_score);
      mlSummary = `<strong>${escapeHtml(prediction.chapter || prediction.category || "Category unavailable")}</strong><span>Code ${escapeHtml(ml.code)} · training ${escapeHtml(ml.payload?.training_coverage || "unknown")}${Number.isFinite(score) ? ` · uncalibrated score ${escapeHtml(score.toFixed(4))}` : " · score not provided"}</span>`;
    } else if (ml.status === "pending") {
      mlSummary = `<span>ML classification pending for explicitly stated complaint code ${escapeHtml(ml.code || "")}.</span>`;
    } else if (ml.status === "unavailable" || ml.status === "error") {
      mlSummary = `<span>ML result ${escapeHtml(ml.status)} · ${escapeHtml(ml.message || "The classifier did not return a result.")}</span>`;
    } else {
      mlSummary = "<span>Not applicable — complaint code unavailable. This classifier does not map ordinary symptom descriptions to codes.</span>";
    }
    if (!assessment) {
      emergencyAssessmentPanelEl.innerHTML = `
        <div class="assessment-live-status" role="status" aria-live="polite">${escapeHtml(emergencyAssessmentStatus)}</div>
        <p>Awaiting a finalized caller utterance. Unknown information stays marked unknown.</p>
        <section class="assessment-ml" aria-label="Natural-language incident classifier"><strong>Natural-language incident classification</strong><span>${naturalSummary}</span><small>Separate from complaint-code classification · trained on 14 authored synthetic demo scenarios only · not clinically validated · no diagnosis, severity prediction, or dispatch authorization.</small></section>
        <section class="assessment-ml" aria-label="Live complaint-code classification"><strong>ML complaint-code result</strong>${mlSummary}<small>Broad code classification only · not diagnosis or severity · advisory · human review required</small></section>`;
      return;
    }
    const renderFact = (fact) => `
      <li>
        <strong>${escapeHtml(fact.field)}:</strong> ${escapeHtml(fact.value)}
        <span class="assessment-evidence">${escapeHtml(String(fact.evidence_status || "unverified").replaceAll("_", " "))}</span>
        ${fact.source_utterance ? `<details><summary>Caller source${fact.source_at ? ` · ${escapeHtml(fact.source_at)}` : ""}</summary><q>${escapeHtml(fact.source_utterance)}</q></details>` : ""}
      </li>`;
    const services = Object.values(assessment.dispatch?.services || {}).filter((service) => service && service.needed);
    const simulation = session.demo_simulation?.simulation_only ? session.demo_simulation : null;
    const dispatchPlan = simulation
      ? `<p>SIMULATION · ${escapeHtml(simulation.stage || "in progress")} · ${escapeHtml(simulation.selected_call_sign || "No demo unit assigned")}</p><small>${escapeHtml(simulation.route?.label || "SIMULATED ROUTE — NOT ROAD NAVIGATION")} · illustrative ETA ${escapeHtml(String(simulation.route?.eta_minutes ?? "—"))} min</small>`
      : `${services.length ? `<ul>${services.map((service) => `<li>${escapeHtml(service.label || "Service")}: ${escapeHtml(service.status_label || service.status || "pending")} · ${escapeHtml(service.eta_text || "No ETA")}${service.distance_km != null ? ` · ${escapeHtml(String(service.distance_km))} km straight-line` : ""}</li>`).join("")}</ul>` : "<small>No simulated responder recommendation yet.</small>"}`;
    const aid = assessment.first_aid;
    const sources = (aid?.sources || []).map((source) => `<a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.organization)} · ${escapeHtml(source.title)}</a>`).join(" · ");
    const location = simulation
      ? { ...(assessment.location || {}), candidate: simulation.location?.display_label || "Configured demo location", status: "simulation only", confidence: "low" }
      : assessment.location || {};
    const missingInformation = assessment.missing_information || [];
    const nextQuestions = assessment.next_questions || [];
    const position = simulation && simulation.location?.lat != null && simulation.location?.lon != null
      ? `configured demo coordinates ${Number(simulation.location.lat).toFixed(5)}, ${Number(simulation.location.lon).toFixed(5)} · not caller verified`
      : location.latitude != null && location.longitude != null
        ? `${Number(location.latitude).toFixed(5)}, ${Number(location.longitude).toFixed(5)}`
      : "No verified coordinates";
    emergencyAssessmentPanelEl.innerHTML = `
      <div class="assessment-topline">
        <span>Incident ${escapeHtml(assessment.incident_id || assessment.session_id || "unknown")} · ${escapeHtml(emergencyAssessmentStatus)} · ${escapeHtml(String(assessment.extraction_status || "partial").replaceAll("_", " "))}</span>
        <time>${escapeHtml(assessment.generated_at || "")}</time>
      </div>
      <section class="assessment-ml" aria-label="Natural-language incident classifier"><strong>Natural-language incident classification</strong><span>${naturalSummary}</span><small>Separate from complaint-code classification · trained on 14 authored synthetic demo scenarios only · not clinically validated · no diagnosis, severity prediction, or dispatch authorization.</small></section>
      <div class="assessment-grid">
        <article><span>Incident type</span><strong>${escapeHtml(assessment.incident_type?.value || "Unknown")}</strong><small>${escapeHtml(assessment.incident_type?.evidence_status || "unknown")}</small></article>
        <article><span>Advisory urgency</span><strong>${escapeHtml(assessment.urgency_recommendation?.value || "Unknown")}</strong><small>Human review required · rule/model output is not a diagnosis</small></article>
        <article><span>Location</span><strong>${escapeHtml(location.candidate || "Unknown")}</strong><small>${escapeHtml(titleCase(location.status || "unresolved"))}${location.confidence ? ` · ${escapeHtml(location.confidence)} confidence` : ""} · ${escapeHtml(position)}</small>${location.clarification ? `<small>${escapeHtml(location.clarification)}</small>` : ""}</article>
      </div>
      <div class="assessment-columns">
        <div><strong>Caller-reported concerns</strong><ul>${(assessment.symptoms || []).length ? assessment.symptoms.map(renderFact).join("") : "<li>Unknown; no supported concern extracted yet.</li>"}</ul></div>
        <div><strong>Known facts and unknowns</strong><ul>${(assessment.facts || []).map(renderFact).join("")}</ul></div>
      </div>
      <div class="assessment-columns">
        <div><strong>Missing information</strong><ul>${missingInformation.length ? missingInformation.map((item) => `<li>${escapeHtml(humanizeField(item))}</li>`).join("") : "<li>No outstanding field in the current rules ledger.</li>"}</ul><strong>Next question focus</strong><p>${nextQuestions.length ? escapeHtml(nextQuestions[0]) : "No follow-up question selected yet."}</p></div>
        <div><strong>Safety guidance · ${escapeHtml(aid?.detected_scenario?.label || "Awaiting caller details")}</strong>${aid ? `<p>${escapeHtml(aid.emergency_escalation?.message || "Advisory only; follow the local emergency call-taker.")}</p><ul>${(aid.immediate_actions || []).slice(0, 3).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul><small>${sources || "No source citation available."}</small>` : "<p>Not available until a caller utterance is received.</p>"}</div>
      </div>
      <section class="assessment-ml" aria-label="Live complaint-code classification"><strong>ML complaint-code result</strong>${mlSummary}<small>Broad code classification only · score is not a probability of severity · not diagnosis or dispatch authorization · advisory · human review required</small></section>
      <div class="assessment-dispatch"><strong>Responder planning</strong><span>SIMULATION ONLY · no services contacted · operator confirmation required</span><p>${escapeHtml(simulation ? "Simulation lifecycle" : assessment.dispatch?.status || "No simulated plan yet.")}</p>${dispatchPlan}</div>
      <div class="assessment-footnote">Evidence source: ${escapeHtml(assessment.extraction_source || "unknown")} · updated ${escapeHtml(assessment.generated_at || "")}</div>`;
  }

  async function refreshSessionDetail() {
    if (!sessionId) return;
    const requestedSessionId = sessionId;
    const requestVersion = ++sessionDetailRequestVersion;
    let session;
    try {
      const response = await fetch(`/api/sessions/${requestedSessionId}`);
      if (!response.ok) throw new Error(`Session refresh failed (${response.status}).`);
      session = await response.json();
    } catch (error) {
      if (requestedSessionId === sessionId && requestVersion === sessionDetailRequestVersion) {
        emergencyAssessmentStatus = "Assessment refresh failed · session state unavailable";
        renderEmergencyAssessment(latestSession || {});
      }
      throw error;
    }
    if (requestVersion !== sessionDetailRequestVersion || requestedSessionId !== sessionId) return;
    const activeProvider = session?.stt_rescue_meta?.active_provider || "gradium";
    if (activeProvider !== lastActiveTranscriptProvider) {
      logEvent(`stt provider switched ${lastActiveTranscriptProvider} -> ${activeProvider}`);
      lastActiveTranscriptProvider = activeProvider;
    }
    latestSession = session;
    if (session.demo_simulation?.simulation_only) {
      emergencyAssessmentStatus = `Simulation · ${session.demo_simulation.stage || "in progress"}`;
    } else if (session.incident_assessment && emergencyAssessmentStatus !== "Processing transcript" && emergencyAssessmentStatus !== "Updating assessment") {
      emergencyAssessmentStatus = "Assessment updated";
    }
    const dashboardState = deriveDashboardState(session);
    renderSnapshot(session);
    renderEmergencyAssessment(session);
    renderOperatorDesk(session, dashboardState);
    renderTriagePanel(session);
    setMemoryPanel(session);
  }

  function startBackgroundLoops() {
    if (sessionPollHandle) clearInterval(sessionPollHandle);
    if (syncHandle) clearInterval(syncHandle);

    sessionPollHandle = setInterval(() => {
      refreshSessionDetail().catch(() => {});
    }, 1200);

    syncHandle = setInterval(() => {
      syncSessionReport("live").catch(() => {});
    }, 2000);
  }

  function stopBackgroundLoops() {
    sessionDetailRequestVersion += 1;
    if (sessionPollHandle) {
      clearInterval(sessionPollHandle);
      sessionPollHandle = null;
    }
    if (syncHandle) {
      clearInterval(syncHandle);
      syncHandle = null;
    }
  }

  function resetDashboard() {
    sessionDetailRequestVersion += 1;
    emergencyAssessmentStatus = "Awaiting additional information";
    renderEmergencyAssessment({});
    resetFirstAidPanel();
    clearProviderTimeout();
    sessionMetricEl.textContent = "Not started";
    dispatchMetricEl.textContent = "Waiting";
    severityHeroEl.className = "severity-hero severity-green";
    severityScoreEl.textContent = "0 / 100";
    severitySubtextEl.textContent = "Waiting for a live incident.";
    takeoverBtn.textContent = "Human Standby";
    takeoverBtn.className = "button takeover-button severity-green";
    takeoverBtn.disabled = true;
    operatorReportEl.innerHTML = `
      <article class="entry event">
        <div class="body">Start a call to populate the human handoff brief.</div>
      </article>
    `;
    snapshotGridEl.innerHTML = `
      <article class="snapshot-card">
        <span class="snapshot-label">Priority</span>
        <strong>Pending</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Issue</span>
        <strong>Pending</strong>
      </article>
      <article class="snapshot-card location-card">
        <span class="snapshot-label">Location</span>
        <strong>Pending</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Pin</span>
        <strong>Unresolved</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Geo Confidence</span>
        <strong>Pending</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Dispatch Lane</span>
        <strong>Pending</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Demo Ticket</span>
        <strong>Not created</strong>
      </article>
      <article class="snapshot-card">
        <span class="snapshot-label">Tool Calls</span>
        <strong>0</strong>
      </article>
    `;
    labelChipsEl.innerHTML = `<span class="chip pending">Waiting for labels</span>`;
    triagePanelEl.innerHTML = `
      <article class="entry event">
        <div class="body">No triage state yet.</div>
      </article>
    `;
    ticketPanelEl.innerHTML = `
      <article class="entry event">
        <div class="body">No incident ticket yet.</div>
      </article>
    `;
    toolChainEl.innerHTML = `
      <article class="entry event">
        <div class="body">No deterministic assist chain activity yet.</div>
      </article>
    `;
    servicePanelEl.innerHTML = `
      <article class="service-card service-card-pending">
        <div class="service-top">
          <strong>Services</strong>
          <span class="service-state">Pending</span>
        </div>
        <p>Start a call to populate the service board.</p>
        <small>Police, ambulance, and fire updates will appear here.</small>
      </article>
    `;
    toolLogEl.innerHTML = `
      <article class="entry event">
        <div class="body">No tool calls yet.</div>
      </article>
    `;
    setMemoryPanel(null);
    clearMapState();
  }

  async function startDemo() {
    clearProviderTimeout();
    stopDemoProgressLoop();
    callStartedPerf = performance.now();
    voiceLatencyStages.clear();
    finalizedCallerTurnIndex = 0;
    finalizedCallerTurnPerf = null;
    callerTurnLatencyStages.clear();
    liveMlClassifier?.reset();
    sessionReportFailureCount = 0;
    sessionReportPaused = false;
    finalSessionReportStatus = null;
    if (retrySessionReportBtn) retrySessionReportBtn.hidden = true;
    renderLiveMlUpdate({ status: "awaiting" });
    const dispatchMode = getSelectedDispatchMode();
    const voiceProvider = voiceProviderSelect?.value === "gemini" ? "gemini" : "gradbot";
    let configurationUnavailable = false;
    connectBtn.disabled = true;
    if (dispatchModeSelect) {
      dispatchModeSelect.disabled = true;
    }
    if (voiceProviderSelect) voiceProviderSelect.disabled = true;
    reportSent = false;
    sessionReportRevision = 0;
    sessionId = crypto.randomUUID();
    clearMapState();
    const callSessionId = sessionId;
    sessionStartedAt = new Date().toISOString();
    clearTranscriptBuffers();
    userTranscriptChunks = [];
    agentTranscriptChunks = [];
    userTranscripts = [];
    agentTranscripts = [];
    eventEntries = [];
    maxBufferMs = 0;
    currentBufferMs = 0;
    playbackQueueDiagnostics = { receivedChunks: 0, receivedBytes: 0, decodedSamples: 0, queuedChunks: 0 };
    if (playbackWatchdogHandle) clearTimeout(playbackWatchdogHandle);
    playbackWatchdogHandle = null;
    latestSession = null;
    interruptCount = 0;
    lastAdaptiveAt = 0;
    recoveryModeArmed = false;
    assistantTurnActive = false;
    assistantTextReceivedThisTurn = false;
    assistantAudioBytesThisTurn = 0;
    assistantPlaybackStartedThisTurn = false;
    ttsAudioChunksReceived = 0;
    ttsAudioBytesReceived = 0;
    audioPlayedSeconds = 0;
    if (responseTimeoutHandle) clearTimeout(responseTimeoutHandle);
    responseTimeoutHandle = null;
    pendingTriageUpdate = null;
    pendingLedgerUpdate = null;
    pendingDispatchUpdate = null;
    latestTriageContext = null;
    latestLedgerContext = null;
    latestDispatchContext = null;
    lastTriageVersionSent = 0;
    lastLedgerVersionSent = 0;
    lastDispatchVersionSent = 0;
    lastRescueIndexSeen = -1;
    lastActiveTranscriptProvider = "gradium";
    humanTakeoverEngaged = false;
    stickySummaryText = null;
    pendingHandoffResumePrompt = null;
    eventLogEl.innerHTML = "";
    resetDashboard();
    renderDialogueStreams();
    sessionMetricEl.textContent = sessionId;
    setSessionReportStatus("Session report: preparing.");
    setStatus("Checking configuration...", "idle");
    setPlaybackStatus("Waiting for provider connection");
    dispatchMetricEl.textContent = dispatchMode === "llm_only" ? "Listening · LLM only" : "Listening · SLM + LLM";
    logEvent(`dispatch mode selected: ${dispatchModeLabel(dispatchMode)}`);

    try {
      const configResponse = await fetch(`/api/audio-config?triage_mode=${encodeURIComponent(dispatchMode)}`);
      if (!configResponse.ok) throw new Error(`Live configuration check failed (${configResponse.status}).`);
      currentAudioConfig = await configResponse.json();
      const missingConfig = [
        ...(voiceProvider === "gemini" ? (currentAudioConfig.gemini_voice_missing_config || []) : (currentAudioConfig.live_voice_missing_config || [])),
        ...(currentAudioConfig.triage_missing_config || []),
      ];
      if (missingConfig.length) {
        configurationUnavailable = true;
        setStatus("Unavailable", "error");
        throw new Error(`Live call unavailable. Configure: ${missingConfig.join(", ")}. No call was started.`);
      }
      setStatus("Connecting...", "idle");
      logEvent("requesting microphone access");
      player = await createPlayer(currentAudioConfig, callSessionId);
      const callPlayer = player;
      const callSocket = new WebSocket(`${getWsUrl()}?voice_provider=${encodeURIComponent(voiceProvider)}`);
      ws = callSocket;
      callSocket.binaryType = "blob";

      callSocket.onopen = () => {
        if (!window.DispatcherAudioUtils.isCurrentAudioSession(sessionId, callSessionId, ws, callSocket)) return;
        callSocket.send(
          JSON.stringify({
            type: "start",
            session_id: sessionId,
            triage_mode: dispatchMode,
            voice_provider: voiceProvider,
            client: {
              user_agent: navigator.userAgent,
              platform: navigator.platform,
              triage_mode: dispatchMode,
            },
          })
        );
        setStatus("Connecting...", "idle");
        dispatchMetricEl.textContent = "Waiting for live provider initialization";
        disconnectBtn.disabled = false;
        logEvent("websocket opened; waiting for live provider initialization");
        startBackgroundLoops();
        refreshSessionDetail().catch(() => {});
      };

      callSocket.onmessage = (event) => {
        if (!window.DispatcherAudioUtils.isCurrentAudioSession(sessionId, callSessionId, ws, callSocket)) return;
        if (humanTakeoverEngaged && event.data instanceof Blob) {
          return;
        }
        const isAudioBlob = event.data instanceof Blob;
        if (event.data instanceof Blob) {
          recordVoiceLatency("first_returned_audio_chunk");
          recordCallerTurnLatency("first_returned_audio_chunk");
          if (!window.DispatcherAudioUtils.isValidPcm16ChunkByteLength(event.data.size)) {
            setPlaybackStatus("Provider sent an empty or malformed PCM chunk", "error");
            logEvent("discarded malformed PCM output chunk", "error");
            return;
          }
          ttsAudioChunksReceived += 1;
          ttsAudioBytesReceived += event.data.size;
          assistantAudioBytesThisTurn += event.data.size;
          playbackQueueDiagnostics.receivedChunks += 1;
          playbackQueueDiagnostics.receivedBytes += event.data.size;
          clearProviderTimeout();
          if (assistantAudioBytesThisTurn === event.data.size) {
            logEvent("dispatcher audio bytes received from provider");
            setPlaybackStatus("Dispatcher audio received; buffering for playback");
          }
          updateAudioUnlockState(callPlayer);
        }
        if (typeof event.data === "string") {
          try {
            const message = JSON.parse(event.data);
            if (message.type === "event" && ["llm_started", "first_word", "first_tts_audio"].includes(message.event?.event_type)) {
              recordVoiceLatency("first_provider_response");
            }
            if (message.type === "connection_status") {
              if (message.status === "connected") {
                setStatus("Connected", "live");
                dispatchMetricEl.textContent = dispatchMode === "llm_only" ? "Listening · LLM only" : "Listening · SLM + LLM";
                logEvent("live provider session initialized");
                setPlaybackStatus("Connected; waiting for dispatcher response");
              }
              return;
            }
            if (message.type === "error") {
              // Server-sent failures (quota, provider, watchdog) must be shown
              // verbatim instead of being routed to the generic audio handler.
              const errorCode = typeof message.code === "string" && message.code ? message.code : "provider_error";
              const errorText =
                typeof message.message === "string" && message.message
                  ? message.message
                  : "The live provider reported an error.";
              setStatus("Error", "error");
              setPlaybackStatus(errorText, "error");
              dispatchMetricEl.textContent = "Provider unavailable · session ended";
              logEvent(`live provider error (${errorCode}): ${errorText}`, "error");
              return;
            }
          } catch (_) {
            // Let the audio player handle and report malformed provider messages.
          }
        }
        callPlayer.handleMessage(event.data);
        if (isAudioBlob) {
          callPlayer.drain().then(() => {
            if (!window.DispatcherAudioUtils.isCurrentAudioSession(sessionId, callSessionId, ws, callSocket)) return;
            recordVoiceLatency("frontend_audio_enqueue");
            recordCallerTurnLatency("frontend_audio_enqueue");
            playbackQueueDiagnostics.queuedChunks += 1;
            playbackQueueDiagnostics.decodedSamples += event.data.size / 2;
            const audioContext = callPlayer.audioProcessor?.audioContext;
            logEvent(window.DispatcherAudioUtils.getSafePlaybackDiagnostic({
              stage: "queued",
              contextState: audioContext?.state,
              receivedChunks: playbackQueueDiagnostics.receivedChunks,
              receivedBytes: playbackQueueDiagnostics.receivedBytes,
              decodedSamples: playbackQueueDiagnostics.decodedSamples,
              queuedChunks: playbackQueueDiagnostics.queuedChunks,
              actualAudioPlayed: audioPlayedSeconds,
              bufferMs: currentBufferMs,
              sampleRate: audioContext?.sampleRate,
            }));
            if (playbackWatchdogHandle) clearTimeout(playbackWatchdogHandle);
            playbackWatchdogHandle = setTimeout(() => {
              playbackWatchdogHandle = null;
              if (!window.DispatcherAudioUtils.isCurrentAudioSession(sessionId, callSessionId, ws, callSocket)) return;
              if (assistantPlaybackStartedThisTurn || assistantAudioBytesThisTurn <= 0) return;
              const context = callPlayer.audioProcessor?.audioContext;
              const state = context?.state || "unavailable";
              const stage = state === "suspended" ? "suspended" : "stalled";
              logEvent(window.DispatcherAudioUtils.getSafePlaybackDiagnostic({
                stage,
                contextState: state,
                receivedChunks: playbackQueueDiagnostics.receivedChunks,
                receivedBytes: playbackQueueDiagnostics.receivedBytes,
                decodedSamples: playbackQueueDiagnostics.decodedSamples,
                queuedChunks: playbackQueueDiagnostics.queuedChunks,
                actualAudioPlayed: audioPlayedSeconds,
                bufferMs: currentBufferMs,
                sampleRate: context?.sampleRate,
              }), "error");
              if (state === "suspended") {
                updateAudioUnlockState(callPlayer);
              } else {
                setPlaybackStatus("Audio reached the browser queue, but the output worklet has not played samples", "error");
              }
            }, 2200);
          }).catch((error) => {
            const errorName = error?.name || "AudioQueueError";
            logEvent(`audio playback failed stage=queue error_type=${errorName}`, "error");
            setPlaybackStatus("Browser audio queue failed; dispatcher audio could not play", "error");
          });
        }
      };

      callSocket.onerror = () => {
        if (!window.DispatcherAudioUtils.isCurrentAudioSession(sessionId, callSessionId, ws, callSocket)) return;
        logEvent("websocket error", "error");
        setStatus("Error", "error");
      };

      callSocket.onclose = (event) => {
        if (!window.DispatcherAudioUtils.isCurrentAudioSession(sessionId, callSessionId, ws, callSocket)) return;
        logEvent(`websocket closed (${event.code})`);
        stopBackgroundLoops();
        finalizePendingUser("close");
        finalizePendingAgent("close");
        flushSessionReport("closed");
        const closeOutcome = window.DispatcherAudioUtils.getSocketCloseOutcome(event.code, event.wasClean);
        const hadSocketError = statusEl.textContent === "Error";
        const failed = hadSocketError || closeOutcome.status === "Unavailable";
        cleanup();
        setStatus(failed ? closeOutcome.status === "Unavailable" ? "Unavailable" : "Error" : closeOutcome.status, failed ? "error" : closeOutcome.kind);
        if (failed) {
          const failureMessage = closeOutcome.status === "Unavailable"
            ? closeOutcome.playbackStatus
            : `WebSocket connection failed before close (${event.code}).`;
          setPlaybackStatus(failureMessage, "error");
          dispatchMetricEl.textContent = "Provider unavailable · session ended";
          sessionMetricEl.textContent = `Ended · ${sessionId}`;
          logEvent(failureMessage, "error");
        } else {
          dispatchMetricEl.textContent = "Session ended";
          sessionMetricEl.textContent = `Ended · ${sessionId}`;
        }
        refreshSessionDetail().catch(() => {});
      };
    } catch (error) {
      logEvent(error.message || String(error), "error");
      setStatus(configurationUnavailable ? "Unavailable" : "Error", "error");
      flushSessionReport("setup_error");
      cleanup();
    }
  }

  function cleanup() {
    stopBackgroundLoops();
    clearTranscriptBuffers();
    liveMlClassifier?.reset();
    if (responseTimeoutHandle) {
      clearTimeout(responseTimeoutHandle);
      responseTimeoutHandle = null;
    }
    if (playbackWatchdogHandle) {
      clearTimeout(playbackWatchdogHandle);
      playbackWatchdogHandle = null;
    }
    if (dispatchModeSelect) {
      dispatchModeSelect.disabled = false;
    }
    if (voiceProviderSelect) voiceProviderSelect.disabled = false;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
    ws = null;
    if (player) {
      stopPcmCapture();
      player.stop();
    }
    player = null;
    connectBtn.disabled = false;
    disconnectBtn.disabled = true;
    bufferMetricEl.textContent = "0 ms";
    if (!["Error", "Unavailable", "Disconnected"].includes(statusEl.textContent)) {
      setStatus("Idle", "idle");
    }
  }

  function stopDemo() {
    flushPcmPending(true);
    finalizePendingUser("stop");
    finalizePendingAgent("stop");
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
      ws.close();
    }
    logEvent("call stopped");
    flushSessionReport("stopped");
    cleanup();
    refreshSessionDetail().catch(() => {});
  }

  connectBtn.addEventListener("click", startDemo);
  disconnectBtn.addEventListener("click", stopDemo);
  incidentDemoRun?.addEventListener("click", runIncidentDemoScenario);
  if (dispatchModeSelect) {
    const savedMode = window.localStorage.getItem(DISPATCH_MODE_STORAGE_KEY);
    const storedMode = savedMode ? normalizeDispatchMode(savedMode) : "llm_only";
    dispatchModeSelect.value = storedMode;
    if (!savedMode) {
      fetch("/api/audio-config")
        .then((response) => (response.ok ? response.json() : null))
        .then((config) => {
          if (config && !window.localStorage.getItem(DISPATCH_MODE_STORAGE_KEY) && !dispatchModeSelect.disabled) {
            dispatchModeSelect.value = normalizeDispatchMode(config.default_dispatch_mode);
          }
        })
        .catch(() => {});
    }
    dispatchModeSelect.addEventListener("change", () => {
      const nextMode = getSelectedDispatchMode();
      window.localStorage.setItem(DISPATCH_MODE_STORAGE_KEY, nextMode);
      logEvent(`dispatch mode ready: ${dispatchModeLabel(nextMode)}`);
    });
  }
  triageDemoForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    triageDemoSubmit.disabled = true;
    triageDemoStatus.textContent = "Analyzing…";
    triageDemoResult.hidden = true;
    try {
      const response = await fetch("/api/triage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: triageDemoText.value, engine: triageDemoEngine.value }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
      triageDemoStatus.textContent = payload.model_artifacts?.demo_only
        ? "Demo-only deterministic rules · not a trained ML model · advisory only"
        : `${titleCase(payload.selected_engine)} · ${payload.runtime_provider} · advisory only`;
      triageDemoResult.textContent = JSON.stringify(payload, null, 2);
      triageDemoResult.hidden = false;
    } catch (error) {
      triageDemoStatus.textContent = error.message || String(error);
    } finally {
      triageDemoSubmit.disabled = false;
    }
  });
  complaintCodeForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const enteredCode = complaintCodeInput.value.trim();
    if (!enteredCode) {
      complaintCodeStatus.textContent = "Enter a complaint code before running classification.";
      complaintCodeResult.hidden = true;
      complaintCodeInput.focus();
      return;
    }
    complaintCodeSubmit.disabled = true;
    complaintCodeStatus.textContent = "Running local scikit-learn model…";
    complaintCodeResult.hidden = true;
    try {
      const response = await fetch("/api/triage/complaint-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: enteredCode }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
      complaintCodeStatus.textContent = `${payload.model.version} · broad complaint-code classification · human review required`;
      complaintCodeResult.textContent = window.DispatcherComplaintCodeUtils.presentComplaintCodeResult(payload);
      complaintCodeResult.hidden = false;
    } catch (error) {
      complaintCodeStatus.textContent = error.message || String(error);
    } finally {
      complaintCodeSubmit.disabled = false;
    }
  });
  firstAidForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (firstAidSubmit) firstAidSubmit.disabled = true;
    try {
      await requestFirstAidGuidance(firstAidText?.value || "", null, true);
    } finally {
      if (firstAidSubmit) firstAidSubmit.disabled = false;
    }
  });
  clearBtn.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      logEvent("stop the call before clearing the console", "error");
      return;
    }
    eventLogEl.innerHTML = "";
    clearTranscriptBuffers();
    userTranscriptChunks = [];
    agentTranscriptChunks = [];
    userTranscripts = [];
    agentTranscripts = [];
    eventEntries = [];
    sessionReportFailureCount = 0;
    sessionReportPaused = false;
    finalSessionReportStatus = null;
    if (retrySessionReportBtn) retrySessionReportBtn.hidden = true;
    sessionId = null;
    sessionStartedAt = null;
    latestSession = null;
    reportSent = false;
    liveMlClassifier?.reset();
    renderLiveMlUpdate({ status: "awaiting", reason: "Complaint-code classification requires an explicitly stated supported code." });
    lastActiveTranscriptProvider = "gradium";
    humanTakeoverEngaged = false;
    stickySummaryText = null;
    pendingHandoffResumePrompt = null;
    latestLedgerContext = null;
    lastLedgerVersionSent = 0;
    pendingLedgerUpdate = null;
    resetDashboard();
    renderDialogueStreams();
  });

  retrySessionReportBtn?.addEventListener("click", () => {
    if (!sessionId) return;
    sessionReportFailureCount = 0;
    sessionReportPaused = false;
    retrySessionReportBtn.hidden = true;
    if (finalSessionReportStatus) {
      reportSent = false;
      flushSessionReport(finalSessionReportStatus);
      return;
    }
    syncSessionReport("live").catch(() => {});
  });

  takeoverBtn.addEventListener("click", () => {
    if (!sessionId || !latestSession) {
      logEvent("start a call before using human takeover", "error");
      return;
    }
    humanTakeoverEngaged = !humanTakeoverEngaged;
    pendingHandoffResumePrompt = humanTakeoverEngaged ? null : HANDOFF_RESUME_PROMPT;
    sendTriageConfig(humanTakeoverEngaged ? "human_takeover_engaged" : "human_takeover_cleared");
    logEvent(
      humanTakeoverEngaged
        ? "human takeover engaged"
        : `handover returned to agent · next question: ${HANDOFF_RESUME_PROMPT}`
    );
    renderOperatorDesk(latestSession, deriveDashboardState(latestSession));
  });

  window.addEventListener("beforeunload", () => {
    finalizePendingUser("unload");
    finalizePendingAgent("unload");
    flushSessionReport("unloaded");
    stopBackgroundLoops();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
    if (player) {
      stopPcmCapture();
      player.stop();
    }
  });

  initMap();
  loadIncidentDemoScenarios();
  resetDashboard();
  renderDialogueStreams();
})();
