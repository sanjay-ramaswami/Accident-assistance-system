(function () {
  const refreshBtn = document.getElementById("refresh-btn");
  const sessionListEl = document.getElementById("session-list");
  const sessionDetailEl = document.getElementById("session-detail");

  function renderJsonBlock(label, value) {
    return `
      <section class="json-block">
        <h3>${label}</h3>
        <pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>
      </section>
    `;
  }

  function escapeHtml(text) {
    return text
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  async function loadSessions() {
    const sessions = await fetch("/api/sessions").then((res) => res.json());
    sessionListEl.innerHTML = "";
    if (!sessions.length) {
      sessionListEl.innerHTML = `<article class="entry event"><div class="body">No recorded sessions yet.</div></article>`;
      sessionDetailEl.innerHTML = "";
      return;
    }
    sessions.forEach((session) => {
      const button = document.createElement("button");
      button.className = "session-item";
      button.innerHTML = `
        <span class="session-id">${session.session_id}</span>
        <span class="session-meta">${session.user_turns} caller / ${session.agent_turns} agent / ${session.tool_call_count} tools</span>
      `;
      button.addEventListener("click", () => loadSession(session.session_id));
      sessionListEl.appendChild(button);
    });
    await loadSession(sessions[0].session_id);
  }

  async function loadSession(sessionId) {
    const session = await fetch(`/api/sessions/${sessionId}`).then((res) => res.json());
    sessionDetailEl.innerHTML = [
      renderJsonBlock("Server", session.server || {}),
      renderJsonBlock("Merged Triage State", session.merged_triage_state || {}),
      renderJsonBlock("Triage Turns", session.triage_turns || []),
      renderJsonBlock("STT Rescue Events", session.stt_rescue_events || []),
      renderJsonBlock("Rule Overrides", session.rule_overrides || []),
      renderJsonBlock("Handoff Packet", session.handoff_packet || null),
      renderJsonBlock("Tool Calls", session.tool_calls || []),
      renderJsonBlock("Client", session.client || {}),
    ].join("");
  }

  refreshBtn.addEventListener("click", loadSessions);
  loadSessions();
})();
