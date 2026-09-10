(function () {
  var busy = false;
  var credentialNames = ["cryosparc_base_url", "cryosparc_email", "cryosparc_password"];

  function byId(id) {
    return document.getElementById(id);
  }

  function field(name) {
    return document.querySelector('#cryosparcConnectForm [data-credential-name="' + name + '"]');
  }

  function setStatus(message, tone) {
    var element = byId("cryosparcConnectStatus");
    if (!element) return;
    element.textContent = message || "";
    element.classList.toggle("ok", tone === "ok");
    element.classList.toggle("error", tone === "error");
  }

  function setGateFallback(connected, details) {
    var gate = byId("cryosparcGate");
    var runForm = byId("cryosparcRunForm");
    var summary = byId("cryosparcConnectionSummary");
    var hint = byId("cryosparcCredentialHint");
    var button = byId("cryosparcConnectButton");
    if (gate) gate.classList.toggle("is-locked", !connected);
    if (runForm) runForm.setAttribute("aria-hidden", connected ? "false" : "true");
    if (!connected) {
      if (summary) summary.textContent = "Not connected";
      if (hint) hint.textContent = "Connect to continue.";
      if (button) button.textContent = "Connect";
      return;
    }
    var user = details && details.email ? " as " + details.email : "";
    var version = details && details.server_version ? " | " + details.server_version : "";
    var display = details && (details.display || details.host) ? (details.display || details.host) : "CryoSPARC";
    if (summary) summary.textContent = "Connected to " + display + user + version;
    if (hint) hint.textContent = "Connected.";
    if (button) button.textContent = "Reconnect";
  }

  function syncCredentials(payload) {
    var forms = document.querySelectorAll("form[data-uses-cryosparc-connection], #cryosparcRunForm");
    for (var i = 0; i < forms.length; i += 1) {
      for (var j = 0; j < credentialNames.length; j += 1) {
        var target = forms[i].elements[credentialNames[j]];
        if (target) target.value = payload[credentialNames[j]] || "";
      }
    }
  }

  function collectPayload() {
    var payload = {};
    for (var i = 0; i < credentialNames.length; i += 1) {
      var name = credentialNames[i];
      var element = field(name);
      var value = element && element.value ? element.value.trim() : "";
      if (value) payload[name] = value;
    }
    if (!payload.cryosparc_base_url) {
      setStatus("Enter a CryoSPARC Instance URL.", "error");
      var urlField = field("cryosparc_base_url");
      if (urlField) urlField.focus();
      return null;
    }
    return payload;
  }

  function requestJson(path, options) {
    return fetch(path, options).then(function (response) {
      return response.text().then(function (text) {
        var payload = {};
        if (text) {
          try {
            payload = JSON.parse(text);
          } catch (_error) {
            payload = {};
          }
        }
        if (!response.ok) {
          throw new Error(payload.error || response.statusText || "Request failed");
        }
        return payload;
      });
    });
  }

  function setBusy(button, value) {
    busy = value;
    if (!button) return;
    button.disabled = value;
    button.textContent = value ? "Checking..." : (window.cryoFilterCryosparcConnected ? "Reconnect" : "Connect");
  }

  function connect(event) {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
      if (event.stopImmediatePropagation) event.stopImmediatePropagation();
    }
    if (busy) return false;
    var button = byId("cryosparcConnectButton");
    var payload = collectPayload();
    if (!payload) return false;
    setBusy(button, true);
    setStatus("Checking connection...");
    requestJson("/api/cryosparc/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      credentials: "same-origin"
    }).then(function (result) {
      window.cryoFilterCryosparcConnected = true;
      if (window.cryoFilterSyncCryosparcRunCredentials) {
        window.cryoFilterSyncCryosparcRunCredentials(payload);
      } else {
        syncCredentials(payload);
      }
      var passwordField = field("cryosparc_password");
      if (passwordField) passwordField.value = "";
      if (window.cryoFilterSetCryosparcGate) {
        window.cryoFilterSetCryosparcGate(true, result);
      } else {
        setGateFallback(true, result);
      }
      setStatus("Connected.", "ok");
      var projectField = document.querySelector("#cryosparcRunForm input[name='project']");
      if (projectField) projectField.focus();
    }).catch(function (error) {
      setStatus(error.message || String(error), "error");
    }).then(function () {
      setBusy(button, false);
    });
    return false;
  }

  function bind() {
    var box = byId("cryosparcConnectForm");
    var button = byId("cryosparcConnectButton");
    if (!box || !button || box.getAttribute("data-connect-bound") === "1") return;
    box.setAttribute("data-connect-bound", "1");
    button.addEventListener("click", connect, true);
    box.addEventListener("keydown", function (event) {
      var tag = event.target && event.target.tagName ? event.target.tagName : "";
      if (event.key === "Enter" && (tag === "INPUT" || tag === "SELECT")) {
        connect(event);
      }
    }, true);
    window.cryoFilterConnectCryosparc = connect;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind, false);
  } else {
    bind();
  }
}());
