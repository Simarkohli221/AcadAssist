const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");

const addMessage = (type, title, body, sources = []) => {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${type}`;

  const heading = document.createElement("h3");
  heading.textContent = title;

  const text = document.createElement("p");
  text.textContent = body;

  wrapper.appendChild(heading);
  wrapper.appendChild(text);

  if (sources.length) {
    const sourceWrap = document.createElement("div");
    sourceWrap.className = "sources";
    sources.slice(0, 4).forEach((src) => {
      const item = document.createElement("div");
      item.textContent = `${src.source} (pp. ${src.page_start}-${src.page_end})`;
      sourceWrap.appendChild(item);
    });
    wrapper.appendChild(sourceWrap);
  }

  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
};

const setThinking = (value) => {
  const existing = document.getElementById("thinking");
  if (value && !existing) {
    const wrapper = document.createElement("div");
    wrapper.id = "thinking";
    wrapper.className = "message bot";

    const heading = document.createElement("h3");
    heading.textContent = "RAKA";

    const text = document.createElement("p");
    text.textContent = "Thinking...";

    wrapper.appendChild(heading);
    wrapper.appendChild(text);
    chatLog.appendChild(wrapper);
    chatLog.scrollTop = chatLog.scrollHeight;
  } else if (!value && existing) {
    existing.remove();
  }
};

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", "You", question);
  questionInput.value = "";
  setThinking(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const message = errorData.detail || "The server returned an error.";
      setThinking(false);
      addMessage("bot", "RAKA", message);
      return;
    }

    const data = await response.json();
    setThinking(false);
    addMessage("bot", "RAKA", data.answer, data.sources);
  } catch (error) {
    setThinking(false);
    addMessage("bot", "RAKA", "Unable to reach the server. Please try again.");
  }
});

addMessage(
  "bot",
  "RAKA",
  "Welcome. Ask me anything about SPIT policies, syllabus, or academic rules."
);
