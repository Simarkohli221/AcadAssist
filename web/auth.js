const form = document.getElementById("authForm");
const notice = document.getElementById("notice");
const loginTab = document.getElementById("loginTab");
const registerTab = document.getElementById("registerTab");
const title = document.getElementById("formTitle");
const subtitle = document.getElementById("formSubtitle");
const nameField = document.getElementById("nameField");
const confirmField = document.getElementById("confirmField");
const fullName = document.getElementById("fullName");
const emailInput = document.getElementById("email");
const passwordInput = document.getElementById("password");
const confirmInput = document.getElementById("confirmPassword");
const submitButton = document.getElementById("submitButton");
const buttonLabel = submitButton.querySelector(".button-label");
const buttonSpinner = submitButton.querySelector(".button-spinner");
const returnLink = document.getElementById("returnLink");
let mode = "login";

const setNotice = (message, kind = "info") => {
  notice.textContent = message;
  notice.className = `notice ${kind}`;
  notice.hidden = !message;
};

const setMode = (nextMode) => {
  mode = nextMode;
  const registering = mode === "register";
  nameField.hidden = !registering;
  confirmField.hidden = !registering;
  fullName.required = registering;
  confirmInput.required = registering;
  passwordInput.autocomplete = registering ? "new-password" : "current-password";
  title.textContent = registering ? "Join your campus" : "Welcome back";
  subtitle.textContent = registering
    ? "Create an account with your verified college email."
    : "Sign in with your college email to continue.";
  buttonLabel.textContent = registering ? "Create account" : "Log in";
  loginTab.classList.toggle("active", !registering);
  registerTab.classList.toggle("active", registering);
  loginTab.setAttribute("aria-selected", String(!registering));
  registerTab.setAttribute("aria-selected", String(registering));
  setNotice("");
};

const validateCollegeEmail = (value) => {
  const email = value.trim();
  const parts = email.split("@");
  if (parts.length !== 2 || parts[1].toLowerCase() !== "spit.ac.in") return false;
  const local = parts[0];
  return /^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+$/.test(local)
    && local.length <= 64
    && !local.startsWith(".")
    && !local.endsWith(".")
    && !local.includes("..");
};

const setLoading = (loading) => {
  submitButton.disabled = loading;
  buttonSpinner.hidden = !loading;
  buttonLabel.textContent = loading ? "Please wait" : mode === "register" ? "Create account" : "Log in";
};

const responseMessage = async (response) => {
  const data = await response.json().catch(() => ({}));
  if (Array.isArray(data.detail)) return data.detail.map((item) => item.msg).join(" ");
  return data.detail || data.message || "Something went wrong. Please try again.";
};

loginTab.addEventListener("click", () => setMode("login"));
registerTab.addEventListener("click", () => setMode("register"));

document.querySelectorAll(".visibility-toggle").forEach((button) => {
  button.addEventListener("click", () => {
    const input = document.getElementById(button.dataset.target);
    const visible = input.type === "password";
    input.type = visible ? "text" : "password";
    button.textContent = visible ? "Hide" : "Show";
    button.setAttribute("aria-label", `${visible ? "Hide" : "Show"} ${input.id === "password" ? "password" : "confirmation password"}`);
  });
});

const verificationToken = new URLSearchParams(window.location.search).get("verification");
if (verificationToken) {
  setLoading(true);
  fetch("/api/auth/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: verificationToken }),
  })
    .then(async (response) => {
      if (!response.ok) throw new Error(await responseMessage(response));
      setMode("login");
      setNotice("Your email is verified. Sign in to open the chatbot.", "success");
      window.history.replaceState({}, "", "/auth");
    })
    .catch((error) => setNotice(error.message, "error"))
    .finally(() => setLoading(false));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setNotice("");

  const email = emailInput.value.trim();
  if (!validateCollegeEmail(email)) {
    setNotice("Use a valid email address with the exact @spit.ac.in domain.", "error");
    emailInput.focus();
    return;
  }

  const payload = { email, password: passwordInput.value };
  if (mode === "register") {
    const password = passwordInput.value;
    if (fullName.value.trim().length < 2) {
      setNotice("Enter your full name.", "error");
      fullName.focus();
      return;
    }
    if (password.length < 12 || !/[a-z]/.test(password) || !/[A-Z]/.test(password) || !/\d/.test(password) || !/[^A-Za-z0-9]/.test(password)) {
      setNotice("Use at least 12 characters with uppercase, lowercase, a number, and a symbol.", "error");
      passwordInput.focus();
      return;
    }
    if (password !== confirmInput.value) {
      setNotice("Your passwords do not match.", "error");
      confirmInput.focus();
      return;
    }
    payload.full_name = fullName.value.trim();
    payload.confirm_password = confirmInput.value;
  }

  setLoading(true);
  try {
    const response = await fetch(`/api/auth/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await responseMessage(response));
    const data = await response.json();
    if (mode === "register") {
      form.reset();
      setNotice(data.message, "success");
      setMode("login");
      setNotice(data.message, "success");
    } else {
      returnLink.hidden = false;
      setNotice("Signed in. Opening your chatbot…", "success");
      window.setTimeout(() => window.location.assign("/"), 500);
    }
  } catch (error) {
    setNotice(error.message || "Unable to reach the server. Please try again.", "error");
  } finally {
    setLoading(false);
  }
});