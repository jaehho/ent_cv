<template>
  <div class="login-root">
    <div class="login-card">
      <div class="login-label">Mount Sinai &amp; The Cooper Union</div>
      <h1 class="login-title">ENT Computer Vision</h1>
      <p class="login-subtitle">Sign in to continue</p>

      <form @submit.prevent="handleLogin" class="login-form">
        <input
          v-model="username"
          type="text"
          placeholder="Username"
          autocomplete="username"
          class="login-input"
          :disabled="loading"
          ref="usernameRef"
        />
        <input
          v-model="password"
          type="password"
          placeholder="Password"
          autocomplete="current-password"
          class="login-input"
          :disabled="loading"
        />
        <div v-if="error" class="login-error">{{ error }}</div>
        <button type="submit" class="login-btn" :disabled="loading">
          {{ loading ? "Signing in…" : "Sign In" }}
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";

const emit = defineEmits(["authenticated"]);

const username = ref("");
const password = ref("");
const error = ref("");
const loading = ref(false);
const usernameRef = ref(null);

onMounted(() => usernameRef.value?.focus());

function getCookie(name) {
  const match = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
  return match ? decodeURIComponent(match[1]) : null;
}

async function handleLogin() {
  error.value = "";
  loading.value = true;

  try {
    // Ensure we have a CSRF token first
    let csrfToken = getCookie("entcv_csrftoken");
    if (!csrfToken) {
      await fetch("/auth/session/", { credentials: "include" });
      csrfToken = getCookie("entcv_csrftoken");
    }

    const res = await fetch("/auth/login/", {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...(csrfToken ? { "X-CSRFToken": csrfToken } : {}),
      },
      body: JSON.stringify({
        username: username.value,
        password: password.value,
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      error.value = data.error || "Login failed";
      return;
    }

    emit("authenticated", data.username);
  } catch {
    error.value = "Network error — is the server running?";
  } finally {
    loading.value = false;
  }
}
</script>

<style scoped>
.login-root {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #06060c;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.login-card {
  width: 360px;
  padding: 48px 36px;
  background: #0e0e1a;
  border: 1px solid #1a1a24;
  border-radius: 12px;
  text-align: center;
}
.login-label {
  font-size: 11px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #4ecdc4;
  margin-bottom: 12px;
}
.login-title {
  font-size: 22px;
  font-weight: 700;
  color: #e0e0e0;
  margin-bottom: 6px;
}
.login-subtitle {
  font-size: 13px;
  color: #555;
  margin-bottom: 28px;
}
.login-form {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.login-input {
  padding: 10px 14px;
  background: #08080e;
  border: 1px solid #1a1a24;
  border-radius: 6px;
  color: #e0e0e0;
  font-size: 14px;
  outline: none;
  transition: border-color 0.15s;
}
.login-input:focus {
  border-color: #4ecdc4;
}
.login-input::placeholder {
  color: #444;
}
.login-btn {
  padding: 10px;
  background: #4ecdc4;
  color: #06060c;
  font-weight: 600;
  font-size: 14px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  margin-top: 4px;
  transition: opacity 0.15s;
}
.login-btn:hover:not(:disabled) {
  opacity: 0.9;
}
.login-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.login-error {
  color: #ff6b6b;
  font-size: 13px;
  padding: 6px 0;
}
</style>
