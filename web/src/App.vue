<template>
  <LoginForm v-if="!authenticated" @authenticated="onAuth" />
  <YOLOVisualizer v-else :username="username" @logout="onLogout" />
</template>

<script setup>
import { ref, onMounted } from "vue";
import LoginForm from "./components/LoginForm.vue";
import YOLOVisualizer from "./components/YOLOVisualizer.vue";

const authenticated = ref(false);
const username = ref("");
const checking = ref(true);

onMounted(async () => {
  try {
    const res = await fetch("/auth/session/", { credentials: "include" });
    if (res.ok) {
      const data = await res.json();
      authenticated.value = true;
      username.value = data.username;
    }
  } catch { /* not authenticated */ }
  checking.value = false;
});

function onAuth(user) {
  authenticated.value = true;
  username.value = user;
}

function getCookie(name) {
  const match = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
  return match ? decodeURIComponent(match[1]) : null;
}

async function onLogout() {
  const csrfToken = getCookie("entcv_csrftoken");
  await fetch("/auth/logout/", {
    method: "POST",
    credentials: "include",
    headers: csrfToken ? { "X-CSRFToken": csrfToken } : {},
  });
  authenticated.value = false;
  username.value = "";
}
</script>
