import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";
import "./styles/tokens.css";
import "./composables/useTheme.js"; // side-effect: apply saved theme before first paint

createApp(App).use(router).mount("#app");
