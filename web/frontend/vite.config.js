import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

const DJANGO_PORT = parseInt(process.env.DJANGO_PORT || "8787", 10);

export default defineConfig({
  plugins: [vue()],
  test: {
    environment: "jsdom",
    globals: true,
  },
  server: {
    allowedHosts: ["entcv.jaehho.com"],
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${DJANGO_PORT}`,
        changeOrigin: true,
      },
      "/auth": {
        target: `http://127.0.0.1:${DJANGO_PORT}`,
        changeOrigin: true,
      },
      "/admin": {
        target: `http://127.0.0.1:${DJANGO_PORT}`,
        changeOrigin: true,
      },
      "/static": {
        target: `http://127.0.0.1:${DJANGO_PORT}`,
        changeOrigin: true,
      },
    },
  },
});
