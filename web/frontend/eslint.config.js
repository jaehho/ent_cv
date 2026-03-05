import js from "@eslint/js";
import pluginVue from "eslint-plugin-vue";

export default [
  { ignores: ["dist/"] },
  js.configs.recommended,
  ...pluginVue.configs["flat/recommended"],
  {
    rules: {
      "vue/multi-word-component-names": "off",
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "no-empty": "warn",
    },
  },
  {
    files: ["vite.config.js"],
    languageOptions: {
      globals: { process: "readonly" },
    },
  },
];
