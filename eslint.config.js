import js from "@eslint/js";
import globals from "globals";

export default [
  { ignores: ["node_modules/", "vendor/", "web/vendor/", "web/models/", "fixtures/", ".venv/"] },
  js.configs.recommended,
  {
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "module",
      globals: { ...globals.browser, ...globals.node, ...globals.worker },
    },
    rules: {
      "no-unused-vars": ["error", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
      "prefer-const": "error",
      eqeqeq: ["error", "smart"],
    },
  },
];
