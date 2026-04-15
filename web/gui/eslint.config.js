import js from "@eslint/js";
import prettier from "eslint-plugin-prettier/recommended";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import simpleImportSort from "eslint-plugin-simple-import-sort";
import tseslint from "typescript-eslint";

export default tseslint.config(
  js.configs.recommended,
  ...tseslint.configs.strict,
  ...tseslint.configs.stylistic,
  react.configs.flat.recommended,
  react.configs.flat["jsx-runtime"],
  prettier,
  {
    settings: {
      react: { version: "detect" },
    },
  },
  {
    plugins: {
      "react-hooks": reactHooks,
      "simple-import-sort": simpleImportSort,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,

      "simple-import-sort/imports": "error",
      "simple-import-sort/exports": "error",

      "no-param-reassign": ["error", { props: false }],

      "object-shorthand": ["error", "always"],
      "prefer-template": "error",
      "prefer-arrow-callback": "error",

      "no-useless-return": "error",
      "no-useless-concat": "error",
      "no-useless-rename": "error",

      "no-unneeded-ternary": "error",

      "no-loop-func": "error",

      "@typescript-eslint/array-type": ["error", { default: "array-simple" }],
    },
  },
  {
    ignores: ["dist/", "node_modules/", "e2e/", "src/renderer/types/generated/"],
  },
);
