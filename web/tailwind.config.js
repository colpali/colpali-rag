/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 900: "#0b1020", 800: "#0f172a", 700: "#111827", 600: "#1e293b" },
        brand: { DEFAULT: "#38bdf8", 2: "#818cf8" },
        // block-kind accents (kept in sync with the backend renderers)
        kind: {
          component: "#38bdf8",
          system: "#818cf8",
          process: "#2dd4bf",
          io: "#fbbf24",
          store: "#a78bfa",
          external: "#fb7185",
          actor: "#4ade80",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(56,189,248,0.25), 0 8px 30px -8px rgba(56,189,248,0.25)",
      },
    },
  },
  plugins: [],
};
