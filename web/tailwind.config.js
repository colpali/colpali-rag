/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  // Blueprint ships its own base/reset; disable Preflight so the two don't fight.
  corePlugins: { preflight: false },
  theme: {
    extend: {
      colors: {
        // deep-navy surface scale (app → panels → borders), tuned for high-contrast readability
        ink: {
          950: "#07121f",
          900: "#0a1a2b",
          800: "#0e2236",
          700: "#143049",
          600: "#1e3d5a",
          500: "#2b5175",
        },
        // cobalt primary + a lighter blue secondary
        brand: { DEFAULT: "#4d8bff", deep: "#2b6cf0", 2: "#7aa2ff" },
        // block-kind accents (kept in sync with the backend renderers)
        kind: {
          component: "#4d8bff",
          system: "#7aa2ff",
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
        glow: "0 0 0 1px rgba(77,139,255,0.28), 0 8px 30px -8px rgba(77,139,255,0.30)",
      },
    },
  },
  plugins: [],
};
