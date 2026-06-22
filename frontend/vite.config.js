import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/recommend": "http://backend:8000",
      "/similar": "http://backend:8000",
      "/health": "http://backend:8000",
    },
    // backend is still on internal port 8000; Docker networking uses service name
  },
});
