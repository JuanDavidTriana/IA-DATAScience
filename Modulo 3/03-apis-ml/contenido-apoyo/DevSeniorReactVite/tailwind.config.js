/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'primary': '#FFFFFF',
        'secondary': '#FFFFFF',
        'accent': {
          DEFAULT: '#FFFFFF',
          dark: '#FFFFFF',
        },
        'surface': {
            light: '#FFFFFF', // Fondo principal en modo claro (blanco)
            dark: '#111827',  // Fondo principal en modo oscuro (gris muy oscuro)
        },
        'on-surface': {
            light: '#FFFFFF', // Texto sobre el fondo claro
            dark: '#FFFFFF',   // Texto sobre el fondo oscuro
        }
      },
    },
  },
  plugins: [],
}