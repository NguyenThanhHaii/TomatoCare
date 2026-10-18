import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [react(), VitePWA({
    registerType: 'autoUpdate',
    includeAssets: ['icon.svg', 'icon-192.png', 'icon-512.png'],
    manifest: {
      name: 'TomatoCare', short_name: 'TomatoCare',
      description: 'Tomato leaf disease recognition with UniRepLKNet-A',
      start_url: '/', scope: '/', display: 'standalone',
      background_color: '#ffffff', theme_color: '#07513f',
      icons: [
        { src: '/icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' },
        { src: '/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' },
        { src: '/icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' },
      ],
    },
    workbox: {
      // Only application assets are precached. API requests and user photos are never cached.
      globPatterns: ['**/*.{js,css,html,svg,png,ico,webmanifest}'],
      navigateFallbackDenylist: [/^\/predict/, /^\/health/],
      runtimeCaching: [],
      cleanupOutdatedCaches: true,
    },
  })],
  server: { port: 5173, strictPort: true },
});
