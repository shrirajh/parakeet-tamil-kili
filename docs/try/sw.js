// Cross-origin isolation for the try-in-browser page (GitHub Pages cannot set response headers).
// WebAssembly threads need SharedArrayBuffer, which browsers only allow on cross-origin isolated pages: this worker
// adds COOP / COEP to this folder's own responses. COEP "credentialless" keeps cross-origin CORS downloads (Hugging
// Face, the ONNX Runtime CDN) working without them having to send extra headers.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const r = e.request;
  if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;
  if (new URL(r.url).origin !== self.location.origin) return;
  e.respondWith(
    fetch(r).then((res) => {
      if (res.status === 0) return res;
      const h = new Headers(res.headers);
      h.set("Cross-Origin-Embedder-Policy", "credentialless");
      h.set("Cross-Origin-Opener-Policy", "same-origin");
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
    })
  );
});
