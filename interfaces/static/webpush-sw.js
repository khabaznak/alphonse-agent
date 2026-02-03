self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("push", (event) => {
  const payload = event.data ? event.data.json() : {};
  const title = payload.title || "Alphonse";
  const options = {
    body: payload.body || "New update",
    data: payload.data || {},
  };
  event.waitUntil(self.registration.showNotification(title, options));
});
