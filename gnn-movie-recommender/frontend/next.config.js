/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy API calls to the Python backend during development
  async rewrites() {
    return [
      // Frontend API routes handle the proxying themselves,
      // so we don't need Next rewrites. But you can add them here
      // if you want to hit the Python API directly from the browser.
    ];
  },
};

module.exports = nextConfig;
