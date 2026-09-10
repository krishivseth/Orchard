// Type declarations for Electron API exposed to renderer process

interface Window {
  electronAPI?: {
    /** Backend base URL (e.g. http://localhost:8000), available synchronously at load. */
    backendUrl: string;
    /** Async variant kept for compatibility. */
    getBackendUrl: () => Promise<string>;
  };
}
