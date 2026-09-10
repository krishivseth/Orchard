import { contextBridge, ipcRenderer } from 'electron';

// The main process passes the backend port synchronously via additionalArguments
const portArg = process.argv.find((arg) => arg.startsWith('--backend-port='));
const backendPort = portArg ? portArg.slice('--backend-port='.length) : '8000';
const backendUrl = `http://localhost:${backendPort}`;

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  backendUrl,
  getBackendUrl: (): Promise<string> => ipcRenderer.invoke('get-backend-url')
});
