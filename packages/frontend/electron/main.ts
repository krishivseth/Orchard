import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import http from 'http';
import net from 'net';
import path from 'path';

const DEV_SERVER_URL = 'http://localhost:3000';
const HEALTH_POLL_INTERVAL_MS = 250;
const HEALTH_TIMEOUT_MS = 60_000;

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
let backendExited = false;
let backendPort = 8000;

// Check if a port is available
function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(false));
    server.once('listening', () => {
      server.close();
      resolve(true);
    });
    server.listen(port);
  });
}

// Find an available port starting from 8000
async function findAvailablePort(startPort = 8000): Promise<number> {
  let port = startPort;
  while (port < startPort + 100) {
    if (await isPortAvailable(port)) {
      return port;
    }
    port++;
  }
  throw new Error('No available ports found');
}

// Single GET request; resolves true on a 2xx response
function probe(url: string): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => {
      res.resume();
      resolve(res.statusCode !== undefined && res.statusCode >= 200 && res.statusCode < 300);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(1000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Poll the backend until it answers /health (or /api/models as a fallback)
async function waitForBackend(port: number): Promise<void> {
  const deadline = Date.now() + HEALTH_TIMEOUT_MS;
  const healthUrl = `http://localhost:${port}/health`;
  const fallbackUrl = `http://localhost:${port}/api/models`;

  while (Date.now() < deadline) {
    if (backendExited) {
      throw new Error('Backend process exited before it became ready');
    }
    if ((await probe(healthUrl)) || (await probe(fallbackUrl))) {
      return;
    }
    await sleep(HEALTH_POLL_INTERVAL_MS);
  }
  throw new Error(`Backend did not become ready within ${HEALTH_TIMEOUT_MS / 1000}s`);
}

// Start the Python backend
async function startBackend(): Promise<void> {
  backendPort = await findAvailablePort(8000);
  console.log(`Starting backend on port ${backendPort}`);

  const isDev = !app.isPackaged;
  let backendPath: string;
  let backendArgs: string[];

  if (isDev) {
    // Development: use Python script directly
    // __dirname is at: packages/frontend/out/main
    // We need to get to the Orchard root (4 levels up)
    const projectRoot = path.join(__dirname, '..', '..', '..', '..');
    backendPath = 'python3';
    backendArgs = [
      path.join(projectRoot, 'packages', 'backend', 'main.py'),
      '--port',
      backendPort.toString()
    ];
  } else {
    // Production: use bundled executable
    backendPath = path.join(process.resourcesPath, 'backend', 'orchard-backend');
    backendArgs = ['--port', backendPort.toString()];
  }

  console.log(`Backend path: ${backendPath}`);
  console.log(`Backend args: ${backendArgs.join(' ')}`);

  backendExited = false;
  backendProcess = spawn(backendPath, backendArgs, {
    stdio: 'inherit',
    env: { ...process.env, PORT: backendPort.toString() }
  });

  backendProcess.on('error', (error) => {
    console.error('Failed to start backend:', error);
    backendExited = true;
  });

  backendProcess.on('exit', (code, signal) => {
    console.log(`Backend process exited with code ${code} and signal ${signal}`);
    backendExited = true;
    backendProcess = null;
  });

  await waitForBackend(backendPort);
  console.log('Backend started successfully');
}

// Stop the Python backend
function stopBackend(): void {
  if (backendProcess) {
    console.log('Stopping backend...');
    try {
      backendProcess.kill('SIGTERM');
    } catch (error) {
      console.error('Failed to stop backend:', error);
    }
    backendProcess = null;
  }
}

function isAllowedNavigation(url: string): boolean {
  if (!app.isPackaged && url.startsWith(DEV_SERVER_URL)) {
    return true;
  }
  return url.startsWith('file://');
}

// Create the main window
function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: 'Orchard - Distributed LLM Platform',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, '../preload/index.cjs'),
      additionalArguments: [`--backend-port=${backendPort}`]
    }
  });

  // Never open new windows from the renderer
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) {
      void shell.openExternal(url);
    }
    return { action: 'deny' };
  });

  // Only allow navigation to the dev server or the bundled app
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (isAllowedNavigation(url)) return;
    event.preventDefault();
    if (/^https?:/.test(url)) {
      void shell.openExternal(url);
    }
  });

  // Load the app
  if (!app.isPackaged) {
    void mainWindow.loadURL(DEV_SERVER_URL); // electron-vite dev server
    mainWindow.webContents.openDevTools();
  } else {
    void mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// App lifecycle
app.whenReady().then(async () => {
  try {
    await startBackend();
    createWindow();

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
      }
    });
  } catch (error) {
    console.error('Failed to start app:', error);
    stopBackend();
    dialog.showErrorBox(
      'Orchard failed to start',
      `The backend could not be started.\n\n${error instanceof Error ? error.message : String(error)}`
    );
    app.quit();
  }
});

// Quit (and therefore stop the backend) when the last window closes on every platform
app.on('window-all-closed', () => {
  app.quit();
});

app.on('before-quit', () => {
  stopBackend();
});

app.on('will-quit', () => {
  stopBackend();
});

// IPC handlers (kept for compatibility; the port is also passed via additionalArguments)
ipcMain.handle('get-backend-url', () => {
  return `http://localhost:${backendPort}`;
});
