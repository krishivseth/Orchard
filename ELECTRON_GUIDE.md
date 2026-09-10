# Electron Desktop App Guide

This guide explains how to run and build the Orchard desktop application using Electron.

## Overview

The Electron integration bundles the React frontend and Python FastAPI backend into a single macOS desktop application. The backend starts automatically when you launch the app.

## Prerequisites

- Node.js 18+
- Python 3.10+
- PyInstaller (for building production app)

Install PyInstaller:
```bash
pip install pyinstaller
```

## Development Mode

To run the app in development mode (recommended for development):

```bash
cd packages/frontend
npm run electron:dev
```

This will:
1. Start the Vite dev server on port 3000
2. Launch Electron with the React app
3. Automatically spawn the Python backend on port 8000 (or next available port)
4. Enable hot-reload for frontend changes

The backend will run from the Python source files directly (`packages/backend/main.py`).

## Building for Production

### Step 1: Install Backend Dependencies

Make sure all backend dependencies are installed:

```bash
cd packages/backend
pip install -r requirements.txt
pip install pyinstaller
```

### Step 2: Build the Backend Executable

```bash
cd packages/frontend
npm run electron:prebuild
```

This will:
- Use PyInstaller from the backend's `.venv` to bundle the Python backend
- Create a directory bundle at `packages/frontend/resources/backend/orchard-backend/` (the executable is `orchard-backend` inside it, next to `_internal/`)
- Include all necessary dependencies

### Why a directory bundle, and why one architecture

Two constraints shaped the packaging. Both were found by launching the packaged app and reading the backend's stderr.

- **Code signing.** electron-builder signs the app with your Apple identity. PyInstaller's one-file mode unpacks a Python framework at runtime that still carries python.org's signature, and macOS refuses to load a library whose Team ID differs from the process. The build therefore uses PyInstaller's directory mode, so every binary is inside the app bundle and gets signed consistently.
- **Architecture.** The backend binary is built for the machine running the build. `npm run electron:build` therefore packages only the host architecture. Build on an Apple Silicon Mac for an arm64 DMG, or on an Intel Mac for x64.

The app also checks it is talking to its own backend: the readiness probe requires `/health` to return `"service": "orchard-backend"`, so another server on the same port is never mistaken for Orchard.

### Step 3: Build the Electron App

Build for your current architecture:
```bash
npm run electron:build
```

Or build for specific architectures:
```bash
# For Apple Silicon (M1/M2/M3)
npm run electron:build:arm64

# For Intel Macs
npm run electron:build:x64
```

The built app will be in `packages/frontend/release/`.

### Step 4: Install and Run

The build process creates:
- `.dmg` file - Drag and drop installer
- `.zip` file - Compressed app bundle

Open the `.dmg` file and drag Orchard to your Applications folder, then launch it.

## How It Works

### Architecture

```
┌─────────────────────────────────────┐
│       Electron Main Process         │
│                                     │
│  • Spawns Python backend            │
│  • Creates browser window           │
│  • Handles IPC communication        │
└─────────────────────────────────────┘
         │                    │
         │                    │
    ┌────▼────┐         ┌────▼─────┐
    │ Python  │         │  React   │
    │ Backend │◄────────┤ Frontend │
    │ FastAPI │         │ (Renderer)│
    └─────────┘         └──────────┘
```

### Port Management

The app automatically finds an available port starting from 8000:
- If port 8000 is available, it uses that
- If not, it tries 8001, 8002, etc.
- The frontend automatically connects to the correct port

### Backend Lifecycle

The Python backend:
- Starts automatically when the Electron app launches
- Runs on localhost (not accessible from other devices)
- Stops automatically when you quit the app
- Logs output to the Electron console

### File Locations

**Development:**
- Frontend: Uses Vite dev server
- Backend: Runs from `packages/backend/main.py`

**Production:**
- Frontend: Built files in app bundle
- Backend: Bundled executable in `Contents/Resources/backend/orchard-backend`

## Troubleshooting

### Backend Won't Start

**Issue:** Backend fails to start or exits immediately.

**Solutions:**
1. Check console logs (View → Developer → Developer Tools)
2. Ensure Python 3 is installed: `python3 --version`
3. In development, manually test: `python3 packages/backend/main.py`
4. Check if port 8000-8100 range is available

### Build Fails

**Issue:** `npm run electron:build` fails.

**Solutions:**
1. Ensure PyInstaller is installed: `pip install pyinstaller`
2. Run prebuild step separately: `npm run electron:prebuild`
3. Check Python build logs for errors
4. Try cleaning: `rm -rf release dist dist-electron`

### App Won't Open on macOS

**Issue:** macOS says the app is damaged or from an unidentified developer.

**Solutions:**
1. Right-click the app → Open (first time only)
2. Or go to System Preferences → Security & Privacy → Allow
3. For code signing, see [Apple Developer documentation](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)

### WebSocket Connection Failed

**Issue:** Real-time updates not working.

**Solutions:**
1. Check if backend is running (look for console logs)
2. Verify backend URL in DevTools console
3. Check CORS settings in `packages/backend/main.py`

## Development Tips

### Viewing Logs

Open Developer Tools in the Electron app:
- macOS: `Cmd + Option + I`
- Or: Application menu → View → Developer → Developer Tools

Backend logs will appear in the terminal where you ran `npm run electron:dev`.

### Hot Reload

In development mode:
- Frontend changes reload automatically (Vite HMR)
- Backend changes require restarting the Electron app
- Electron main process changes require restarting the app

### Debugging Backend

To debug the Python backend:
1. Stop the Electron app
2. Start backend manually: `python3 packages/backend/main.py --port 8000`
3. Start Electron: `npm run electron:dev`
4. Backend will use the running instance instead of spawning a new one

## Customization

### Change App Icon

1. Create icon files:
   - macOS: `packages/frontend/build/icon.icns`
   - PNG source: `packages/frontend/build/icon.png` (1024x1024)

2. Use an icon generator or:
```bash
# macOS
mkdir icon.iconset
# Add icon files at various sizes
iconutil -c icns icon.iconset -o build/icon.icns
```

### Change App Name

Edit `packages/frontend/electron-builder.json`:
```json
{
  "appId": "com.yourcompany.orchard",
  "productName": "Your App Name"
}
```

### Configure Window Size

Edit `packages/frontend/electron/main.ts`:
```typescript
mainWindow = new BrowserWindow({
  width: 1400,    // Change width
  height: 900,    // Change height
  minWidth: 1000,
  minHeight: 700,
  // ... other options
});
```

## Distribution

For distributing to other users:

1. **Code Signing (Recommended)**
   - Requires Apple Developer account ($99/year)
   - Prevents "damaged app" warnings
   - Required for Mac App Store

2. **Notarization (Recommended)**
   - Required for macOS 10.15+
   - Prevents Gatekeeper warnings
   - See [Apple's notarization guide](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)

3. **Simple Distribution**
   - Share the `.dmg` or `.zip` file
   - Users may need to bypass Gatekeeper warnings
   - Not recommended for public distribution

## Next Steps

- Add auto-updater functionality
- Implement native macOS integrations (Touch Bar, notifications)
- Add Windows and Linux support
- Create installers for other platforms

