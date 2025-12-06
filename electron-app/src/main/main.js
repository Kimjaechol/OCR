/**
 * LawOCR - Electron Main Process
 * ==============================
 * Handles file system, PDF chunking, and RunPod API communication
 */

const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const Store = require('electron-store');

const { PDFChunker } = require('../utils/pdfChunker');
const { RunPodClient } = require('../utils/runpodClient');
const { ParallelProcessor } = require('../utils/parallelProcessor');

// Initialize store for settings
const store = new Store({
  defaults: {
    serverUrl: 'http://localhost:8888',
    outputDir: app.getPath('documents') + '/LawOCR_Output',
    chunkSize: 10,  // Pages per chunk
    maxParallel: 5,  // Maximum parallel uploads
    applyGemini: true
  }
});

let mainWindow;
let processingState = {
  isProcessing: false,
  currentJobs: new Map(),
  totalPages: 0,
  processedPages: 0
};

// Create main window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, '../assets/icon.png'),
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#1a1a2e'
  });

  // Load the renderer
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// App lifecycle
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// ============================================
// IPC Handlers - Settings
// ============================================
ipcMain.handle('get-settings', () => {
  return {
    serverUrl: store.get('serverUrl'),
    outputDir: store.get('outputDir'),
    chunkSize: store.get('chunkSize'),
    maxParallel: store.get('maxParallel'),
    applyGemini: store.get('applyGemini')
  };
});

ipcMain.handle('save-settings', (event, settings) => {
  Object.keys(settings).forEach(key => {
    store.set(key, settings[key]);
  });
  return { success: true };
});

ipcMain.handle('select-output-dir', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: '결과 저장 폴더 선택'
  });

  if (!result.canceled && result.filePaths.length > 0) {
    store.set('outputDir', result.filePaths[0]);
    return result.filePaths[0];
  }
  return null;
});

// ============================================
// IPC Handlers - File Selection
// ============================================
ipcMain.handle('select-files', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'PDF Files', extensions: ['pdf'] },
      { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'tiff', 'bmp'] }
    ],
    title: '파일 선택'
  });

  if (!result.canceled) {
    return result.filePaths.map(filePath => ({
      path: filePath,
      name: path.basename(filePath),
      size: fs.statSync(filePath).size
    }));
  }
  return [];
});

ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: '폴더 선택'
  });

  if (!result.canceled && result.filePaths.length > 0) {
    const folderPath = result.filePaths[0];
    const files = scanFolder(folderPath);
    return {
      folderPath,
      files
    };
  }
  return null;
});

// Recursively scan folder for PDF files
function scanFolder(folderPath, files = []) {
  const items = fs.readdirSync(folderPath);

  items.forEach(item => {
    const fullPath = path.join(folderPath, item);
    const stat = fs.statSync(fullPath);

    if (stat.isDirectory()) {
      scanFolder(fullPath, files);
    } else if (stat.isFile()) {
      const ext = path.extname(item).toLowerCase();
      if (['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'].includes(ext)) {
        files.push({
          path: fullPath,
          name: item,
          size: stat.size,
          type: ext === '.pdf' ? 'pdf' : 'image'
        });
      }
    }
  });

  return files;
}

// ============================================
// IPC Handlers - OCR Processing
// ============================================
ipcMain.handle('start-processing', async (event, files) => {
  if (processingState.isProcessing) {
    return { success: false, error: '이미 처리 중입니다.' };
  }

  processingState.isProcessing = true;
  processingState.totalPages = 0;
  processingState.processedPages = 0;

  const settings = {
    serverUrl: store.get('serverUrl'),
    outputDir: store.get('outputDir'),
    chunkSize: store.get('chunkSize'),
    maxParallel: store.get('maxParallel'),
    applyGemini: store.get('applyGemini')
  };

  // Ensure output directory exists
  if (!fs.existsSync(settings.outputDir)) {
    fs.mkdirSync(settings.outputDir, { recursive: true });
  }

  try {
    const processor = new ParallelProcessor(settings, (progress) => {
      mainWindow.webContents.send('processing-progress', progress);
    });

    const results = await processor.processFiles(files);

    processingState.isProcessing = false;
    mainWindow.webContents.send('processing-complete', results);

    return { success: true, results };

  } catch (error) {
    processingState.isProcessing = false;
    mainWindow.webContents.send('processing-error', error.message);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('cancel-processing', async () => {
  // TODO: Implement cancellation logic
  processingState.isProcessing = false;
  return { success: true };
});

ipcMain.handle('get-processing-state', () => {
  return processingState;
});

// ============================================
// IPC Handlers - Results
// ============================================
ipcMain.handle('open-output-folder', () => {
  const outputDir = store.get('outputDir');
  if (fs.existsSync(outputDir)) {
    shell.openPath(outputDir);
    return { success: true };
  }
  return { success: false, error: '폴더가 존재하지 않습니다.' };
});

ipcMain.handle('open-file', (event, filePath) => {
  if (fs.existsSync(filePath)) {
    shell.openPath(filePath);
    return { success: true };
  }
  return { success: false, error: '파일이 존재하지 않습니다.' };
});

// ============================================
// IPC Handlers - Server Connection
// ============================================
ipcMain.handle('test-connection', async () => {
  const serverUrl = store.get('serverUrl');
  const client = new RunPodClient(serverUrl);

  try {
    const result = await client.healthCheck();
    return { success: true, ...result };
  } catch (error) {
    return { success: false, error: error.message };
  }
});
