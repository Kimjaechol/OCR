/**
 * LawOCR - Preload Script
 * =======================
 * Secure bridge between main and renderer processes
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to renderer
contextBridge.exposeInMainWorld('lawOCR', {
  // Settings
  getSettings: () => ipcRenderer.invoke('get-settings'),
  saveSettings: (settings) => ipcRenderer.invoke('save-settings', settings),
  selectOutputDir: () => ipcRenderer.invoke('select-output-dir'),

  // File Selection
  selectFiles: () => ipcRenderer.invoke('select-files'),
  selectFolder: () => ipcRenderer.invoke('select-folder'),

  // Processing
  startProcessing: (files) => ipcRenderer.invoke('start-processing', files),
  cancelProcessing: () => ipcRenderer.invoke('cancel-processing'),
  getProcessingState: () => ipcRenderer.invoke('get-processing-state'),

  // Results
  openOutputFolder: () => ipcRenderer.invoke('open-output-folder'),
  openFile: (filePath) => ipcRenderer.invoke('open-file', filePath),

  // Server
  testConnection: () => ipcRenderer.invoke('test-connection'),

  // Event listeners
  onProgress: (callback) => {
    ipcRenderer.on('processing-progress', (event, data) => callback(data));
  },
  onComplete: (callback) => {
    ipcRenderer.on('processing-complete', (event, data) => callback(data));
  },
  onError: (callback) => {
    ipcRenderer.on('processing-error', (event, error) => callback(error));
  },

  // Remove listeners
  removeAllListeners: () => {
    ipcRenderer.removeAllListeners('processing-progress');
    ipcRenderer.removeAllListeners('processing-complete');
    ipcRenderer.removeAllListeners('processing-error');
  }
});
