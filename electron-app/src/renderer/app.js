/**
 * LawOCR - Renderer Application
 * =============================
 * UI logic and event handling
 */

// HTML escape function to prevent XSS
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// State
let selectedFiles = [];
let isProcessing = false;
let settings = {};

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileListContainer = document.getElementById('fileListContainer');
const fileList = document.getElementById('fileList');
const fileSummary = document.getElementById('fileSummary');
const processingControls = document.getElementById('processingControls');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const serverStatus = document.getElementById('serverStatus');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  setupEventListeners();
  setupDragAndDrop();
  checkServerConnection();

  // Listen for IPC events
  window.lawOCR.onProgress(handleProgress);
  window.lawOCR.onComplete(handleComplete);
  window.lawOCR.onError(handleError);
});

// Navigation
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    const page = item.dataset.page;
    goToPage(page);
  });
});

function goToPage(pageName) {
  // Update nav
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.page === pageName);
  });

  // Update pages
  document.querySelectorAll('.page').forEach(page => {
    page.classList.toggle('active', page.id === `page-${pageName}`);
  });
}

// Settings
async function loadSettings() {
  settings = await window.lawOCR.getSettings();

  document.getElementById('settingServerUrl').value = settings.serverUrl || '';
  document.getElementById('settingOutputDir').value = settings.outputDir || '';
  document.getElementById('settingChunkSize').value = settings.chunkSize || 10;
  document.getElementById('settingMaxParallel').value = settings.maxParallel || 5;
  document.getElementById('settingApplyGemini').checked = settings.applyGemini !== false;
}

async function saveSettings() {
  const newSettings = {
    serverUrl: document.getElementById('settingServerUrl').value,
    outputDir: document.getElementById('settingOutputDir').value,
    chunkSize: parseInt(document.getElementById('settingChunkSize').value) || 10,
    maxParallel: parseInt(document.getElementById('settingMaxParallel').value) || 5,
    applyGemini: document.getElementById('settingApplyGemini').checked
  };

  await window.lawOCR.saveSettings(newSettings);
  settings = newSettings;

  alert('설정이 저장되었습니다.');
  checkServerConnection();
}

async function selectOutputDir() {
  const dir = await window.lawOCR.selectOutputDir();
  if (dir) {
    document.getElementById('settingOutputDir').value = dir;
  }
}

async function testConnection() {
  const result = await window.lawOCR.testConnection();
  if (result.success) {
    alert(`서버 연결 성공!\n상태: ${result.status}\n버전: ${result.version}`);
  } else {
    alert(`서버 연결 실패: ${result.error}`);
  }
  updateServerStatus(result.success);
}

async function checkServerConnection() {
  try {
    const result = await window.lawOCR.testConnection();
    updateServerStatus(result.success);
  } catch {
    updateServerStatus(false);
  }
}

function updateServerStatus(online) {
  const dot = serverStatus.querySelector('.status-dot');
  const text = serverStatus.querySelector('span:last-child');

  if (online) {
    dot.className = 'status-dot online';
    text.textContent = '서버 연결됨';
  } else {
    dot.className = 'status-dot offline';
    text.textContent = '서버 연결 안됨';
  }
}

// Drag and Drop
function setupDragAndDrop() {
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file => {
      const ext = file.name.split('.').pop().toLowerCase();
      return ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'].includes(ext);
    });

    if (validFiles.length > 0) {
      const fileInfos = validFiles.map(f => ({
        path: f.path,
        name: f.name,
        size: f.size
      }));
      addFiles(fileInfos);
    }
  });
}

function setupEventListeners() {
  // Additional event listeners if needed
}

// File Selection
async function selectFiles() {
  const files = await window.lawOCR.selectFiles();
  if (files && files.length > 0) {
    addFiles(files);
  }
}

async function selectFolder() {
  const result = await window.lawOCR.selectFolder();
  if (result && result.files) {
    addFiles(result.files);
  }
}

function addFiles(files) {
  files.forEach(file => {
    // Avoid duplicates
    if (!selectedFiles.find(f => f.path === file.path)) {
      selectedFiles.push(file);
    }
  });

  updateFileList();
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  updateFileList();
}

function clearFiles() {
  selectedFiles = [];
  updateFileList();
}

function updateFileList() {
  if (selectedFiles.length === 0) {
    fileListContainer.style.display = 'none';
    processingControls.style.display = 'none';
    return;
  }

  fileListContainer.style.display = 'block';
  processingControls.style.display = 'flex';

  fileList.innerHTML = selectedFiles.map((file, index) => `
    <div class="file-item">
      <span class="file-icon">${file.name.endsWith('.pdf') ? '📄' : '🖼️'}</span>
      <div class="file-info">
        <div class="file-name">${escapeHtml(file.name)}</div>
        <div class="file-size">${formatFileSize(file.size)}</div>
      </div>
      <button class="file-remove" onclick="removeFile(${index})">✕</button>
    </div>
  `).join('');

  const totalSize = selectedFiles.reduce((sum, f) => sum + f.size, 0);
  fileSummary.textContent = `${selectedFiles.length}개 파일, ${formatFileSize(totalSize)}`;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Processing
async function startProcessing() {
  if (selectedFiles.length === 0 || isProcessing) return;

  isProcessing = true;

  // UI Updates
  document.getElementById('startBtn').style.display = 'none';
  document.getElementById('cancelBtn').style.display = 'inline-block';
  progressSection.style.display = 'block';
  resultsSection.style.display = 'none';

  // Reset progress
  updateProgress({
    phase: 'starting',
    message: '처리 시작 중...',
    percent: 0
  });

  // Start processing
  const result = await window.lawOCR.startProcessing(selectedFiles);

  if (!result.success) {
    handleError(result.error);
  }
}

async function cancelProcessing() {
  if (!isProcessing) return;

  await window.lawOCR.cancelProcessing();
  isProcessing = false;

  document.getElementById('startBtn').style.display = 'inline-block';
  document.getElementById('cancelBtn').style.display = 'none';
  progressSection.style.display = 'none';

  updateProgress({
    phase: 'cancelled',
    message: '취소됨',
    percent: 0
  });
}

function handleProgress(progress) {
  updateProgress(progress);
}

function updateProgress(progress) {
  const progressTitle = document.getElementById('progressTitle');
  const progressPercent = document.getElementById('progressPercent');
  const progressFill = document.getElementById('progressFill');
  const progressDetails = document.getElementById('progressDetails');

  progressTitle.textContent = progress.message || '처리 중...';
  progressPercent.textContent = `${progress.percent || 0}%`;
  progressFill.style.width = `${progress.percent || 0}%`;

  let details = '';
  if (progress.phase === 'splitting') {
    details = `PDF 분할 중: ${progress.fileIndex || 1}/${progress.totalFiles || 1} 파일`;
  } else if (progress.phase === 'processing') {
    details = `OCR 처리 중: ${progress.processedChunks || 0}/${progress.totalChunks || 0} 청크`;
    if (progress.currentFile) {
      details += ` (현재: ${progress.currentFile})`;
    }
  } else if (progress.phase === 'merging') {
    details = '결과 병합 중...';
  } else if (progress.phase === 'saving') {
    details = '파일 저장 중...';
  }
  progressDetails.textContent = details;
}

function handleComplete(results) {
  isProcessing = false;

  // UI Updates
  document.getElementById('startBtn').style.display = 'inline-block';
  document.getElementById('cancelBtn').style.display = 'none';

  updateProgress({
    phase: 'complete',
    message: '완료!',
    percent: 100
  });

  // Show results
  resultsSection.style.display = 'block';
  displayResults(results);

  // Update stats
  updateStats(results);
}

function displayResults(results) {
  const resultsList = document.getElementById('resultsList');

  if (!results.files || results.files.length === 0) {
    resultsList.innerHTML = '<p class="empty-state">변환된 파일이 없습니다.</p>';
    return;
  }

  resultsList.innerHTML = results.files.map(file => {
    const escapedPath = file.outputPath.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
    return `
    <div class="result-item">
      <span class="result-icon">✅</span>
      <div class="result-info">
        <div class="result-name">${escapeHtml(file.fileName)}</div>
        <div class="result-path">${escapeHtml(file.outputPath)}</div>
      </div>
      <button class="btn result-action" onclick="openFile('${escapedPath}')">
        열기
      </button>
    </div>
  `;
  }).join('');
}

function handleError(error) {
  isProcessing = false;

  document.getElementById('startBtn').style.display = 'inline-block';
  document.getElementById('cancelBtn').style.display = 'none';

  updateProgress({
    phase: 'error',
    message: '오류 발생',
    percent: 0
  });

  alert(`처리 중 오류가 발생했습니다: ${error}`);
}

// Results
async function openOutputFolder() {
  await window.lawOCR.openOutputFolder();
}

async function openFile(filePath) {
  await window.lawOCR.openFile(filePath);
}

// Stats (simplified - could be enhanced with persistent storage)
function updateStats(results) {
  const totalDocs = parseInt(localStorage.getItem('totalDocs') || 0) + (results.files?.length || 0);
  const totalPages = parseInt(localStorage.getItem('totalPages') || 0) + (results.totalPages || 0);

  localStorage.setItem('totalDocs', totalDocs);
  localStorage.setItem('totalPages', totalPages);

  document.getElementById('statTotalDocs').textContent = totalDocs;
  document.getElementById('statTotalPages').textContent = totalPages;
}

// Load stats on init
function loadStats() {
  document.getElementById('statTotalDocs').textContent = localStorage.getItem('totalDocs') || 0;
  document.getElementById('statTotalPages').textContent = localStorage.getItem('totalPages') || 0;
}

loadStats();
