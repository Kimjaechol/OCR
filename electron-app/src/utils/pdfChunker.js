/**
 * LawOCR - PDF Chunker
 * ====================
 * Splits large PDFs into smaller chunks for parallel processing
 */

const { PDFDocument } = require('pdf-lib');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { v4: uuidv4 } = require('uuid');

class PDFChunker {
  constructor(chunkSize = 10) {
    this.chunkSize = chunkSize;  // Pages per chunk
    this.tempDir = path.join(os.tmpdir(), 'lawocr-chunks');
    this.ensureTempDir();
  }

  ensureTempDir() {
    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
  }

  /**
   * Get PDF page count without loading entire document
   */
  async getPageCount(pdfPath) {
    const pdfBytes = fs.readFileSync(pdfPath);
    const pdfDoc = await PDFDocument.load(pdfBytes, { ignoreEncryption: true });
    return pdfDoc.getPageCount();
  }

  /**
   * Split a PDF into chunks of specified size
   * Returns array of chunk file paths
   */
  async splitPDF(pdfPath, onProgress = null) {
    const pdfBytes = fs.readFileSync(pdfPath);
    const pdfDoc = await PDFDocument.load(pdfBytes, { ignoreEncryption: true });
    const totalPages = pdfDoc.getPageCount();
    const chunks = [];

    const jobId = uuidv4();
    const baseName = path.basename(pdfPath, '.pdf');

    // Calculate number of chunks needed
    const numChunks = Math.ceil(totalPages / this.chunkSize);

    for (let i = 0; i < numChunks; i++) {
      const startPage = i * this.chunkSize;
      const endPage = Math.min(startPage + this.chunkSize, totalPages);

      // Create new PDF for this chunk
      const chunkDoc = await PDFDocument.create();
      const pageIndices = [];

      for (let j = startPage; j < endPage; j++) {
        pageIndices.push(j);
      }

      const copiedPages = await chunkDoc.copyPages(pdfDoc, pageIndices);
      copiedPages.forEach(page => chunkDoc.addPage(page));

      // Save chunk to temp file
      const chunkFileName = `${baseName}_chunk_${String(i + 1).padStart(4, '0')}_${jobId}.pdf`;
      const chunkPath = path.join(this.tempDir, chunkFileName);
      const chunkBytes = await chunkDoc.save();

      fs.writeFileSync(chunkPath, chunkBytes);

      chunks.push({
        index: i,
        path: chunkPath,
        startPage: startPage + 1,  // 1-indexed
        endPage: endPage,
        pageCount: endPage - startPage,
        originalFile: pdfPath,
        originalFileName: baseName
      });

      // Report progress
      if (onProgress) {
        onProgress({
          phase: 'splitting',
          current: i + 1,
          total: numChunks,
          percent: Math.round(((i + 1) / numChunks) * 100),
          message: `PDF 분할 중: ${i + 1}/${numChunks} 청크`
        });
      }
    }

    return {
      originalFile: pdfPath,
      originalFileName: baseName,
      totalPages,
      chunkSize: this.chunkSize,
      chunks
    };
  }

  /**
   * Split multiple PDFs
   */
  async splitMultiplePDFs(pdfPaths, onProgress = null) {
    const allChunks = [];
    let totalChunks = 0;
    let processedChunks = 0;

    // First pass: count total pages
    for (const pdfPath of pdfPaths) {
      try {
        const pageCount = await this.getPageCount(pdfPath);
        totalChunks += Math.ceil(pageCount / this.chunkSize);
      } catch (error) {
        console.error(`Error reading ${pdfPath}:`, error);
      }
    }

    // Second pass: actually split
    for (let i = 0; i < pdfPaths.length; i++) {
      const pdfPath = pdfPaths[i];

      try {
        const result = await this.splitPDF(pdfPath, (progress) => {
          if (onProgress) {
            const overallPercent = Math.round(
              ((processedChunks + progress.current) / totalChunks) * 100
            );
            onProgress({
              phase: 'splitting',
              file: path.basename(pdfPath),
              fileIndex: i + 1,
              totalFiles: pdfPaths.length,
              chunkProgress: progress,
              overallPercent,
              message: `[${i + 1}/${pdfPaths.length}] ${path.basename(pdfPath)} 분할 중...`
            });
          }
        });

        processedChunks += result.chunks.length;
        allChunks.push(result);

      } catch (error) {
        console.error(`Error splitting ${pdfPath}:`, error);
        allChunks.push({
          originalFile: pdfPath,
          originalFileName: path.basename(pdfPath, '.pdf'),
          error: error.message,
          chunks: []
        });
      }
    }

    return allChunks;
  }

  /**
   * Clean up temporary chunk files
   */
  cleanupChunks(chunks) {
    chunks.forEach(chunk => {
      try {
        if (fs.existsSync(chunk.path)) {
          fs.unlinkSync(chunk.path);
        }
      } catch (error) {
        console.error(`Error deleting chunk ${chunk.path}:`, error);
      }
    });
  }

  /**
   * Clean up all temporary files
   */
  cleanupAll() {
    try {
      if (fs.existsSync(this.tempDir)) {
        const files = fs.readdirSync(this.tempDir);
        files.forEach(file => {
          const filePath = path.join(this.tempDir, file);
          fs.unlinkSync(filePath);
        });
      }
    } catch (error) {
      console.error('Error cleaning up temp directory:', error);
    }
  }
}

module.exports = { PDFChunker };
