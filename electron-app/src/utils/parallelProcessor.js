/**
 * LawOCR - Parallel Processor
 * ===========================
 * Orchestrates parallel PDF chunk processing for maximum speed
 */

const fs = require('fs');
const path = require('path');
const { PDFChunker } = require('./pdfChunker');
const { RunPodClient } = require('./runpodClient');

class ParallelProcessor {
  constructor(settings, onProgress) {
    this.settings = settings;
    this.onProgress = onProgress;
    this.chunker = new PDFChunker(settings.chunkSize);
    this.client = new RunPodClient(settings.serverUrl);
    this.activeJobs = new Map();
    this.results = [];
    this.errors = [];
    this.cancelled = false;
  }

  /**
   * Process multiple files with parallel chunk processing
   */
  async processFiles(files) {
    this.cancelled = false;
    this.results = [];
    this.errors = [];

    const pdfFiles = files.filter(f => f.path.toLowerCase().endsWith('.pdf'));
    const imageFiles = files.filter(f => !f.path.toLowerCase().endsWith('.pdf'));

    // Phase 1: Split all PDFs into chunks
    this.reportProgress({
      phase: 'splitting',
      message: 'PDF 파일 분할 중...',
      percent: 0
    });

    const allChunks = [];
    let splitResults = [];

    if (pdfFiles.length > 0) {
      splitResults = await this.chunker.splitMultiplePDFs(
        pdfFiles.map(f => f.path),
        (progress) => this.reportProgress(progress)
      );

      // Flatten chunks from all PDFs
      splitResults.forEach(result => {
        if (result.chunks) {
          allChunks.push(...result.chunks);
        }
      });
    }

    // Add images as single-page "chunks"
    imageFiles.forEach((file, index) => {
      allChunks.push({
        index: pdfFiles.length + index,
        path: file.path,
        startPage: 1,
        endPage: 1,
        pageCount: 1,
        originalFile: file.path,
        originalFileName: path.basename(file.path, path.extname(file.path)),
        isImage: true
      });
    });

    const totalChunks = allChunks.length;

    if (totalChunks === 0) {
      return {
        success: true,
        files: [],
        message: '처리할 파일이 없습니다.'
      };
    }

    // Phase 2: Process chunks in parallel
    this.reportProgress({
      phase: 'processing',
      message: 'OCR 처리 시작...',
      totalChunks,
      processedChunks: 0,
      percent: 0
    });

    const chunkResults = await this.processChunksParallel(allChunks, totalChunks);

    // Phase 3: Merge results by original file
    this.reportProgress({
      phase: 'merging',
      message: '결과 병합 중...',
      percent: 95
    });

    const mergedResults = this.mergeResults(splitResults, chunkResults, imageFiles);

    // Phase 4: Save results to output directory
    this.reportProgress({
      phase: 'saving',
      message: '결과 저장 중...',
      percent: 98
    });

    const savedFiles = await this.saveResults(mergedResults);

    // Cleanup temporary chunks
    this.chunker.cleanupAll();

    this.reportProgress({
      phase: 'complete',
      message: '완료!',
      percent: 100
    });

    return {
      success: true,
      files: savedFiles,
      totalPages: chunkResults.reduce((sum, r) => sum + (r.pageCount || 0), 0),
      errors: this.errors
    };
  }

  /**
   * Process chunks in parallel with concurrency limit
   */
  async processChunksParallel(chunks, totalChunks) {
    const results = new Array(chunks.length);
    let completedCount = 0;
    let activeCount = 0;
    let chunkIndex = 0;

    return new Promise((resolve, reject) => {
      const processNext = async () => {
        if (this.cancelled) {
          reject(new Error('처리가 취소되었습니다.'));
          return;
        }

        if (chunkIndex >= chunks.length) {
          // No more chunks to process
          if (activeCount === 0) {
            resolve(results.filter(r => r !== undefined));
          }
          return;
        }

        const currentIndex = chunkIndex;
        const chunk = chunks[currentIndex];
        chunkIndex++;
        activeCount++;

        try {
          const result = await this.processOneChunk(chunk);
          results[currentIndex] = {
            ...chunk,
            result,
            success: true
          };
        } catch (error) {
          console.error(`Chunk ${currentIndex} failed:`, error);
          results[currentIndex] = {
            ...chunk,
            error: error.message,
            success: false
          };
          this.errors.push({
            file: chunk.originalFileName,
            chunk: currentIndex,
            error: error.message
          });
        }

        completedCount++;
        activeCount--;

        this.reportProgress({
          phase: 'processing',
          message: `OCR 처리 중: ${completedCount}/${totalChunks}`,
          totalChunks,
          processedChunks: completedCount,
          percent: Math.round((completedCount / totalChunks) * 90) + 5,  // 5-95%
          currentFile: chunk.originalFileName,
          currentChunk: chunk.index + 1
        });

        // Start next chunk
        processNext();
      };

      // Start initial batch of parallel workers
      const initialBatch = Math.min(this.settings.maxParallel, chunks.length);
      for (let i = 0; i < initialBatch; i++) {
        processNext();
      }
    });
  }

  /**
   * Process a single chunk
   */
  async processOneChunk(chunk) {
    const result = await this.client.processChunk(
      chunk.path,
      {
        startPage: 1,  // Chunk already contains correct page range
        endPage: chunk.pageCount,
        applyGemini: this.settings.applyGemini
      },
      (status) => {
        // Optional: Report individual chunk status
      }
    );

    return result;
  }

  /**
   * Merge chunk results back into original file structure
   */
  mergeResults(splitResults, chunkResults, imageFiles) {
    const fileResults = new Map();

    // Group chunks by original file
    chunkResults.forEach(chunkResult => {
      const originalFile = chunkResult.originalFile;

      if (!fileResults.has(originalFile)) {
        fileResults.set(originalFile, {
          originalFile,
          originalFileName: chunkResult.originalFileName,
          chunks: [],
          totalPages: 0
        });
      }

      const fileResult = fileResults.get(originalFile);
      fileResult.chunks.push(chunkResult);
      fileResult.totalPages += chunkResult.pageCount || 0;
    });

    // Sort chunks by index and merge markdown
    const merged = [];

    fileResults.forEach((fileResult, originalFile) => {
      // Sort chunks by startPage
      fileResult.chunks.sort((a, b) => a.startPage - b.startPage);

      // Combine markdown from all chunks
      const combinedMarkdown = fileResult.chunks
        .filter(c => c.success && c.result && c.result.markdown)
        .map((c, idx) => {
          // Add page range header
          const header = `<!-- 페이지 ${c.startPage}-${c.endPage} -->\n`;
          return header + c.result.markdown;
        })
        .join('\n\n---\n\n');

      merged.push({
        originalFile,
        fileName: fileResult.originalFileName,
        totalPages: fileResult.totalPages,
        markdown: combinedMarkdown,
        success: fileResult.chunks.some(c => c.success),
        errors: fileResult.chunks.filter(c => !c.success).map(c => c.error)
      });
    });

    return merged;
  }

  /**
   * Save results to output directory
   */
  async saveResults(mergedResults) {
    const savedFiles = [];
    const timestamp = new Date().toISOString().slice(0, 10).replace(/-/g, '');

    for (const result of mergedResults) {
      if (!result.success || !result.markdown) {
        continue;
      }

      const outputFileName = `${result.fileName}_${timestamp}.md`;
      const outputPath = path.join(this.settings.outputDir, outputFileName);

      try {
        // Add header with metadata
        const content = `# ${result.fileName}\n\n` +
          `> 처리일: ${new Date().toLocaleString('ko-KR')}\n` +
          `> 총 페이지: ${result.totalPages}페이지\n\n` +
          `---\n\n` +
          result.markdown;

        fs.writeFileSync(outputPath, content, 'utf8');

        savedFiles.push({
          originalFile: result.originalFile,
          outputPath,
          fileName: outputFileName,
          totalPages: result.totalPages
        });
      } catch (error) {
        console.error(`Error saving ${outputFileName}:`, error);
        this.errors.push({
          file: result.fileName,
          error: `저장 실패: ${error.message}`
        });
      }
    }

    return savedFiles;
  }

  /**
   * Report progress to main window
   */
  reportProgress(progress) {
    if (this.onProgress) {
      this.onProgress(progress);
    }
  }

  /**
   * Cancel processing
   */
  cancel() {
    this.cancelled = true;
    // TODO: Cancel active API requests
  }
}

module.exports = { ParallelProcessor };
