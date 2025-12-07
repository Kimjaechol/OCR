/**
 * LawOCR - RunPod API Client
 * ==========================
 * Handles communication with RunPod FastAPI server
 */

const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

class RunPodClient {
  constructor(serverUrl, timeout = 300000) {  // 5 minute default timeout
    this.serverUrl = serverUrl.replace(/\/$/, '');  // Remove trailing slash
    this.timeout = timeout;
    this.client = axios.create({
      baseURL: this.serverUrl,
      timeout: this.timeout
    });
  }

  /**
   * Health check
   */
  async healthCheck() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      throw new Error(`서버 연결 실패: ${error.message}`);
    }
  }

  /**
   * Upload a single file (PDF chunk or image)
   */
  async uploadFile(filePath, options = {}) {
    const {
      startPage = 1,
      endPage = null,
      applyGemini = true
    } = options;

    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));

    const params = new URLSearchParams({
      start_page: startPage,
      apply_gemini: applyGemini
    });

    if (endPage) {
      params.append('end_page', endPage);
    }

    try {
      const response = await this.client.post(
        `/ocr/upload?${params.toString()}`,
        form,
        {
          headers: form.getHeaders(),
          maxContentLength: Infinity,
          maxBodyLength: Infinity
        }
      );
      return response.data;
    } catch (error) {
      throw new Error(`파일 업로드 실패: ${error.message}`);
    }
  }

  /**
   * Upload multiple images
   */
  async uploadMultipleImages(imagePaths, applyGemini = true) {
    const form = new FormData();

    imagePaths.forEach(imagePath => {
      form.append('files', fs.createReadStream(imagePath));
    });

    const params = new URLSearchParams({
      apply_gemini: applyGemini
    });

    try {
      const response = await this.client.post(
        `/ocr/upload-multiple?${params.toString()}`,
        form,
        {
          headers: form.getHeaders(),
          maxContentLength: Infinity,
          maxBodyLength: Infinity
        }
      );
      return response.data;
    } catch (error) {
      throw new Error(`이미지 업로드 실패: ${error.message}`);
    }
  }

  /**
   * Get task status
   */
  async getTaskStatus(taskId) {
    try {
      const response = await this.client.get(`/ocr/status/${taskId}`);
      return response.data;
    } catch (error) {
      throw new Error(`작업 상태 조회 실패: ${error.message}`);
    }
  }

  /**
   * Get task result
   */
  async getTaskResult(taskId, format = 'json') {
    try {
      const response = await this.client.get(
        `/ocr/result/${taskId}`,
        { params: { format } }
      );
      return response.data;
    } catch (error) {
      throw new Error(`결과 조회 실패: ${error.message}`);
    }
  }

  /**
   * Cancel a task
   */
  async cancelTask(taskId) {
    try {
      const response = await this.client.delete(`/ocr/cancel/${taskId}`);
      return response.data;
    } catch (error) {
      throw new Error(`작업 취소 실패: ${error.message}`);
    }
  }

  /**
   * Poll for task completion
   * Compatible with both 'state' and 'status' fields from server
   */
  async waitForCompletion(taskId, onProgress = null, pollInterval = 2000) {
    while (true) {
      const statusRes = await this.getTaskStatus(taskId);

      if (onProgress) {
        onProgress(statusRes);
      }

      // Check both 'state' and 'status' for compatibility
      const taskState = statusRes.state || statusRes.status;

      if (taskState === 'completed' || taskState === 'SUCCESS') {
        // Return markdown result directly
        return statusRes.result || statusRes.full_result?.markdown || '';
      }

      if (taskState === 'failed' || taskState === 'FAILURE') {
        throw new Error(statusRes.error || '처리 실패');
      }

      // Wait before next poll (2 seconds like user's code)
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }
  }

  /**
   * Upload chunk and wait for result (combined operation)
   */
  async processChunk(chunkPath, options = {}, onProgress = null) {
    // Upload
    const uploadResult = await this.uploadFile(chunkPath, options);
    const taskId = uploadResult.task_id;

    // Wait for completion - result is now markdown string
    const markdown = await this.waitForCompletion(taskId, onProgress);

    return {
      taskId,
      markdown: typeof markdown === 'string' ? markdown : (markdown?.markdown || '')
    };
  }
}

module.exports = { RunPodClient };
