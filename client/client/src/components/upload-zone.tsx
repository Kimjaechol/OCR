import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileText, Loader2, Check, AlertCircle, Download, Eye, FolderOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

// API base URL - adjust based on environment
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface OCRResult {
  task_id: string;
  status: string;
  progress: number;
  current_page?: number;
  total_pages?: number;
  result?: string;
  full_result?: {
    markdown?: string;
    html?: string;
    markdown_file?: string;
    html_file?: string;
    output_dir?: string;
    page_count?: number;
    processing_time?: number;
  };
  error?: string;
}

export function UploadZone() {
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<"idle" | "uploading" | "processing" | "completed" | "error">("idle");
  const [progress, setProgress] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [result, setResult] = useState<OCRResult | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Clean up polling on unmount
  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  // Cleanup polling on component unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  // HTML escape function to prevent XSS
  const escapeHtml = (text: string): string => {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  };

  // Poll for task status
  const pollTaskStatus = useCallback(async (taskId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/ocr/status/${taskId}`);
      const data: OCRResult = await response.json();

      setProgress(data.progress || 0);
      setCurrentPage(data.current_page || 0);
      setTotalPages(data.total_pages || 0);
      setStatusMessage(data.status || "처리 중...");

      if (data.state === "completed" || data.status === "completed") {
        stopPolling();
        setStatus("completed");
        setResult(data);
        setProgress(100);
      } else if (data.state === "failed" || data.status === "failed") {
        stopPolling();
        setStatus("error");
        setErrorMessage(data.error || "처리 중 오류가 발생했습니다.");
      }
    } catch (error) {
      console.error("Polling error:", error);
    }
  }, [stopPolling]);

  // Start polling
  const startPolling = useCallback((taskId: string) => {
    stopPolling();
    pollingRef.current = setInterval(() => pollTaskStatus(taskId), 1500);
  }, [pollTaskStatus, stopPolling]);

  // Upload file to API
  const uploadFile = async (file: File) => {
    setStatus("uploading");
    setProgress(0);
    setSelectedFile(file);
    setErrorMessage("");
    setStatusMessage("파일 업로드 중...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/ocr/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "업로드 실패");
      }

      const data = await response.json();

      setStatus("processing");
      setStatusMessage("OCR 처리 시작...");
      setTotalPages(data.estimated_pages || 0);

      // Start polling for status
      startPolling(data.task_id);

    } catch (error) {
      console.error("Upload error:", error);
      setStatus("error");
      setErrorMessage(error instanceof Error ? error.message : "업로드 중 오류 발생");
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (isValidFile(file)) {
        uploadFile(file);
      } else {
        setStatus("error");
        setErrorMessage("지원하지 않는 파일 형식입니다. PDF 또는 이미지 파일을 업로드하세요.");
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (isValidFile(file)) {
        uploadFile(file);
      } else {
        setStatus("error");
        setErrorMessage("지원하지 않는 파일 형식입니다. PDF 또는 이미지 파일을 업로드하세요.");
      }
    }
  };

  const isValidFile = (file: File): boolean => {
    const validTypes = [
      'application/pdf',
      'image/png',
      'image/jpeg',
      'image/tiff',
      'image/bmp',
      'image/webp'
    ];
    const validExtensions = /\.(pdf|png|jpg|jpeg|tiff|bmp|webp)$/i;
    return validTypes.includes(file.type) || validExtensions.test(file.name);
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleReset = () => {
    stopPolling();
    setStatus("idle");
    setProgress(0);
    setCurrentPage(0);
    setTotalPages(0);
    setResult(null);
    setSelectedFile(null);
    setErrorMessage("");
    setStatusMessage("");
  };

  const downloadResult = async (format: 'markdown' | 'html') => {
    if (!result?.task_id) return;

    try {
      const response = await fetch(`${API_BASE_URL}/ocr/download/${result.task_id}?format=${format}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ocr_result.${format === 'markdown' ? 'md' : 'html'}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Download error:", error);
    }
  };

  const viewResult = async (format: 'markdown' | 'html') => {
    if (!result?.task_id) return;

    try {
      const response = await fetch(`${API_BASE_URL}/ocr/result/${result.task_id}?format=${format}`);
      if (format === 'html') {
        const html = await response.text();
        const newWindow = window.open('', '_blank');
        if (newWindow) {
          newWindow.document.write(html);
          newWindow.document.close();
        }
      } else {
        const data = await response.json();
        const newWindow = window.open('', '_blank');
        if (newWindow) {
          // Escape HTML to prevent XSS
          const escapedMarkdown = escapeHtml(data.markdown || '');
          newWindow.document.write(`<pre style="white-space: pre-wrap; font-family: monospace; padding: 20px;">${escapedMarkdown}</pre>`);
          newWindow.document.close();
        }
      }
    } catch (error) {
      console.error("View error:", error);
    }
  };

  return (
    <section id="upload" className="py-24 bg-card/50 relative border-t border-white/5">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-serif font-bold text-white mb-4">직접 체험해보세요</h2>
          <p className="text-muted-foreground">사건 기록 PDF를 드래그하여 변환 속도를 확인하세요.</p>
        </div>

        <div className="max-w-3xl mx-auto">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp,.webp"
            className="hidden"
          />

          <div
            className={`
              relative rounded-xl border-2 border-dashed transition-all duration-300 p-12 text-center
              ${isDragging ? "border-primary bg-primary/5 scale-[1.02]" : "border-white/10 hover:border-primary/50 hover:bg-white/5"}
              ${status === "completed" ? "border-green-500/50 bg-green-500/5" : ""}
              ${status === "error" ? "border-red-500/50 bg-red-500/5" : ""}
            `}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <AnimatePresence mode="wait">
              {status === "idle" && (
                <motion.div
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4"
                >
                  <div className="w-20 h-20 mx-auto rounded-full bg-primary/10 flex items-center justify-center mb-4">
                    <Upload className="w-10 h-10 text-primary" />
                  </div>
                  <h3 className="text-xl font-medium text-white">사건 기록 PDF 업로드</h3>
                  <p className="text-muted-foreground">
                    또는 클릭하여 파일 선택 (최대 5,000페이지)
                  </p>
                  <Button onClick={handleButtonClick} variant="outline" className="mt-4 border-primary/30 text-primary hover:bg-primary hover:text-primary-foreground">
                    파일 선택하기
                  </Button>
                </motion.div>
              )}

              {(status === "uploading" || status === "processing") && (
                <motion.div
                  key="processing"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-6 py-8"
                >
                  <div className="flex items-center justify-center gap-3 mb-2">
                    <Loader2 className="w-6 h-6 text-primary animate-spin" />
                    <span className="text-lg font-medium text-white">
                      {status === "uploading" ? "업로드 중..." : "OCR 분석 및 변환 중..."}
                    </span>
                  </div>
                  <div className="w-full max-w-md mx-auto space-y-2">
                    <Progress value={progress} className="h-2 bg-white/10" />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>{selectedFile?.name || "처리 중..."}</span>
                      <span>
                        {totalPages > 0
                          ? `${currentPage}/${totalPages} 페이지 (${Math.round(progress)}%)`
                          : `${Math.round(progress)}%`
                        }
                      </span>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground">{statusMessage}</p>
                  <div className="grid grid-cols-3 gap-4 max-w-lg mx-auto mt-8">
                    <div className={`bg-background/50 p-3 rounded border border-white/5 text-xs ${progress > 20 ? 'text-primary' : 'text-muted-foreground'}`}>
                      텍스트 추출 중...
                    </div>
                    <div className={`bg-background/50 p-3 rounded border border-white/5 text-xs ${progress > 50 ? 'text-primary' : 'text-muted-foreground'}`}>
                      표/서식 분석...
                    </div>
                    <div className={`bg-background/50 p-3 rounded border border-white/5 text-xs ${progress > 80 ? 'text-primary' : 'text-muted-foreground'}`}>
                      Gemini 교정...
                    </div>
                  </div>
                </motion.div>
              )}

              {status === "completed" && (
                <motion.div
                  key="completed"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="space-y-6"
                >
                  <div className="w-20 h-20 mx-auto rounded-full bg-green-500/20 flex items-center justify-center mb-4">
                    <Check className="w-10 h-10 text-green-400" />
                  </div>
                  <h3 className="text-2xl font-serif font-bold text-white">변환 완료!</h3>
                  <p className="text-muted-foreground">
                    {result?.full_result?.page_count || 0}페이지가 성공적으로 변환되었습니다.
                    {result?.full_result?.processing_time && (
                      <span className="block text-sm mt-1">
                        처리 시간: {result.full_result.processing_time.toFixed(1)}초
                      </span>
                    )}
                  </p>

                  {/* Download buttons */}
                  <div className="flex items-center justify-center gap-4 mt-8">
                    <Button
                      onClick={() => downloadResult('markdown')}
                      className="bg-primary text-primary-foreground hover:bg-primary/90 gap-2"
                    >
                      <Download className="w-4 h-4" /> Markdown 다운로드
                    </Button>
                    <Button
                      onClick={() => downloadResult('html')}
                      variant="outline"
                      className="border-white/20 text-white hover:bg-white/10 gap-2"
                    >
                      <Download className="w-4 h-4" /> HTML 다운로드
                    </Button>
                  </div>

                  {/* View buttons */}
                  <div className="flex items-center justify-center gap-4">
                    <Button
                      onClick={() => viewResult('markdown')}
                      variant="ghost"
                      className="text-muted-foreground hover:text-white gap-2"
                    >
                      <Eye className="w-4 h-4" /> Markdown 미리보기
                    </Button>
                    <Button
                      onClick={() => viewResult('html')}
                      variant="ghost"
                      className="text-muted-foreground hover:text-white gap-2"
                    >
                      <Eye className="w-4 h-4" /> HTML 미리보기
                    </Button>
                  </div>

                  {/* Output info */}
                  {result?.full_result?.output_dir && (
                    <div className="mt-6 p-4 bg-background/50 rounded-lg border border-white/5 text-left max-w-xl mx-auto">
                      <div className="flex items-center gap-2 mb-2 text-sm text-muted-foreground">
                        <FolderOpen className="w-4 h-4" />
                        <span>출력 폴더:</span>
                      </div>
                      <p className="text-xs font-mono text-white/70 break-all">
                        {result.full_result.output_dir}
                      </p>
                    </div>
                  )}

                  {/* Preview */}
                  <div className="mt-6 p-4 bg-background/50 rounded-lg border border-white/5 text-left max-w-xl mx-auto">
                    <div className="flex items-center gap-2 mb-2 border-b border-white/5 pb-2">
                      <span className="text-xs font-bold text-primary">AI PREVIEW</span>
                    </div>
                    <p className="text-sm text-muted-foreground font-mono leading-relaxed max-h-40 overflow-y-auto">
                      {result?.result?.substring(0, 500) || result?.full_result?.markdown?.substring(0, 500) || "결과를 불러오는 중..."}
                      {((result?.result?.length || 0) > 500 || (result?.full_result?.markdown?.length || 0) > 500) && (
                        <span className="text-primary/50">... (더 보기는 다운로드 후 확인)</span>
                      )}
                    </p>
                  </div>

                  <button
                    onClick={handleReset}
                    className="text-sm text-muted-foreground hover:text-white underline underline-offset-4 mt-4"
                  >
                    다른 파일 변환하기
                  </button>
                </motion.div>
              )}

              {status === "error" && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="space-y-6"
                >
                  <div className="w-20 h-20 mx-auto rounded-full bg-red-500/20 flex items-center justify-center mb-4">
                    <AlertCircle className="w-10 h-10 text-red-400" />
                  </div>
                  <h3 className="text-2xl font-serif font-bold text-white">오류 발생</h3>
                  <p className="text-red-400">{errorMessage}</p>

                  <button
                    onClick={handleReset}
                    className="text-sm text-muted-foreground hover:text-white underline underline-offset-4 mt-4"
                  >
                    다시 시도하기
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* File type info */}
          <div className="mt-6 text-center">
            <p className="text-xs text-muted-foreground">
              지원 형식: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
