import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileText, Loader2, Check, FileCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

export function UploadZone() {
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<"idle" | "processing" | "completed">("idle");
  const [progress, setProgress] = useState(0);

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
    startSimulation();
  };

  const startSimulation = () => {
    setStatus("processing");
    let p = 0;
    const interval = setInterval(() => {
      p += Math.random() * 10;
      if (p >= 100) {
        p = 100;
        clearInterval(interval);
        setTimeout(() => setStatus("completed"), 500);
      }
      setProgress(p);
    }, 200);
  };

  return (
    <section className="py-24 bg-card/50 relative border-t border-white/5">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-serif font-bold text-white mb-4">직접 체험해보세요</h2>
          <p className="text-muted-foreground">사건 기록 PDF를 드래그하여 변환 속도를 확인하세요.</p>
        </div>

        <div className="max-w-3xl mx-auto">
          <div
            className={`
              relative rounded-xl border-2 border-dashed transition-all duration-300 p-12 text-center
              ${isDragging ? "border-primary bg-primary/5 scale-[1.02]" : "border-white/10 hover:border-primary/50 hover:bg-white/5"}
              ${status === "completed" ? "border-green-500/50 bg-green-500/5" : ""}
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
                  <Button onClick={startSimulation} variant="outline" className="mt-4 border-primary/30 text-primary hover:bg-primary hover:text-primary-foreground">
                    파일 선택하기
                  </Button>
                </motion.div>
              )}

              {status === "processing" && (
                <motion.div
                  key="processing"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-6 py-8"
                >
                  <div className="flex items-center justify-center gap-3 mb-2">
                    <Loader2 className="w-6 h-6 text-primary animate-spin" />
                    <span className="text-lg font-medium text-white">OCR 분석 및 변환 중...</span>
                  </div>
                  <div className="w-full max-w-md mx-auto space-y-2">
                    <Progress value={progress} className="h-2 bg-white/10" />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>2024고합1234.pdf</span>
                      <span>{Math.round(progress)}%</span>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4 max-w-lg mx-auto mt-8">
                    <div className="bg-background/50 p-3 rounded border border-white/5 text-xs text-muted-foreground">
                      텍스트 추출 중...
                    </div>
                    <div className="bg-background/50 p-3 rounded border border-white/5 text-xs text-muted-foreground">
                      표/서식 분석...
                    </div>
                    <div className="bg-background/50 p-3 rounded border border-white/5 text-xs text-muted-foreground">
                      Markdown 변환...
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
                    AI가 읽을 수 있는 최적의 포맷으로 변환되었습니다.
                  </p>
                  
                  <div className="flex items-center justify-center gap-4 mt-8">
                    <Button className="bg-primary text-primary-foreground hover:bg-primary/90 gap-2">
                      <FileText className="w-4 h-4" /> Markdown 다운로드
                    </Button>
                    <Button variant="outline" className="border-white/20 text-white hover:bg-white/10 gap-2">
                      <FileCheck className="w-4 h-4" /> 원본 비교하기
                    </Button>
                  </div>

                  <div className="mt-6 p-4 bg-background/50 rounded-lg border border-white/5 text-left max-w-xl mx-auto">
                    <div className="flex items-center gap-2 mb-2 border-b border-white/5 pb-2">
                      <span className="text-xs font-bold text-primary">AI PREVIEW</span>
                    </div>
                    <p className="text-sm text-muted-foreground font-mono leading-relaxed">
                      # 사건 번호: 2024고합1234<br/>
                      ## 1. 공소사실의 요지<br/>
                      피고인은 2023. 5. 12. 경 서울 서초구...<br/>
                      <span className="text-primary/50">... (AI가 완벽하게 인식 가능한 구조화된 텍스트)</span>
                    </p>
                  </div>
                  
                  <button 
                    onClick={() => {setStatus("idle"); setProgress(0);}}
                    className="text-sm text-muted-foreground hover:text-white underline underline-offset-4 mt-4"
                  >
                    다른 파일 변환하기
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </section>
  );
}