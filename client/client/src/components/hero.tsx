import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ArrowRight, CheckCircle2 } from "lucide-react";
// @ts-ignore
import heroBg from "@assets/generated_images/abstract_digital_legal_technology_background.png";

export function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
      {/* Background Overlay */}
      <div className="absolute inset-0 z-0">
        <img 
          src={heroBg} 
          alt="Legal Tech Background" 
          className="w-full h-full object-cover opacity-40"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background/80 via-background/60 to-background" />
      </div>

      <div className="container relative z-10 px-6 py-20 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-4xl mx-auto space-y-8"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-primary/30 bg-primary/10 backdrop-blur-sm">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-xs font-medium text-primary tracking-wider uppercase">변호사를 위한 프리미엄 OCR 솔루션</span>
          </div>

          <h1 className="text-5xl md:text-7xl font-serif font-bold leading-tight text-white drop-shadow-lg">
            기록 검토의 혁명,<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-amber-200">
              초정밀 법률 문서 변환
            </span>
          </h1>

          <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            99.99% 정확도로 5,000페이지 사건 기록을 단 몇 분 만에.<br />
            외부 서버 전송 없이, 당신의 PC에서 안전하게 AI가 읽을 수 있는 포맷으로 변환합니다.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Button size="lg" className="h-14 px-8 text-lg bg-primary text-primary-foreground hover:bg-primary/90 shadow-[0_0_20px_rgba(234,179,8,0.3)] transition-all hover:scale-105">
              지금 시작하기 <ArrowRight className="ml-2 w-5 h-5" />
            </Button>
            <Button size="lg" variant="outline" className="h-14 px-8 text-lg border-white/20 text-white hover:bg-white/10 backdrop-blur-sm">
              데모 영상 보기
            </Button>
          </div>

          <div className="pt-12 flex flex-wrap justify-center gap-8 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-primary" />
              <span>로컬 처리로 완벽한 보안</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-primary" />
              <span>AI 최적화 (Markdown/HTML)</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-primary" />
              <span>장당 50원의 합리적 비용</span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}