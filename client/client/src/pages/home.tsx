import { Nav } from "@/components/nav";
import { Hero } from "@/components/hero";
import { UploadZone } from "@/components/upload-zone";
import { PricingCard } from "@/components/pricing-card";
import { Shield, Cpu, FileJson, Lock } from "lucide-react";

function FeatureItem({ icon: Icon, title, desc }: { icon: any, title: string, desc: string }) {
  return (
    <div className="flex flex-col items-start p-6 rounded-xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.05] transition-colors">
      <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
        <Icon className="w-6 h-6 text-primary" />
      </div>
      <h3 className="text-xl font-serif font-bold text-white mb-2">{title}</h3>
      <p className="text-muted-foreground leading-relaxed">{desc}</p>
    </div>
  );
}

export default function Home() {
  return (
    <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/30">
      <Nav />
      
      <main>
        <Hero />
        
        {/* Features Grid */}
        <section id="features" className="py-24 bg-background">
          <div className="container mx-auto px-6">
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              <FeatureItem 
                icon={Lock}
                title="로컬 보안 처리"
                desc="클라우드 전송 없이 사용자의 PC 내에서 모든 변환 작업이 이루어집니다. 의뢰인의 비밀을 완벽하게 보호하세요."
              />
              <FeatureItem 
                icon={Cpu}
                title="초고속 엔진"
                desc="자체 최적화된 OCR 엔진으로 100페이지 문서를 단 30초 만에 처리합니다. 기다림 없는 업무 환경을 경험하세요."
              />
              <FeatureItem 
                icon={FileJson}
                title="AI 최적화 포맷"
                desc="단순 텍스트가 아닙니다. 제목, 문단, 표가 구조화된 Markdown으로 변환되어 LLM(ChatGPT, Claude)이 즉시 이해합니다."
              />
              <FeatureItem 
                icon={Shield}
                title="99.99% 정확도"
                desc="법률 용어에 특화된 학습 모델을 사용하여, 깨진 글자나 노이즈가 있는 스캔 문서도 정확하게 인식합니다."
              />
            </div>
          </div>
        </section>

        <UploadZone />
        
        <PricingCard />

        {/* CTA Section */}
        <section className="py-24 bg-primary/5 border-y border-primary/10">
          <div className="container mx-auto px-6 text-center">
            <h2 className="text-3xl md:text-4xl font-serif font-bold text-white mb-6">
              변호사님의 시간은 더 가치 있는 곳에 쓰여야 합니다.
            </h2>
            <p className="text-muted-foreground mb-10 max-w-2xl mx-auto">
              단순 타이핑과 문서 정리는 LexScan Pro에게 맡기세요.<br/>
              지금 바로 가장 안전하고 똑똑한 디지털 법률 비서를 고용하세요.
            </p>
            <button className="px-8 py-4 bg-primary text-primary-foreground font-bold rounded-md text-lg hover:bg-primary/90 transition-transform hover:scale-105 shadow-lg shadow-primary/20">
              무료로 시작하기
            </button>
          </div>
        </section>
      </main>

      <footer className="py-12 bg-background border-t border-white/5 text-center md:text-left">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="space-y-2">
              <span className="font-serif text-xl font-bold text-white">LexScan Pro</span>
              <p className="text-sm text-muted-foreground">© 2024 LexScan Inc. All rights reserved.</p>
            </div>
            <div className="flex gap-8 text-sm text-muted-foreground">
              <a href="#" className="hover:text-primary">이용약관</a>
              <a href="#" className="hover:text-primary">개인정보처리방침</a>
              <a href="#" className="hover:text-primary">고객센터</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}