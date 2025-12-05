import { Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

export function PricingCard() {
  return (
    <section id="pricing" className="py-24 bg-background relative overflow-hidden">
      {/* Decorative glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-primary/5 rounded-full blur-3xl pointer-events-none" />

      <div className="container mx-auto px-6 relative z-10">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-serif font-bold text-white mb-4">
            투명하고 합리적인 요금
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            복잡한 구독 모델이나 숨겨진 비용은 없습니다.<br />
            사용한 만큼만, 장당 50원으로 AI 법률 비서의 기초를 마련하세요.
          </p>
        </div>

        <div className="max-w-lg mx-auto">
          <Card className="bg-card border-primary/20 shadow-2xl overflow-hidden relative group">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary to-transparent" />
            
            <CardHeader className="text-center pt-12 pb-8">
              <CardTitle className="text-lg font-medium text-primary mb-2">Standard Plan</CardTitle>
              <div className="flex items-baseline justify-center gap-1">
                <span className="text-5xl font-bold text-white">50</span>
                <span className="text-xl text-muted-foreground">원</span>
                <span className="text-muted-foreground text-sm ml-2">/ 페이지</span>
              </div>
            </CardHeader>

            <CardContent className="space-y-6 px-8 md:px-12">
              <ul className="space-y-4">
                {[
                  "99.99% 정확도 OCR 엔진",
                  "로컬 PC 처리로 완벽한 보안",
                  "AI 최적화 포맷 (Markdown, HTML)",
                  "무제한 용량 PDF 지원 (5000p+)",
                  "표 및 서식 구조화 인식",
                  "24시간 기술 지원"
                ].map((feature, i) => (
                  <li key={i} className="flex items-center gap-3 text-foreground/90">
                    <div className="w-5 h-5 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
                      <Check className="w-3 h-3 text-primary" />
                    </div>
                    {feature}
                  </li>
                ))}
              </ul>
            </CardContent>

            <CardFooter className="pb-12 px-8 md:px-12 pt-6">
              <Button className="w-full h-12 text-lg bg-primary text-primary-foreground hover:bg-primary/90 shadow-lg shadow-primary/20">
                지금 시작하기
              </Button>
            </CardFooter>
          </Card>

          <p className="text-center text-xs text-muted-foreground mt-6">
            * 대량 처리(월 10만 페이지 이상) 기업 고객은 별도 문의 바랍니다.
          </p>
        </div>
      </div>
    </section>
  );
}