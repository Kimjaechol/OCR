import { Button } from "@/components/ui/button";
import { Scale, FileText, Shield } from "lucide-react";
import { SettingsDialog } from "@/components/settings-dialog";

export function Nav() {
  return (
    <nav className="fixed top-0 w-full z-50 border-b border-white/10 bg-background/80 backdrop-blur-md">
      <div className="container mx-auto px-6 h-20 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Scale className="w-6 h-6 text-primary" />
          <span className="font-serif text-xl font-bold tracking-tight text-foreground">LexScan Pro</span>
        </div>
        
        <div className="hidden md:flex items-center gap-8">
          <a href="#features" className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors">기능 소개</a>
          <a href="#security" className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors">보안 및 프라이버시</a>
          <a href="#pricing" className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors">요금 안내</a>
        </div>

        <div className="flex items-center gap-4">
          <SettingsDialog />
          <Button variant="ghost" className="text-muted-foreground hover:text-foreground">로그인</Button>
          <Button className="bg-primary text-primary-foreground hover:bg-primary/90 font-medium">
            무료 체험하기
          </Button>
        </div>
      </div>
    </nav>
  );
}