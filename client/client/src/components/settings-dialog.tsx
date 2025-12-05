import { useState, useEffect } from "react";
import { Settings, Save, Server, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";

export function SettingsDialog() {
  const [isOpen, setIsOpen] = useState(false);
  const [serverUrl, setServerUrl] = useState("http://localhost:8000");
  const [apiKey, setApiKey] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const { toast } = useToast();

  // Simulate checking connection
  const checkConnection = () => {
    setIsChecking(true);
    setTimeout(() => {
      setIsChecking(false);
      setIsConnected(true);
      toast({
        title: "연결 성공",
        description: "로컬 OCR 엔진과 성공적으로 연결되었습니다.",
      });
    }, 1500);
  };

  const handleSave = () => {
    localStorage.setItem("ocr_server_url", serverUrl);
    setIsOpen(false);
    toast({
      title: "설정 저장됨",
      description: "서버 설정이 업데이트되었습니다.",
    });
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
          <Settings className="w-5 h-5" />
        </Button>
      </DialogTrigger>
      <DialogContent className="bg-card border-primary/20 text-foreground sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-xl font-serif">
            <Server className="w-5 h-5 text-primary" />
            OCR 엔진 연결 설정
          </DialogTitle>
          <DialogDescription>
            로컬에서 실행 중인 Python OCR 서버와 연결합니다.
          </DialogDescription>
        </DialogHeader>
        
        <div className="grid gap-6 py-4">
          <div className="space-y-2">
            <Label htmlFor="url" className="text-right">서버 주소 (Localhost)</Label>
            <div className="flex gap-2">
              <Input
                id="url"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                className="bg-background/50 border-white/10 font-mono text-sm"
              />
              <Button 
                variant="outline" 
                size="icon" 
                onClick={checkConnection}
                className="shrink-0 border-primary/30 hover:bg-primary/10"
                disabled={isChecking}
              >
                {isChecking ? (
                  <RefreshCw className="w-4 h-4 animate-spin text-primary" />
                ) : (
                  <RefreshCw className={`w-4 h-4 ${isConnected ? 'text-green-500' : 'text-muted-foreground'}`} />
                )}
              </Button>
            </div>
            <div className="flex items-center gap-2 text-xs">
              상태: 
              {isConnected ? (
                <Badge variant="outline" className="border-green-500/50 text-green-500 bg-green-500/10">연결됨 (v1.2.0)</Badge>
              ) : (
                <Badge variant="outline" className="border-yellow-500/50 text-yellow-500 bg-yellow-500/10">대기 중</Badge>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="api-key">API Key (Optional)</Label>
            <Input
              id="api-key"
              type="password"
              placeholder="sk-..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="bg-background/50 border-white/10 font-mono text-sm"
            />
            <p className="text-[10px] text-muted-foreground">
              * 로컬 전용 모드에서는 API 키가 필요하지 않을 수 있습니다.
            </p>
          </div>
        </div>

        <DialogFooter>
          <Button onClick={handleSave} className="w-full bg-primary text-primary-foreground hover:bg-primary/90">
            <Save className="w-4 h-4 mr-2" /> 설정 저장하기
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}