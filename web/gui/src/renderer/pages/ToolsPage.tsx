import {
  Activity,
  Camera,
  Database,
  Gauge,
  Grid3x3,
  Heart,
  Image,
  Layers,
  ListOrdered,
  ListRestart,
  Moon,
  RefreshCw,
  Shuffle,
  Sparkles,
  Stethoscope,
  Type,
  Video,
  Wand2,
} from "lucide-react";
import type { ReactNode } from "react";
import { useState } from "react";

import { BucketAnalysisModal } from "@/components/modals/BucketAnalysisModal";
import { CaptionToolModal } from "@/components/modals/CaptionToolModal";
import { CharacterDpoModal } from "@/components/modals/CharacterDpoModal";
import { ConvertModelModal } from "@/components/modals/ConvertModelModal";
import { DistillDatasetModal } from "@/components/modals/DistillDatasetModal";
import { DPOSaturationModal } from "@/components/modals/DPOSaturationModal";
import { DPOToolModal } from "@/components/modals/DPOToolModal";
import { MaskToolModal } from "@/components/modals/MaskToolModal";
import { MergeOFTModal } from "@/components/modals/MergeOFTModal";
import { NightDarkenModal } from "@/components/modals/NightDarkenModal";
import { OffPolicyDpoModal } from "@/components/modals/OffPolicyDpoModal";
import { ProfilingPanel } from "@/components/modals/ProfilingPanel";
import { RepairRejectedModal } from "@/components/modals/RepairRejectedModal";
import { ReplacePairsModal } from "@/components/modals/ReplacePairsModal";
import { StandaloneSamplingModal } from "@/components/modals/StandaloneSamplingModal";
import { SwarmGenerateModal } from "@/components/modals/SwarmGenerateModal";
import { TextAlignmentDatasetModal } from "@/components/modals/TextAlignmentDatasetModal";
import { TriageFixModal } from "@/components/modals/TriageFixModal";
import { VideoToolModal } from "@/components/modals/VideoToolModal";
import { Button, Card } from "@/components/shared";
import { useUiStore } from "@/store/uiStore";

interface ToolCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  onLaunch?: () => void;
}

function ToolCard({ icon, title, description, onLaunch }: ToolCardProps) {
  return (
    <Card className="flex flex-col items-center text-center gap-4">
      <div className="w-12 h-12 rounded-full flex items-center justify-center bg-[var(--color-border-subtle)]">
        {icon}
      </div>
      <h3 className="text-base font-semibold text-[var(--color-on-surface)]">{title}</h3>
      <p className="text-sm text-[var(--color-on-surface-secondary)]">{description}</p>
      <Button variant="secondary" disabled={!onLaunch} onClick={onLaunch}>
        Launch
      </Button>
    </Card>
  );
}

export default function ToolsPage() {
  const [profilingOpen, setProfilingOpen] = useState(false);
  const [captionOpen, setCaptionOpen] = useState(false);
  const [maskOpen, setMaskOpen] = useState(false);
  const [videoOpen, setVideoOpen] = useState(false);
  const [convertOpen, setConvertOpen] = useState(false);
  const [samplingOpen, setSamplingOpen] = useState(false);
  const [dpoOpen, setDpoOpen] = useState(false);
  const [distillOpen, setDistillOpen] = useState(false);
  const [bucketOpen, setBucketOpen] = useState(false);
  const [repairOpen, setRepairOpen] = useState(false);
  const [mergeOftOpen, setMergeOftOpen] = useState(false);
  const [offPolicyOpen, setOffPolicyOpen] = useState(false);
  const [swarmOpen, setSwarmOpen] = useState(false);
  const [replaceOpen, setReplaceOpen] = useState(false);
  const [triageFixOpen, setTriageFixOpen] = useState(false);
  const [saturationOpen, setSaturationOpen] = useState(false);
  const [nightDarkenOpen, setNightDarkenOpen] = useState(false);
  const [textAlignOpen, setTextAlignOpen] = useState(false);
  const [characterDpoOpen, setCharacterDpoOpen] = useState(false);
  const setActiveTab = useUiStore((s) => s.setActiveTab);

  return (
    <>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <ToolCard
          icon={<Database className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Generate Captions"
          description="Batch generate captions using BLIP, WD14, OpenAI, or Gemini API"
          onLaunch={() => setCaptionOpen(true)}
        />
        <ToolCard
          icon={<Wand2 className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Generate Masks"
          description="Batch generate or interactively edit masks with YOLO support"
          onLaunch={() => setMaskOpen(true)}
        />
        <ToolCard
          icon={<Video className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Video Tools"
          description="Extract clips, images, and download videos"
          onLaunch={() => setVideoOpen(true)}
        />
        <ToolCard
          icon={<RefreshCw className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Convert Model"
          description="Convert between model formats"
          onLaunch={() => setConvertOpen(true)}
        />
        <ToolCard
          icon={<Image className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Sampling"
          description="Generate samples from a trained model"
          onLaunch={() => setSamplingOpen(true)}
        />
        <ToolCard
          icon={<Activity className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Profiling"
          description="Profile training performance"
          onLaunch={() => setProfilingOpen(true)}
        />
        <ToolCard
          icon={<ListOrdered className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Queue"
          description="Batch training with multiple configurations"
          onLaunch={() => setActiveTab("queue")}
        />
        <ToolCard
          icon={<Heart className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="DPO Pair Tool"
          description="Manage chosen/rejected preference pairs for DPO training"
          onLaunch={() => setDpoOpen(true)}
        />
        <ToolCard
          icon={<Wand2 className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Generate Off-policy Negatives"
          description="Generate images in SwarmUI from captions that need them, at each photo's exact trainer bucket size"
          onLaunch={() => setSwarmOpen(true)}
        />
        <ToolCard
          icon={<Type className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Text Alignment Dataset"
          description="Keep SwarmUI generations whose on-image text is rendered but wrong, for the offline text reward; write reward prompts as a wildcard"
          onLaunch={() => setTextAlignOpen(true)}
        />
        <ToolCard
          icon={<Camera className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Off-policy DPO Builder"
          description="Pair real photos (chosen) against the model's own generations of their captions (rejected), with a caption check"
          onLaunch={() => setOffPolicyOpen(true)}
        />
        <ToolCard
          icon={<Layers className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Character DPO"
          description="Per-character pipeline: publish the LoRA, generate negatives, pair, curate, import, finalize and stack queue entries"
          onLaunch={() => setCharacterDpoOpen(true)}
        />
        <ToolCard
          icon={<ListRestart className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Replace Pairs"
          description="Generate alternatives with SwarmUI, then blindly select the best and worst images to replace each DPO pair"
          onLaunch={() => setReplaceOpen(true)}
        />
        <ToolCard
          icon={<Stethoscope className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Triage & Fix"
          description="Tag chosen images of a DPO dataset with the fixes they need, repair them with Flux.2 Klein through SwarmUI, compare and commit"
          onLaunch={() => setTriageFixOpen(true)}
        />
        <ToolCard
          icon={<Gauge className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="DPO Saturation"
          description="Score how much preference each DPO pair has already absorbed since the lineage's pre-DPO base and retire the pairs that no longer pull"
          onLaunch={() => setSaturationOpen(true)}
        />
        <ToolCard
          icon={<Moon className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Night Darken"
          description="Find chosen images whose caption says night or dark but that are not, review each with a before/after wipe (darken, noise, hue, saturation), replace them in place (backed up) and add the most changed as pairs in a separate Night Edits dataset"
          onLaunch={() => setNightDarkenOpen(true)}
        />
        <ToolCard
          icon={<Sparkles className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Distillation Dataset"
          description="Copy good images into a normalized metadata dataset for DMD2 distillation"
          onLaunch={() => setDistillOpen(true)}
        />
        <ToolCard
          icon={<Grid3x3 className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="DPO Bucket Analysis"
          description="Analyze aspect-bucket counts and clean-up targets for your batch size"
          onLaunch={() => setBucketOpen(true)}
        />
        <ToolCard
          icon={<Shuffle className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Re-pair by Similarity"
          description="Rename rejected images so each pairs with the most visually similar chosen image in its caption group"
          onLaunch={() => setRepairOpen(true)}
        />
        <ToolCard
          icon={<Layers className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Merge OFT"
          description="Bake an OFT / DoRA-OFT adapter into a base checkpoint using OneTrainer's training math"
          onLaunch={() => setMergeOftOpen(true)}
        />
      </div>

      <ProfilingPanel open={profilingOpen} onClose={() => setProfilingOpen(false)} />
      <CaptionToolModal open={captionOpen} onClose={() => setCaptionOpen(false)} />
      <MaskToolModal open={maskOpen} onClose={() => setMaskOpen(false)} />
      <VideoToolModal open={videoOpen} onClose={() => setVideoOpen(false)} />
      <ConvertModelModal open={convertOpen} onClose={() => setConvertOpen(false)} />
      <StandaloneSamplingModal open={samplingOpen} onClose={() => setSamplingOpen(false)} />
      <DPOToolModal open={dpoOpen} onClose={() => setDpoOpen(false)} />
      <DistillDatasetModal open={distillOpen} onClose={() => setDistillOpen(false)} />
      <BucketAnalysisModal open={bucketOpen} onClose={() => setBucketOpen(false)} />
      <RepairRejectedModal open={repairOpen} onClose={() => setRepairOpen(false)} />
      <MergeOFTModal open={mergeOftOpen} onClose={() => setMergeOftOpen(false)} />
      <OffPolicyDpoModal open={offPolicyOpen} onClose={() => setOffPolicyOpen(false)} />
      <SwarmGenerateModal open={swarmOpen} onClose={() => setSwarmOpen(false)} />
      <ReplacePairsModal open={replaceOpen} onClose={() => setReplaceOpen(false)} />
      <TriageFixModal open={triageFixOpen} onClose={() => setTriageFixOpen(false)} />
      <DPOSaturationModal open={saturationOpen} onClose={() => setSaturationOpen(false)} />
      <NightDarkenModal open={nightDarkenOpen} onClose={() => setNightDarkenOpen(false)} />
      <TextAlignmentDatasetModal open={textAlignOpen} onClose={() => setTextAlignOpen(false)} />
      <CharacterDpoModal open={characterDpoOpen} onClose={() => setCharacterDpoOpen(false)} />
    </>
  );
}
