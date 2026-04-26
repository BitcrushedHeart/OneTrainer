import { useCallback } from "react";

import { EloStep } from "./dpo/EloStep";
import { ExportStep } from "./dpo/ExportStep";
import { ReviewStep } from "./dpo/ReviewStep";
import { ScanningStep } from "./dpo/ScanningStep";
import { SelectionStep } from "./dpo/SelectionStep";
import { SetupStep } from "./dpo/SetupStep";
import type { CurationStep } from "./dpo/types";
import { useDpoSession } from "./dpo/useDpoSession";
import { ModalBase } from "./ModalBase";

interface Props {
  open: boolean;
  onClose: () => void;
}

const titleByStep: Record<CurationStep, string> = {
  setup: "DPO Pair Tool",
  scanning: "DPO Pair Tool — Scanning",
  selecting: "DPO Pair Tool — Selection",
  elo: "DPO Pair Tool — ELO Ranking",
  review: "DPO Pair Tool — Review Pairs",
  export: "DPO Pair Tool — Finalize",
};

const sizeByStep: Record<CurationStep, "lg" | "full"> = {
  setup: "lg",
  scanning: "lg",
  selecting: "full",
  elo: "full",
  review: "full",
  export: "lg",
};

export function DPOToolModal({ open, onClose }: Props) {
  const session = useDpoSession();
  const {
    step,
    setStep,
    sourceFolder,
    outputDir,
    pairsPerGroup,
    scanCount,
    group,
    phase,
    bestImage,
    remainingImages,
    pairsDone,
    eloPair,
    eloRatings,
    eloDone,
    eloSuggested,
    eloFinished,
    showAcceptDialog,
    pendingBest,
    pendingWorst,
    totalPairs,
    finalResult,
    resume,
    error,
    actions,
  } = session;

  const handleClose = useCallback(() => {
    if (step === "scanning" || step === "selecting" || step === "elo") {
      void actions.cancelSession().finally(onClose);
    } else {
      onClose();
    }
  }, [step, actions, onClose]);

  const suggestedForGroup = Math.max(
    15,
    Math.ceil((group?.images.length ?? 0) * Math.log2(Math.max(group?.images.length ?? 2, 2))),
  );

  return (
    <ModalBase open={open} onClose={handleClose} title={titleByStep[step]} size={sizeByStep[step]}>
      {step === "setup" && (
        <SetupStep
          sourceFolder={sourceFolder}
          outputDir={outputDir}
          pairsPerGroup={pairsPerGroup}
          onSourceChange={actions.setSourceFolder}
          onOutputChange={actions.setOutputDir}
          onPairsPerGroupChange={actions.setPairsPerGroup}
          onStart={actions.startSession}
          onReview={actions.goToReview}
          onClose={onClose}
          resume={resume}
          onSelectOutput={actions.loadOutputManifest}
          error={error}
        />
      )}

      {step === "scanning" && <ScanningStep scanCount={scanCount} onCancel={() => void actions.cancelSession()} />}

      {step === "selecting" && group && (
        <SelectionStep
          group={group}
          onPick={actions.pickImage}
          onAcceptPair={(keepScoring) => void actions.confirmPair(keepScoring)}
          onDismissPair={() => void actions.cancelPendingPair()}
          onSkipGroup={() => void actions.skipGroup()}
          onCancel={() => void actions.cancelSession()}
          phase={phase}
          bestImage={bestImage}
          remainingImages={remainingImages}
          pairsDone={pairsDone}
          showAcceptDialog={showAcceptDialog}
          pendingBest={pendingBest}
          pendingWorst={pendingWorst}
        />
      )}

      {step === "elo" && group && (
        <EloStep
          group={group}
          pair={eloPair}
          ratings={eloRatings}
          done={eloDone}
          suggested={eloSuggested || suggestedForGroup}
          finished={eloFinished}
          onVote={actions.eloVote}
          onAccept={actions.eloAccept}
          onSkipGroup={() => void actions.skipGroup()}
          onCancel={() => void actions.cancelSession()}
          pairsDone={pairsDone}
        />
      )}

      {step === "review" && <ReviewStep onBack={() => setStep("setup")} onClose={onClose} />}

      {step === "export" && (
        <ExportStep totalPairs={totalPairs} onFinalize={actions.finalize} onClose={onClose} finalResult={finalResult} />
      )}
    </ModalBase>
  );
}
