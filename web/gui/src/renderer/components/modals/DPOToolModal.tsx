import { useCallback } from "react";

import { ExportStep } from "./dpo/ExportStep";
import { RankingStep } from "./dpo/RankingStep";
import { ReviewStep } from "./dpo/ReviewStep";
import { ScanningStep } from "./dpo/ScanningStep";
import { SelectionStep } from "./dpo/SelectionStep";
import { SetupStep } from "./dpo/SetupStep";
import { TournamentStep } from "./dpo/TournamentStep";
import { TriageStep } from "./dpo/TriageStep";
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
  swiss: "DPO Pair Tool — Tournament",
  ranking: "DPO Pair Tool — Ranked Review",
  triage: "DPO Pair Tool — Triage",
  review: "DPO Pair Tool — Review Pairs",
  export: "DPO Pair Tool — Finalize",
};

const sizeByStep: Record<CurationStep, "lg" | "full"> = {
  setup: "lg",
  scanning: "lg",
  selecting: "full",
  swiss: "full",
  ranking: "full",
  triage: "full",
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
    scanTotal,
    hashCount,
    cacheHits,
    group,
    phase,
    bestImage,
    remainingImages,
    pairsDone,
    swissMatch,
    swissRound,
    swissTotalRounds,
    swissMatchesPlayed,
    swissMatchesTotal,
    rankedOrder,
    rankingScores,
    maxPairs,
    pairCount,
    showAcceptDialog,
    pendingBest,
    pendingWorst,
    totalPairs,
    finalResult,
    resume,
    error,
    actions,
  } = session;

  const sessionActive =
    step === "scanning" || step === "selecting" || step === "swiss" || step === "ranking" || step === "triage";

  const handleClose = useCallback(() => {
    if (sessionActive) {
      void actions.cancelSession().finally(onClose);
    } else {
      onClose();
    }
  }, [sessionActive, actions, onClose]);

  return (
    <ModalBase
      open={open}
      onClose={handleClose}
      title={titleByStep[step]}
      size={sizeByStep[step]}
      // A stray click outside the dialog (or an accidental Escape) must not
      // kill an in-flight curation session — that throws away the whole scan.
      // Closing mid-session stays possible via the X button or Cancel.
      closeOnBackdrop={!sessionActive}
      closeOnEscape={!sessionActive}
    >
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

      {step === "scanning" && (
        <ScanningStep
          scanCount={scanCount}
          scanTotal={scanTotal}
          hashCount={hashCount}
          cacheHits={cacheHits}
          onCancel={() => void actions.cancelSession()}
        />
      )}

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

      {step === "swiss" && group && (
        <TournamentStep
          group={group}
          match={swissMatch}
          round={swissRound}
          totalRounds={swissTotalRounds}
          matchesPlayed={swissMatchesPlayed}
          matchesTotal={swissMatchesTotal}
          onVote={actions.swissVote}
          onFinishEarly={actions.swissFinishEarly}
          onSkipGroup={() => void actions.skipGroup()}
          onCancel={() => void actions.cancelSession()}
          pairsDone={pairsDone}
        />
      )}

      {step === "ranking" && group && (
        <RankingStep
          group={group}
          order={rankedOrder}
          scores={rankingScores}
          maxPairs={maxPairs}
          pairCount={pairCount}
          onPairCountChange={actions.setPairCount}
          onSwap={actions.swapRanked}
          onExport={actions.swissExportPairs}
          onSkipGroup={() => void actions.skipGroup()}
          onCancel={() => void actions.cancelSession()}
          pairsDone={pairsDone}
        />
      )}

      {step === "triage" && group && (
        <TriageStep
          group={group}
          pairsDone={pairsDone}
          onCommitPairs={actions.commitTriagePairs}
          onAutoAlign={actions.triageAlign}
          onSkipGroup={() => void actions.skipGroup()}
          onDiscardGroup={() => void actions.discardGroup()}
          onCancel={() => void actions.cancelSession()}
        />
      )}

      {step === "review" && <ReviewStep onBack={() => setStep("setup")} onClose={onClose} />}

      {step === "export" && (
        <ExportStep totalPairs={totalPairs} onFinalize={actions.finalize} onClose={onClose} finalResult={finalResult} />
      )}
    </ModalBase>
  );
}
