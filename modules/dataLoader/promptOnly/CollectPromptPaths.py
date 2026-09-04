import os

from mgds.pipelineModules.CollectPaths import CollectPaths


class CollectPromptPaths(CollectPaths):
    """Enumerates `.txt` caption files instead of images.

    Prompt-only distillation reuses a concept directory's captions without
    touching its images (which may or may not exist). Each row's `image_path`
    is emitted as the caption's synthetic `<stem>.png` sibling — never opened,
    but it keeps every downstream path derivation working unchanged:
    `ModifyPath(extension=".txt")` reconstructs `sample_prompt_path`, and the
    distill sidecar lookup resolves `<stem>.json` via `.with_suffix(".json")`.
    """

    def __init__(
        self,
        concept_in_name: str,
        path_in_name: str,
        include_subdirectories_in_name: str,
        enabled_in_name: str,
        path_out_name: str,
        concept_out_name: str,
        exclude_postfix: list[str],
    ):
        super().__init__(
            concept_in_name=concept_in_name,
            path_in_name=path_in_name,
            include_subdirectories_in_name=include_subdirectories_in_name,
            enabled_in_name=enabled_in_name,
            path_out_name=path_out_name,
            concept_out_name=concept_out_name,
            extensions=[".txt"],
            include_postfix=None,
            exclude_postfix=exclude_postfix,
        )

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        item = super().get_item(variation, index, requested_name)
        txt_path = item[self.path_out_name]
        item[self.path_out_name] = os.path.splitext(txt_path)[0] + ".png"
        return item
