# Unified Image Protection & Editing Pipeline

This project provides a unified interface to combine Image Protection methods (adversarial attacks/noise addition) with Image Editing techniques. It allows users to protect images against unauthorized manipulation and then evaluate how well these protections hold up against various editing methods.

## Supported Methods & Original Repositories

This project integrates the following research codes as submodules:

### Protection Methods (modules/)
*   **AtkPDM**: [Pixel Is Not A Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models](https://github.com/Harry1060408/AtkPDM) (AAAI 2025)
    *   **Environment**: Requires Python 3.10.
    *   `conda create -n atkpdm python=3.10`
    *   `pip install -r modules/AtkPDM/requirements.txt`
*   **Diff-Protect**: [Toward effective protection against diffusion-based mimicry through score distillation](https://github.com/psyker-team/mist) (ICLR 2024)
    *   **Environment**: Requires `env.yml` from original repo.
    *   `conda env create -f modules/Diff-Protect/env.yml`
*   **PID**: [PID: Prompt-Independent Data Protection Against Latent Diffusion Models](https://github.com/alchemi5t/Diffusion-PID-Protection) (ICML 2024)
    *   **Environment**: Requires Python 3.8+.
    *   `conda create -n pid_env python=3.8`
    *   `pip install -r modules/Diffusion-PID-Protection/requirements.txt`

### Editing Methods (modules/)
*   **FlowEdit**: [FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](https://github.com/fallenshock/FlowEdit) (ICCV 2025 Best Student Paper)
    *   **Environment**: Compatible with modern diffusers (Python 3.10+ recommended).
    *   `conda create -n flowedit_env python=3.10`
    *   `pip install -r modules/FlowEdit/requirements.txt` (if exists, otherwise standard diffusers)
    *   **Input**: Single image path.
    *   **Prompt**: Requires `source_prompt` (original image description) and `target_prompt` (desired edit).
*   **Dreambooth**: Fine-tuning based editing (Integrated via PID's codebase / HuggingFace).
    *   **Input**: Directory of instance images (though wrapper handles single image).
    *   **Prompt**: Requires `instance_prompt` (e.g. "a photo of sks dog") and `target_prompt`.

## Input Requirements Summary

Each module has specific requirements for inputs and prompts. The unified pipeline handles most of this, but understanding the underlying requirements is helpful:

| Module | Input Format | Prompts Required | Notes |
| :--- | :--- | :--- | :--- |
| **AtkPDM** | Directory of images | None (uses internal logic or default) | Expects images in subfolders (e.g., `data/cat/1.png`). Wrapper handles this structure. |
| **Diff-Protect** | Directory of images | None (uses default "a photo") | Expects directory input. Wrapper creates temp directory. |
| **PID** | Directory of images | None | Training based. Wrapper handles temp directory creation. |
| **FlowEdit** | Single Image Path | `source_prompt`, `target_prompt` | Requires accurate source prompt for best inversion. |
| **Dreambooth** | Single Image (Wrapper) | `source_prompt` (as instance), `target_prompt` | Fine-tunes model on input, then generates target. |

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2.  **Initialize Submodules**:
    This project relies on external repositories as submodules. You can use the provided script or set them up manually.

    **Option A: Automatic Setup**
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

    **Option B: Manual Setup**
    If you prefer to set up the environment manually, run the following commands to create the `modules` directory and clone the required repositories:

    ```bash
    # Create modules directory
    mkdir -p modules
    cd modules

    # Clone AtkPDM
    git clone https://github.com/Harry1060408/AtkPDM

    # Clone Diff-Protect (Repository name is 'mist', renamed to 'Diff-Protect')
    git clone https://github.com/psyker-team/mist Diff-Protect

    # Clone PID
    git clone https://github.com/alchemi5t/Diffusion-PID-Protection

    # Clone FlowEdit
    git clone https://github.com/fallenshock/FlowEdit

    # Return to root
    cd ..
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    You may also need to install specific dependencies for each submodule found in `modules/*/requirements.txt`.

## Usage

The main entry point is `main.py`. You can specify the protection method, the editing method, and the models to use.

### Arguments

*   `--input_image`: Path to the input image file.
*   `--output_dir`: Directory where results will be saved.
*   `--protection_method`: One of `atk_pdm`, `diff_protect`, `pid`.
*   `--protection_model`: Model to use for protection (`sd1.4`, `sd3`, `flux`). *Note: Support depends on the specific method.*
*   `--editing_method`: One of `flow_edit`, `dreambooth`.
*   `--edit_model`: Model to use for editing (`sd3`, `flux`, `sd1.4`).
*   `--source_prompt`: Description of the original image.
*   `--target_prompt`: Description of the desired edit.

### Batch Evaluation (FlowEdit Style)

To run FlowEdit on a dataset (e.g., the original `edits.yaml`) and evaluate the results without applying any protection:

```bash
python main.py \
  --dataset_yaml modules/FlowEdit/edits.yaml \
  --editing_method flow_edit \
  --edit_model sd3 \
  --output_dir results/flowedit_eval_only
```

*Note: Omitting `--protection_method` will skip the protection step, copying the original image as the "protected" input for the editing stage. The pipeline will then evaluate the quality of the edit (CLIP score, etc.) against the original.*

### Example

Protect an image of a cat using **AtkPDM** (SD1.4) and attempt to edit it into a dog using **FlowEdit** (SD3):

```bash
# Edit config.json first
./run_experiment.sh
```

Or manually:
```bash
python main.py \
  --input_image data/cat.png \
  --output_dir results/experiment_1 \
  --protection_method atk_pdm \
  --protection_model sd1.4 \
  --editing_method flow_edit \
  --edit_model sd3 \
  --source_prompt "a photo of a cat" \
  --target_prompt "a photo of a dog"
```

## Project Structure

*   `main.py`: CLI entry point.
*   `src/`: Core logic and wrappers.
    *   `protection/`: Wrappers for AtkPDM, Diff-Protect, PID.
    *   `editing/`: Wrappers for FlowEdit, Dreambooth.
    *   `evaluation/`: Metrics (LPIPS, PSNR, etc.).
    *   `pipeline.py`: Orchestrates the protection -> editing -> evaluation flow.
*   `modules/`: Contains the external git submodules.

## Scripts Overview

*   **`run_experiment.sh`**: The **primary entry point** for running full experiments.
    *   It reads `config.json`, detects the required protection/editing method, activates the correct Conda environment (`pid_env`, `atkpdm`, etc.), and runs `main.py`.
    *   Use this for reproducing experiments or running batches.

*   **`main.py`**: The core logic script.
    *   Can be run directly if you are already in the correct environment.
    *   Supports single-image mode and batch mode (via `--dataset_yaml`).

## Evaluation

The pipeline automatically calculates metrics after each stage:
*   **Protection Quality**: LPIPS, MSE, PSNR (between Original and Protected).
*   **Editing Quality**: Structure preservation metrics (between Protected and Edited).

Results are saved to `results.json` in the output directory.

## References

If you use this code, please cite the original papers:

**AtkPDM**
```bibtex
@inproceedings{shih2024atkpdm,
  title     = {Pixel Is Not A Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models},
  author    = {Chun-Yen Shih and Li-Xuan Peng and Jia-Wei Liao and Ernie Chu and Cheng-Fu Chou and Jun-Cheng Chen},
  booktitle = {Annual AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2025},
}
```

**Diff-Protect (Mist)**
```bibtex
@inproceedings{xue2023toward,
  title={Toward effective protection against diffusion-based mimicry through score distillation},
  author={Xue, Haotian and Liang, Chumeng and Wu, Xiaoyu and Chen, Yongxin},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```

**PID**
```bibtex
@inproceedings{li2024pid,
  title={PID: Prompt-Independent Data Protection Against Latent Diffusion Models},
  author={Li, Ang and Mo, Yichuan and Li, Mingjie and Wang, Yisen},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

**FlowEdit**
```bibtex
@inproceedings{kulikov2025flowedit,
  title={Flowedit: Inversion-free text-based editing using pre-trained flow models},
  author={Kulikov, Vladimir and Kleiner, Matan and Huberman-Spiegelglas, Inbar and Michaeli, Tomer},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19721--19730},
  year={2025}
}
```
