import torch
import logging
import platform


def verify_device() -> str:
    if torch.backends.mps.is_available():
        logging.info("Running on MPS")
        return "mps"
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        logging.info(f"Running CUDA on {num_devices} devices")
        return "cuda"

    if platform.system() == "Darwin":
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.info(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                logging.info(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )

            return "cpu"

    logging.info("Running on CPU")
    return "cpu"
