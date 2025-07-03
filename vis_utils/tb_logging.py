from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

MELTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(os.path.dirname(MELTS_ROOT), "MELTS_runs")
os.makedirs(LOG_ROOT, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(LOG_ROOT, f"run_{timestamp}")
os.makedirs(log_dir, exist_ok=True)

print(f"[TensorBoard] Logging to: {log_dir}")

TENSORBOARD_LOGGER: SummaryWriter = SummaryWriter(log_dir=log_dir)
TRAINING_LOG_STEP = 0
AUGMENTATION_LOG_STEP = 0
TI_LOG_STEP = 0
DEBUG_LOG_STEP = 0

LOG_INTERVAL = 10