import os

# Parameters for sampling and windowing.
FS = 2000           # Hz
WINDOW_MS = 200     # Milliseconds
STEP_MS = 100       # Milliseconds
N_CHANNELS = 12
N_CLASSES = 18

# Filter parameters
LOWCUT = 20         # Hz
HIGHCUT = 500       # Hz
FILTER_ORDER = 4

# Update this to your USB mount path
USB_PATH = '/Volumes/KINGSTON'
DATA_PATH = os.path.join(USB_PATH, 'ninapro_db2')
N_SUBJECTS = 40

def get_subject_path(subject_id, exercise=1):
    """
    Get path to a Ninapro DB2 subject file.
    
    Args:
        subject_id: integer 1-40
        exercise: 1, 2, or 3 (default 1)
    
    Returns:
        Full path to .mat file
    """
    folder = f'DB2_s{subject_id}'
    filename = f'S{subject_id}_E{exercise}_A1.mat'
    return os.path.join(DATA_PATH, folder, filename)