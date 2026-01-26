import logging

# Global Settings
NA_VALUE = 128
PARALLEL_JOBS = -1  # Use all available cores

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OpenLDM")
