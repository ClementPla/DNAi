from PIL import Image
from dnafiber.ui.consts import DefaultValues as DV
from dnafiber.ui.utils import init_session_states

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError


init_session_states()
