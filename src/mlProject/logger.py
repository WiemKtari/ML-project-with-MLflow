import logging
import os

# Crée un dossier pour les logs
os.makedirs("logs", exist_ok=True)

LOG_FILE = os.path.join("logs", "mlProject.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s] %(message)s",
    filemode="a"
)

logger = logging.getLogger(__name__)

# Optionnel : afficher aussi dans la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s: %(levelname)s: %(module)s] %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
