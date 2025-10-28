import time
from src.services.external_client import ExternalClient

with open(".webui_port", "r", encoding="utf-8") as f:
    port = int(f.read())
ec = ExternalClient(port=port)
# ec.start_listening()
time.sleep(1)
ec.get_choice_response(
    question="Which framework do you prefer for frontend development?",
    options=["React", "Vue", "Svelte", "Angular"],
)
