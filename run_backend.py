from src.services.external_server import ExternalServer

es = ExternalServer(log_filepath="run_backend.log")
with open(".webui_port", "w") as f:
    f.write(str(es.port))

es.launch_server()
