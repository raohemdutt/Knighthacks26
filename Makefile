# Switches for a running box. They go through the gateway, so the phone and the event log stay in step.
#   make offline   nothing leaves the LAN (the firewall follows within seconds); detection, answers and alerts carry on
#   make online    back to the allowlist
ATLAS_URL ?= http://127.0.0.1:8000
AUTH = $(if $(ATLAS_TOKEN),-H "Authorization: Bearer $(ATLAS_TOKEN)",)

offline:
	curl -fsS -X PUT $(ATLAS_URL)/vault/policy -H "Content-Type: application/json" $(AUTH) -d '{"offline": true}'

online:
	curl -fsS -X PUT $(ATLAS_URL)/vault/policy -H "Content-Type: application/json" $(AUTH) -d '{"offline": false}'

.PHONY: offline online
