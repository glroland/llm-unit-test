#
# Configuration
#
base_url ?= http://ocpwork:8000/v1
token ?= "none"
model ?= granite-3-parasol-instruct
temp ?= 0.3
#temp ?= 0.0
top_p ?= 1
pp ?= 0
fp ?= 0

install:
	pip install -r requirements.txt

test:
	TEMP=$(temp) TOP_P=$(top_p) PP=$(pp) FP=$(fp) OPENAI_API_KEY=$(token) BASE_URL=$(base_url) TOKEN=$(token) MODEL=$(model) pytest

lint:
	pylint src
