#
# Configuration
#
base_url ?= http://ocpwork:8000/v1
token ?= "none"
model ?= granite-3-parasol-instruct

install:
	pip install -r requirements.txt

test:
	BASE_URL=$(base_url) TOKEN=$(token) MODEL=$(model) pytest
