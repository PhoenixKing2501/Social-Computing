MAKE = make

.PHONY: all up attach start stop down

all:
	$(MAKE) start
	$(MAKE) attach
	$(MAKE) stop

up:
	docker compose up -d --build
	$(MAKE) attach
	$(MAKE) stop

attach:
	docker compose attach snap-container

start:
	docker compose start

stop:
	docker compose stop

down:
	docker compose down
