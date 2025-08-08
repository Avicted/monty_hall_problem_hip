#!/bin/bash

# set -xe

python -m venv monty_hall_venv

source monty_hall_venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -r requirements.txt



source monty_hall_venv/bin/activate

python plot.py
