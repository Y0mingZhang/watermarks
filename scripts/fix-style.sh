#!/bin/bash

black $(dirname $0)/../src
isort $(dirname $0)/../src --ds --profile black
ruff $(dirname $0)/../src