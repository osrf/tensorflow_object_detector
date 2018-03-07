#!/bin/bash
set -o verbose
stress -m 1 --vm-hang 0 --vm-bytes 4096M
