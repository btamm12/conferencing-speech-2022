#!/usr/bin/env bash
echo "Running job_exectuable_sleep.sh"
echo "Running from: $(pwd)"
echo "$(condor_config_val EXECUTE)"
echo "$_CONDOR_SCRATCH_DIR"
sleep 19200
