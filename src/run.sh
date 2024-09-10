#!/bin/bash

module sw lang/tcsds-1.2.38

./compile.sh

cd BIN_deepmd_water

pjsub run_load_balance
pjsub run_strong_scaling_8x12x8
pjsub run_strong_scaling_12x15x12
pjsub run_strong_scaling_16x18x16
pjsub run_strong_scaling_16x24x16
pjsub run_strong_scaling_20x30x20

cd ../BIN_deepmd_copper

pjsub run_load_balance
pjsub run_strong_scaling_8x12x8
pjsub run_strong_scaling_12x15x12
pjsub run_strong_scaling_16x18x16
pjsub run_strong_scaling_16x24x16
pjsub run_strong_scaling_20x30x20