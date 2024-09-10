module sw lang/tcsds-1.2.38

(make -j48 threadpool 2>&1) | tee compile.log

mv lmp_threadpool lmp_execute
cp lmp_execute BIN_deepmd_water
cp lmp_execute BIN_deepmd_copper
