echo "starting at `date`"
# nascar_mpc_bicycle, 300 total
for n in {0..99};
do
  python batchRunSingle.py nascar_mpc_bicycle $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "nascar_mpc_bicycle 100 done at `date`"

for n in {100..199};
do
  python batchRunSingle.py nascar_mpc_bicycle $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "nascar_mpc_bicycle 200 done at `date`"

for n in {200..299};
do
  python batchRunSingle.py nascar_mpc_bicycle $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "nascar_mpc_bicycle 300 done at `date`"
