# four algo: 900 (0-899)
echo "starting at `date`"
for n in {0..10};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "nascar_mpc_bicycle 100 done at `date`"

