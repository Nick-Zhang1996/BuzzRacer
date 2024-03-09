# four algo: 900 (0-899)
echo "starting at `date`"
echo "starting at `date`">>concurrent.log
for n in {0..99};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 100 done at `date`"
echo "four_algo 100 done at `date`">>concurrent.log

