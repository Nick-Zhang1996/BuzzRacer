echo "starting at `date`"
for n in {0..99};
do
  python batchRunSingle.py exploit_vs_exploit $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_vs_exploit 100 done at `date`"

echo "starting at `date`"
for n in {0..99};
do
  python batchRunSingle.py exploit_vs_aggressive $n > "../log/concurrent/log1$n.txt" 2>&1 &
done
wait
echo "exploit_vs_aggressive 100 done at `date`"
