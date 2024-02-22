for n in {0..60};
do
  #time python batchRunSingle.py exploit_sine $n > "../log/concurrent/log$n.txt" 2>&1 &
  time python batchRunSingle.py sine_mpc $n > "../log/concurrent/log$n.txt" 2>&1 &
done

wait
echo "done with 60"

for n in {61..120};
do
  #time python batchRunSingle.py exploit_sine $n > "../log/concurrent/log$n.txt" 2>&1 &
  time python batchRunSingle.py sine_mpc $n > "../log/concurrent/log$n.txt" 2>&1 &
done

wait

echo "done with 120"

for n in {121..179};
do
  #time python batchRunSingle.py exploit_sine $n > "../log/concurrent/log$n.txt" 2>&1 &
  time python batchRunSingle.py sine_mpc $n > "../log/concurrent/log$n.txt" 2>&1 &
done

wait
