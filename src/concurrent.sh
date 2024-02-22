for n in {0..119};
do
  time python batchRunSingle.py exploit_sine $n > "../log/concurrent/log$n.txt" 2>&1 &
done

wait
