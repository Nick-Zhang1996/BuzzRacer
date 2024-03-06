echo "starting at `date`"
# sine_mpc_dense, 300 total
for n in {0..99};
do
  time python batchRunSingle.py sine_mpc_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "sine_mpc_dense 100 done at `date`"

for n in {100..199};
do
  time python batchRunSingle.py sine_mpc_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "sine_mpc_dense 200 done at `date`"

for n in {200..299};
do
  time python batchRunSingle.py sine_mpc_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "sine_mpc_dense 300 done at `date`"


# triangle_mpc_dense, 300 total
for n in {0..99};
do
  time python batchRunSingle.py triangle_mpc_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "triangle_mpc_dense 100 done at `date`"

for n in {100..199};
do
  time python batchRunSingle.py triangle_mpc_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "triangle_mpc_dense 200 done at `date`"

for n in {200..299};
do
  time python batchRunSingle.py triangle_mpc_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "triangle_mpc_dense 300 done at `date`"


# nascar_mpc_dense, 300 total
for n in {0..99};
do
  time python batchRunSingle.py nascar_mpc_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "nascar_mpc_dense 100 done at `date`"

for n in {100..199};
do
  time python batchRunSingle.py nascar_mpc_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "nascar_mpc_dense 200 done at `date`"

for n in {200..299};
do
  time python batchRunSingle.py nascar_mpc_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "nascar_mpc_dense 300 done at `date`"


# exploit_sine_dense, 400 total
for n in {0..99};
do
  time python batchRunSingle.py exploit_sine_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_sine_dense 100 done at `date`"

for n in {100..199};
do
  time python batchRunSingle.py exploit_sine_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_sine_dense 200 done at `date`"

for n in {200..299};
do
  time python batchRunSingle.py exploit_sine_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_sine_dense 300 done at `date`"

for n in {300..399};
do
  time python batchRunSingle.py exploit_sine_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_sine_dense 400 done at `date`"


# exploit_triangle_dense, 400 total
for n in {0..99};
do
  time python batchRunSingle.py exploit_triangle_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_triangle_dense 100 done at `date`"

for n in {100..199};
do
  time python batchRunSingle.py exploit_triangle_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_triangle_dense 200 done at `date`"

for n in {200..299};
do
  time python batchRunSingle.py exploit_triangle_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_triangle_dense 300 done at `date`"

for n in {300..399};
do
  time python batchRunSingle.py exploit_triangle_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_triangle_dense 400 done at `date`"


# exploit_nascar_dense, 400 total
for n in {0..99};
do
  time python batchRunSingle.py exploit_nascar_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_nascar_dense 100 done at `date`"

for n in {100..199};
do
  time python batchRunSingle.py exploit_nascar_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_nascar_dense 200 done at `date`"

for n in {200..299};
do
  time python batchRunSingle.py exploit_nascar_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_nascar_dense 300 done at `date`"

for n in {300..399};
do
  time python batchRunSingle.py exploit_nascar_dense $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "exploit_nascar_dense 400 done at `date`"
