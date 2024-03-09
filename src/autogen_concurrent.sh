
# four algo: 900 (0-899)
echo "starting at `date`"
echo "starting at `date`">>concurrent.log

for n in {0..39};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 39 done at `date`"
echo "four_algo 39 done at `date`">>concurrent.log

for n in {40..79};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 79 done at `date`"
echo "four_algo 79 done at `date`">>concurrent.log

for n in {80..119};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 119 done at `date`"
echo "four_algo 119 done at `date`">>concurrent.log

for n in {120..159};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 159 done at `date`"
echo "four_algo 159 done at `date`">>concurrent.log

for n in {160..199};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 199 done at `date`"
echo "four_algo 199 done at `date`">>concurrent.log

for n in {200..239};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 239 done at `date`"
echo "four_algo 239 done at `date`">>concurrent.log

for n in {240..279};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 279 done at `date`"
echo "four_algo 279 done at `date`">>concurrent.log

for n in {280..319};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 319 done at `date`"
echo "four_algo 319 done at `date`">>concurrent.log

for n in {320..359};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 359 done at `date`"
echo "four_algo 359 done at `date`">>concurrent.log

for n in {360..399};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 399 done at `date`"
echo "four_algo 399 done at `date`">>concurrent.log

for n in {400..439};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 439 done at `date`"
echo "four_algo 439 done at `date`">>concurrent.log

for n in {440..479};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 479 done at `date`"
echo "four_algo 479 done at `date`">>concurrent.log

for n in {480..519};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 519 done at `date`"
echo "four_algo 519 done at `date`">>concurrent.log

for n in {520..559};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 559 done at `date`"
echo "four_algo 559 done at `date`">>concurrent.log

for n in {560..599};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 599 done at `date`"
echo "four_algo 599 done at `date`">>concurrent.log

for n in {600..639};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 639 done at `date`"
echo "four_algo 639 done at `date`">>concurrent.log

for n in {640..679};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 679 done at `date`"
echo "four_algo 679 done at `date`">>concurrent.log

for n in {680..719};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 719 done at `date`"
echo "four_algo 719 done at `date`">>concurrent.log

for n in {720..759};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 759 done at `date`"
echo "four_algo 759 done at `date`">>concurrent.log

for n in {760..799};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 799 done at `date`"
echo "four_algo 799 done at `date`">>concurrent.log

for n in {800..839};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 839 done at `date`"
echo "four_algo 839 done at `date`">>concurrent.log

for n in {840..879};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 879 done at `date`"
echo "four_algo 879 done at `date`">>concurrent.log

for n in {880..899};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo 899 done at `date`"
echo "four_algo 899 done at `date`">>concurrent.log
