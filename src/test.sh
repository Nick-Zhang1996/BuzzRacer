for n in {1..5}
do
  sleep $n &
  echo "$n"
done

wait

for n in {1..5}
do
  sleep $n &
  echo "$n"
done
