# generate shell script to run concurrent command

start_text = '''
# four algo: 900 (0-899)
echo "starting at `date`"
echo "starting at `date`">>concurrent.log
'''
text = '''
for n in {%(start)s..%(end)s};
do
  python batchRunSingle.py four_algo $n > "../log/concurrent/log$n.txt" 2>&1 &
done
wait
echo "four_algo %(end)s done at `date`"
echo "four_algo %(end)s done at `date`">>concurrent.log
'''


with open('autogen_concurrent.sh','w') as f:
    f.writelines(start_text)
    for i in range(0,900,40):
        script = text%{'start':i,'end':min(i+39,899)}
        f.writelines(script)

