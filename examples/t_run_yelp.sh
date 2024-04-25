
#for i in 0.06 0.08 0.1 0.2
#do
#  echo "start: ${i}"
#  python run.py --user_temperature ${i}
#done

#echo "start: item_temperature"
#for i in 0.02 0.2
#do
#  echo "start: ${i}"
#  python run.py --item_temperature ${i}
#done

#echo "start: embedding"
#for i in 40 50 60 70 80 90
#do
#  echo "start: ${i}"
#  python run.py --embedding_dim ${i}
#done

echo  "start: ratio"
for i in 0.6 0.7 0.8 0.9
do
  echo "start: ${i}"
  python run.py --ratio ${i}
done
