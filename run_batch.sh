for i in 1 2 3 4 6 7 8 9 10; do 
     screen -S hy190$i -dm bash -c "python Scenario1.py -z_eval $i; sleep 500";
    i=$((i+1))
done