if [ -f $3]; then
    rm -f $3
fi
python main.py $1 $2 $3 $4 $5 $6