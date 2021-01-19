question=$1
train=$2
test=$3
output=$4
if [[ ${question} == "1" ]]; then
python q1a.py $train $test $output
fi
if [[ ${question} == "2" ]]; then
python q1b.py $train $test $output
fi
if [[ ${question} == "3" ]]; then
python q1c.py $train $test $output
fi