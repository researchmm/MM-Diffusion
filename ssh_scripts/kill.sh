KEY=$1
ps -ef | grep $KEY | awk '{print $2}' | xargs kill -9