# Remove bootstrap module which causes issues
# (since we already know that everything is in place before this
# script starts)
rm -rf ah_bootstrap.py

echo "" > ah_bootstrap.py

python setup.py install  --single-version-externally-managed --record=record.txt
