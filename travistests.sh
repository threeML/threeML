#source condasetup.sh

# This is needed to make matplotlib plot testing work
#if [[ $TRAVIS_OS_NAME == 'linux' ]] || [[ -z "$DISPLAY" ]]; then
#if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
#    export DISPLAY=:99.0;
#    sh -e /etc/init.d/xvfb start;
#    export QT_API=pyqt;
#else
#    export DISPLAY=:99.0;
#    /usr/bin/Xvfb :99 -screen 0 1280x1024x24 &
#fi

export MPLBACKEND='Agg'

python -m pytest -vv --cov=threeML # --cov-config=fermipy/tests/coveragerc --durations=30
#status=$?
