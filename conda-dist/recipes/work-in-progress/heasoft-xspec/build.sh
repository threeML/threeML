#!/bin/bash


export CFLAGS='-I${PREFIX}/include -I. -O2 -Wall --pedantic -Wno-comment -Wno-long-long -g  -ffloat-store -fPIC'
export CXXFLAGS='-I${PREFIX}/include -I. -O2 -Wall --pedantic -Wno-comment -Wno-long-long -g  -ffloat-store -fPIC'
export CPPFLAGS="-I${PREFIX}/include"
export LDFLAGS="-Wl,-rpath,${PREFIX}/lib"
export PERL=${PREFIX}/bin/perl
export PERL5LIB=${PREFIX}/lib/perl5/5.22.2

case "$(uname)" in
Linux)
    
    export CFLAGS="${CFLAGS} -ffloat-store"
    export CXXFLAGS="${CXXFLAGS} -ffloat-store"
    
;;
Darwin)
    # We need to pad the header for rpath, otherwise relinking will fail
    export LDFLAGS="-headerpad_max_install_names $LDFLAGS"
;;
esac


$PERL --version


# Make a wrapper around cc with the proper MACOSX_DEPLOYMENT_TARGET, because perl packages guess it wrong
echo "#!/bin/bash" > cc
echo 'env MACOSX_DEPLOYMENT_TARGET=10.5 clang $@' >> cc
chmod u+x cc
export PATH=`pwd`:$PATH

# Remove software we don't need to compile
mkdir delete

mv demo heasim hitomi integral nicer nustar suzaku swift delete

# Create an empty configure and Makefile files
mkdir dumb_package

echo "echo FAKE CONFIGURE" > dumb_package/configure
chmod u+x dumb_package/configure
echo "main: ;" > dumb_package/Makefile
echo "all: ;" >> dumb_package/Makefile
echo "install: ;" >> dumb_package/Makefile
echo "shared: ;" >> dumb_package/Makefile
echo "tclxpa: ;" >> dumb_package/Makefile
echo "installdirs: ;" >> dumb_package/Makefile
echo "install-headers: ;" >> dumb_package/Makefile

# Remove readline and substitute it with dumb package
mkdir -p delete/heacore
mv heacore/readline delete/heacore
mkdir heacore/readline
cp dumb_package/* heacore/readline

mkdir heacore/readline/shlib
cp dumb_package/* heacore/readline/shlib

# get the public cfitsio and substitute the one embedded, to make
# sure we are compatible
# First get the version of the cfitsio installed (removing the .)
cfitsio_version=`conda list cfitsio | grep '^cfitsio ' | tr -s " " | cut -f2 -d" " | tr -d "."`
curl ftp://heasarc.gsfc.nasa.gov/software/fitsio/c/cfitsio${cfitsio_version}.tar.gz -o cfitsio.tar.gz

tar xvf cfitsio.tar.gz
rm -rf heacore/cfitsio
mv cfitsio heacore/

#cd heacore/BUILD_DIR

# Fix versions of cfitsio and readline
# Get defined versions
cd heacore/BUILD_DIR
export OLD_CFITSIO_VERS=`./hd_scanenv ./hd_config_info CFITSIO_VERS`
export OLD_READLINE_VERS=`./hd_scanenv ./hd_config_info READLINE_VERS`
cd ../..

# Change them to an empty string in all packages, so that the linking will happen on the lib*.so file
# instead of lib*[version].so file (which might not match)

echo "Changing versions of readline and cfitsio"

for file in `find . -name "hd_config_info"`
do
    echo $file
    case "$(uname)" in
    Linux)
        sed -i "s/CFITSIO_VERS=${OLD_CFITSIO_VERS}/CFITSIO_VERS=''/g" $file || true
        sed -i "s/READLINE_VERS=${OLD_READLINE_VERS}/READLINE_VERS=''/g" $file || true
    ;;
    Darwin)
        sed -i '' "s/CFITSIO_VERS=${OLD_CFITSIO_VERS}/CFITSIO_VERS=''/g" $file || true
        sed -i '' "s/READLINE_VERS=${OLD_READLINE_VERS}/READLINE_VERS=''/g" $file || true
    ;;
    *)
        echo "Unsupported"
        exit 1
    ;;
    esac
done

echo "done"

echo "Fixing architecture"

# Fix architecture (on OSX we cannot build fat libraries because readline in conda only supports x86_64)
for file in `find . \( -name "config*" -o -name "hmakerc" \)`
do
    echo $file
    case "$(uname)" in
    Linux)
        sed -i "s/-arch i386 -arch x86_64/-arch x86_64/g" $file || true
        sed -i "s/-arch i386/-arch x86_64/g" $file || true
    ;;
    Darwin)
        sed -i '' "s/-arch i386 -arch x86_64/-arch x86_64/g"  $file || true
        sed -i '' "s/-arch i386/-arch x86_64/g"  $file || true
    ;;
    *)
        echo "Unsupported"
        exit 1
    ;;
    esac
done

echo "done"

# Go to the main build dir
cd BUILD_DIR

autoconf -i

./configure --prefix=${PREFIX} --exec_prefix=${PREFIX} LDFLAGS="${LDFLAGS}" --x-libraries=${PREFIX}/lib --x-includes=${PREFIX}/include --with-components="heacore tcltk attitude heatools heagen Xspec ftools" 

# Go into heacore and select the components to build (ignore Perl modules)
cd ../heacore/BUILD_DIR

./configure --prefix=${PREFIX} --exec_prefix=${PREFIX} LDFLAGS="${LDFLAGS}" --enable-mac_32bit=no --x-libraries=${PREFIX}/lib --x-includes=${PREFIX}/include # --with-components="lynx cfitsio ape heaio heautils ahlog heainit ahgen ahfits heaapp mpfit ast wcslib CCfits heasp hoops st_stream pilperl hdutilsperl hdinitperl"

# Now go to the tclreadline and make sure we are using the readline from conda (on OSX it finds the system one and fails otherwise)
cd ../../tcltk/tclreadline
./configure --with-readline-includes=${PREFIX}/include/readline --with-readline-library=${PREFIX}/lib --prefix=${PREFIX} --exec_prefix=${PREFIX} --x-libraries=${PREFIX}/lib --x-includes=${PREFIX}/include 
cd ../../

# Now fix the HD_EXEC_PFX and the HD_TOP_EXEC_PFX by removing the host, otherwise they would become
# like $HD_TOP_EXEC_PFX=${PREFIX/x86_64-apple-darwin16.7.0 instead of just ${PREFIX}
host=`cat BUILD_DIR/config.log | grep '^host=' | cut -f2 -d"=" | cut -f2 -d"'"`
for file in `find . \( -name "config*" -o -name "hmakerc" \)`
do
    echo $file
    case "$(uname)" in
    Linux)
        sed -i "s:${PREFIX}/$host:${PREFIX}:g" $file || true
    ;;
    Darwin)
        sed -i '' "s:${PREFIX}/$host:${PREFIX}:g" $file || true
    ;; 
    *)
        echo "Unsupported"
        exit 1
    ;;
    esac
done


for file in `find . \( -name "config*" -o -name "hmakerc" -o -name "Makefil*" \)`
do
    echo $file
    case "$(uname)" in
    Linux)
        sed -i "s:HD_TOP_EXEC_PFX=\"\":HD_TOP_EXEC_PFX=${PREFIX}:g" $file || true
        
    ;;
    Darwin)
        sed -i '' "s:HD_TOP_EXEC_PFX=\"\":HD_TOP_EXEC_PFX=${PREFIX}:g" $file || true
    ;;
    *)
        echo "Unsupported"
        exit 1
    ;;
    esac
done

echo "done"

#for file in `grep -R '#!/usr/bin/perl' * | cut -f1 -d":" | grep -v 'man/man'`
#do
#    echo $file
#    case "$(uname)" in
#    Linux)
#        sed -i  's:#\!/usr/bin/perl:#\!/usr/bin/env perl:g' $file
#        
#    ;;
#    Darwin)
#        sed -i '' 's:#\!/usr/bin/perl:#\!/usr/bin/env perl:g' $file
#    ;;
#    *)
#        echo "Unsupported"
#        exit 1
#    ;;
#    esac
#done

# Finally we can start the build

cd heacore/BUILD_DIR
./hmake HD_EXEC_PFX=${PREFIX} HD_TOP_EXEC_PFX=${PREFIX} LD_LIBRARY_PATH=${PREFIX}/lib

cd ../../BUILD_DIR

./hmake HD_EXEC_PFX=${PREFIX} HD_TOP_EXEC_PFX=${PREFIX} LD_LIBRARY_PATH=${PREFIX}/lib

#echo "done"


./hmake install HD_EXEC_PFX=${PREFIX} HD_TOP_EXEC_PFX=${PREFIX}

## Remove the BUILD_DIR
#rm -rf BUILD_DIR
#rm -rf ${PREFIX}/BUILD_DIR
#
## Move the refdata from $PREFIX/refdata to $PREFIX/etc/heacore/refdata
#mkdir -p ${PREFIX}/etc/heacore/refdata
#mv ${PREFIX}/refdata/* ${PREFIX}/etc/heacore/refdata
#
## Move the guide in share
#mkdir -p ${PREFIX}/share/heacore
#mv ${PREFIX}/help/* ${PREFIX}/share/heacore
#
## Move the pfiles
#mkdir -p ${PREFIX}/etc/heacore/syspfiles
#mv ${PREFIX}/syspfiles/* ${PREFIX}/etc/heacore/syspfiles
#
## Move the perl scripts
#mkdir temp
#mv ${PREFIX}/lib/perl/* temp/
#mkdir -p ${PREFIX}/lib/perl/heacore
#mv temp/* ${PREFIX}/lib/perl/heacore
#rm -rf temp
