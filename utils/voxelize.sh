name=$1
res=$2
center=$3
display=$4
bbox=$5



if [[ -n $name && -n $res ]]; then
    Xvfb :${display} -screen 0 1900x1080x24 &
    export DISPLAY=:${display}
    if [[ $3 == "c" ]]; then
        ./binvox $5 $6 $7 $8 $9 ${10} ${11} -aw -e -cb -d $2 -nf 0.9 $1
    elif [[ $3 == "nc" ]]; then
        ./binvox $5 $6 $7 $8 $9 ${10} ${11} -aw -e -d $2 -nf 0.9 $1
    else
        echo "the 3rd arg can either be 'c' (center) or 'nc' (don't center)"
    fi
else
    echo "please specify input mesh name and resolution"
fi
