pushd ~/mxnet/data
wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -xf 256_ObjectCategories.tar

mkdir -p caltech_256_train_60
for i in 256_ObjectCategories/*; do
    c=`basename $i`
    mkdir -p caltech_256_train_60/$c
    for j in `ls $i/*.jpg | shuf | head -n 60`; do
        mv $j caltech_256_train_60/$c/
    done
done

python3 ../im2rec.py --list True --recursive True caltech-256-60-train caltech_256_train_60/
python3 ../im2rec.py --list True --recursive True caltech-256-60-val 256_ObjectCategories/
python3 ../im2rec.py --resize 256 --quality 90 --num-thread 16 caltech-256-60-val 256_ObjectCategories/
python3 ../im2rec.py --resize 256 --quality 90 --num-thread 16 caltech-256-60-train caltech_256_train_60/
popd