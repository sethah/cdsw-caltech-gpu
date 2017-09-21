%pushd /home/cdsw/mxnet-gluon
!mkdir data
!python3 ./im2rec.py --list True --recursive True data/caltech-valid /home/cdsw/train_data/256_ObjectCategories/valid
!python3 ./im2rec.py --list True --recursive True data/caltech-train /home/cdsw/train_data/256_ObjectCategories/train
!python3 ./im2rec.py --list True --recursive True data/caltech-test /home/cdsw/train_data/256_ObjectCategories/test
!python3 ./im2rec.py --newsize 224,224 --quality 100 --num-thread 16 data/caltech-valid /home/cdsw/train_data/256_ObjectCategories/valid
!python3 ./im2rec.py --newsize 224,224 --quality 100 --num-thread 16 data/caltech-train /home/cdsw/train_data/256_ObjectCategories/train
!python3 ./im2rec.py --newsize 224,224 --quality 100 --num-thread 16 data/caltech-test /home/cdsw/train_data/256_ObjectCategories/test
%popd

