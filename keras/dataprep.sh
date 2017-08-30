pushd ~/keras/data
curl -L -O http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -xf 256_ObjectCategories.tar
mkdir -p 256_ObjectCategories/train 256_ObjectCategories/test 256_ObjectCategories/valid
find ./256_ObjectCategories/ -regextype posix-extended -type f -regex ".*/[0-9]{3}\..+" -print | shuf | head -n 5000 | xargs -I {} mv {} ./256_ObjectCategories/valid/
find ./256_ObjectCategories/ -regextype posix-extended -type f -regex ".*/[0-9]{3}\..+" -print | shuf | head -n 6000 | xargs -I {} mv {} ./256_ObjectCategories/test/
find ./256_ObjectCategories/ -regextype posix-extended -type f -regex ".*/[0-9]{3}\..+" -print | xargs -I {} mv {} ./256_ObjectCategories/train/
find ./256_ObjectCategories/ -regextype posix-extended -type d -regex ".*/[0-9]{3}\..+" -delete
popd