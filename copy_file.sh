cd /mnt/data/signData/valid/landmark

home_path="/home/shd/BomNae-SLR"
ori_pwd=`pwd`;
ori_pwd=`echo $ori_pwd | cut -c 10-`
save_path="$home_path$ori_pwd"
#echo $save_path

for file in `cat landmark_files.txt`
do
  cp -r $file $save_path`echo $file | cut -c 2-4`
  #cp -r $file $save_path
done

#cd ..;
#cd morpheme';

#ori_pwd=`pwd`  
#cp -r $(find . -name *_F_*") ${ori_pwd:11}

