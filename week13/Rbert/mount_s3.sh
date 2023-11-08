chmod 600 ${HOME}/s3_pass

s3fs deeplearningbucket ./s3 -o passwd_file=${HOME}/s3_pass

sudo usermod -a -G root ubuntu