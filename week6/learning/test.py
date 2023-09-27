# 使用标准库进行文件读写
path = '/Users/jianxiaoyang/Downloads'
with open(path+'/20230925-100429.png', 'rb') as file:
    blob_data = file.read()

# 如果需要，可以将blob数据保存到另一个文件
with open(path+'/output.blob', 'wb') as blob_file:
    blob_file.write(blob_data)
