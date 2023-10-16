train_sample = 100
batch_size = 8
for batch_index in range(train_sample // batch_size + 1):
    start = batch_index * batch_size
    end = start + batch_size
    if end > train_sample:
        end = train_sample
    print(start, end)