from collections import defaultdict

def writetofile(data, dfile):
    with open(dfile, 'w') as f:
        for u, ilist in sorted(data.items()):
            f.write(str(u)+' '+' '.join(ilist))
            # for i in ilist:
            f.write('\n')


def transform_data(data):
    ori_file = f'{data_name}_ori.txt'
    gen_file = f'{data_name}.txt'
    item_count = defaultdict(int)
    user_items = dict()
    usermap = dict()
    lines = open(ori_file).readlines()
    for line in lines:
        user, item = line.split()
        user = int(user)
        if user not in usermap:
            usermap[user] = 1
            user_items[user] = []
        user_items[user].append(item)
    writetofile(user_items,gen_file)
data_names = ['beauty', 'ml-100k', 'sports', 'toys', 'yelp']
for data_name in data_names:
    transform_data(data_name)