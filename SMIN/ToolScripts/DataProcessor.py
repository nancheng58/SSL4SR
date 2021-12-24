import random
import gc

def RandomShuffle(infile, outfile, deleteSchema=False):
	with open(infile, 'r', encoding='utf-8') as fs:
		arr = fs.readlines()
	if not arr[-1].endswith('\n'):
		arr[-1] += '\n'
	if deleteSchema:
		arr = arr[1:]
	random.shuffle(arr)
	with open(outfile, 'w', encoding='utf-8') as fs:
		for line in arr:
			fs.write(line)
	del arr

def SubDataSet(infile, outfile1, outfile2, rate):
	out1 = list()
	out2 = list()
	with open(infile, 'r', encoding='utf-8') as fs:
		for line in fs:
			if random.random() <= rate:
				out1.append(line)
			else:
				out2.append(line)
	with open(outfile1, 'w', encoding='utf-8') as fs:
		for line in out1:
			fs.write(line)
	with open(outfile2, 'w', encoding='utf-8') as fs:
		for line in out2:
			fs.write(line)

#5-fold cross-validation
def SubDataSet2(infile, outfile1, outfile2, outfile3, outfile4, outfile5):
	out1 = list()
	out2 = list()
	out3 = list()
	out4 = list()
	out5 = list()
	with open(infile, 'r', encoding='utf-8') as fs:
		for line in fs:
			r = random.random()
			if r >= 0.0 and r < 0.2:
				out1.append(line)
			elif r >=0.2 and r < 0.4:
				out2.append(line)
			elif r >= 0.4 and r < 0.6:
				out3.append(line)
			elif r >= 0.6 and r < 0.8:
				out4.append(line)
			elif r >= 0.8 and r < 1.0:
				out5.append(line)
			else:
				print("bug")

	print("%d %d %d %d %d"%(len(out1), len(out2), len(out3), len(out4), len(out5)))
	count = 0
	with open(outfile1, 'w', encoding='utf-8') as fs:
		for line in out1:
			count += 1
			fs.write(line)
	print("out1 = %d"%(count))
	count = 0
	with open(outfile2, 'w', encoding='utf-8') as fs:
		for line in out2:
			count += 1
			fs.write(line)
	print("out2 = %d"%(count))
	count = 0
	with open(outfile3, 'w', encoding='utf-8') as fs:
		for line in out3:
			count += 1
			fs.write(line)
	print("out3 = %d"%(count))
	count = 0
	with open(outfile4, 'w', encoding='utf-8') as fs:
		for line in out4:
			count += 1
			fs.write(line)
	print("out4 = %d"%(count))
	count = 0
	with open(outfile5, 'w', encoding='utf-8') as fs:
		for line in out5:
			count += 1
			fs.write(line)
	print("out5 = %d"%(count))


# out1 = 1639065
# out2 = 1638097
# out3 = 1641045
# out4 = 1635613
# out4 = 1642257
