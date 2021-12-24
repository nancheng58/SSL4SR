import os
import sys

def diff(file1, file2):
	os.system('powershell;$a=Get-Content ' + file1 + ';$b=Get-Content ' + file2 + ';diff $a $b')

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('python diff.py $file1 $file2')
	else:
		diff(sys.argv[1], sys.argv[2])