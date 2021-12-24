import smtplib
from email.mime.text import MIMEText
from email.header import Header

def TitleWriter():
	return '亲爱的同志'

def SendMail(message, subject=None):
	if subject == None:
		subject = '实验跑完了'
	subject = TitleWriter() + '，' + subject

	sender = 'aka_xia@qq.com'
	receivers = ['aka_xia@qq.com']
	message = MIMEText(message, 'plain', 'utf-8')
	message['Subject'] = Header(subject, 'utf-8')
	message['From'] = '智能实验管理委员会'
	message['To'] = '人工实验处理委员会'

	mail_host = 'smtp.qq.com'
	mail_user = 'aka_xia@qq.com'
	mail_pass = 'kpuroslsvybmbbad'

	smtpObj = smtplib.SMTP()
	smtpObj.connect(mail_host, 25)
	smtpObj.login(mail_user, mail_pass)
	smtpObj.sendmail(sender, receivers, message.as_string())

