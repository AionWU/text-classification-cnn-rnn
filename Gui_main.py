import tkinter as tk

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab

# 模型部分
try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
class Model:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]
# Gui部分
def begin_predict():
    str = inp1.get('0.0', 'end')
    textvar1.set(model.predict(str))


# MAIN
model=Model()

window = tk.Tk()
window.title('新闻内容类型识别')
window.geometry('800x500')
# 控件
label1 = tk.Label(window, text='请输入一段文字内容：', font=('Arial', 12), width=30, height=2)
label1.pack()
inp1 = tk.Text(window)
inp1.pack()
btn1 = tk.Button(window, text='确定', width=10,
                 height=1, command=begin_predict)
btn1.pack()
textvar1 = tk.StringVar()
label2 = tk.Label(window, textvariable=textvar1, font=('Arial', 12), width=30, height=2)
label2.pack()
label3 = tk.Label(window, text='类型预测：', font=('Arial', 12), width=30, height=2)
label3.pack()

textvar_tips = tk.StringVar()
tips = tk.Label(window,textvariable=textvar1,font=('Arial', 12), width=30, height=2)
tips.place(x=0,y=400)
# 显示
window.mainloop()