import tkinter as tk
import tkinter.filedialog
import os
import shutil
import tkinter.messagebox
from PIL import Image, ImageTk
from predict import a
import predict
import torch
from model import hybrid
from torchvision import transforms
import matplotlib.pyplot as plt
import json


# from count_detect import detect
import torch


class CountSoft:
    file_path = ''
    image_path = ''
    shiliu_num = 0
    font = ('微软雅黑', 12)
    color = '#A0EEE1'
############################################
    file ='./pics'
    imglist = os.listdir(file)
    i=0

    def __init__(self):
        root = tk.Tk()
        root.iconbitmap("a.ico")
        root.title("番茄病害识别系统")
        root.geometry('1100x580')

        # 石榴数量
        self.var = tk.StringVar()
###############
        # 菜单
        menu = tk.Menu(root)
        # 文件菜单
        filemenu = tk.Menu(menu, tearoff=0)
        filemenu.add_command(label='打开文件')
        filemenu.add_command(label='保存文件')
        filemenu.add_separator()
        filemenu.add_command(label='退出', command=root.quit)
        menu.add_cascade(label='  文件  ', menu=filemenu)
        # 编辑菜单
        editmenu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='  编辑  ', menu=editmenu)
        # 帮助菜单
        helpmenu = tk.Menu(menu, tearoff=0)
        helpmenu.add_command(label='使用帮助')
        helpmenu.add_command(label='关于')
        menu.add_cascade(label='  帮助  ', menu=helpmenu)
###############
       
        



        self.path_label = tk.Label(root,  font=self.font, bg=self.color)
        self.path_label.place(x=20, y=20, width=320, height=30)

        self.path_select_but = tk.Button(root, text='选择路径', font=self.font,  command=self.addfile,bg="#8CC7B5")
        self.path_select_but.place(x=370, y=20, width=100, height=30)
        self.img_label = tk.Label(root, text="此处叶片检测结果", font=self.font, bg=self.color)
        self.img_label.place(x=20, y=70, width=450, height=400)

        self.slice_label1 = tk.Label(root, text="此处依次读取分割后的叶片", font=self.font, bg=self.color,wraplength="200")
        self.slice_label1.place(x=600, y=50, width=200, height=200)
        self.show_label6 = tk.Label(root, text="当前叶片：", font=self.font,)
        self.show_label6.place(x=600, y=10, width=100, height=30)

        self.show_label7 = tk.Label(root, text="叶片病害：", font=self.font,)
        self.show_label7.place(x=500, y=270, width=100, height=30)
        self.show_label2 = tk.Label(root, text="此处显示病害种类", font=self.font, relief="raised",bg=self.color)
        self.show_label2.place(x=600, y=270, width=200, height=30)

        self.show_label7 = tk.Label(root, text="治疗建议：", font=self.font,)
        self.show_label7.place(x=500, y=320, width=100, height=30)
        self.tip_label = tk.Label(root, text="此处给出治疗建议", font=self.font, bg=self.color,relief="raised",wraplength="390")
        self.tip_label.place(x=500, y=350, width=400, height=150)

        self.button4 = tk.Button(root, text='切换叶片', font=self.font,command=self.switchfile,bg="#13ce66")
        self.button4.place(x=950, y=250, width=100, height=30)
        self.button1 = tk.Button(root, text='检测叶片', font=self.font,command=self.yo,bg="#FFBA00")
        self.button1.place(x=950, y=50, width=100, height=30)
        self.button2 = tk.Button(root, text='病害识别', font=self.font, command=self.todis,bg="#909399")
        self.button2.place(x=950, y=50+100, width=100, height=30)
        self.button3 = tk.Button(root, text='退出', font=self.font,bg="#ff4949")
        self.button3.place(x=950, y=350, width=100, height=30)



        root.config(menu=menu)
        root.mainloop()

#选择病害图像路径
    def addfile(self):
        self.file_path = tk.filedialog.askopenfilename()
        image_file = Image.open(self.file_path).resize((400, 400))
        photo = ImageTk.PhotoImage(image_file)
        self.img_label.config(image=photo)
        self.img_label.image = photo
        self.path_label.config(text=self.file_path)
        for pic in self.imglist:
            del_pic = os.path.join('./pics/',pic)
            os.remove(del_pic)

#检测叶片    
    def yo(self):
        os.system('python detect.py --source '+self.file_path)
        yore = os.path.split(self.file_path)
        print(yore)
        image_file = Image.open('./output/'+yore[1]).resize((400, 400))
        photo = ImageTk.PhotoImage(image_file)
        self.img_label.config(image=photo)
        self.img_label.image = photo
        self.imglist = os.listdir(self.file)
        
    
#切换叶片
    def switchfile(self):
        self.i += 1
        sub_path = self.file + '/' + self.imglist[self.i]
        print(self.imglist)
        print(sub_path)
        sub_file = Image.open(sub_path).resize((224, 224))
        photo = ImageTk.PhotoImage(sub_file)
        self.slice_label1.config(image=photo)
        self.slice_label1.image = photo
        s = predict.a(sub_path)
        self.show_label2.config(text=s[0])
        tip = self.tips(s[0])
        self.tip_label.config(text=tip)
        if self.i+1==len(self.imglist):
            self.i=-1
#病害识别        
    def todis(self):
        sub_path = self.file + '/' + self.imglist[0]
        print(sub_path)
        sub_file = Image.open(sub_path).resize((200, 200))
        photo = ImageTk.PhotoImage(sub_file)
        self.slice_label1.config(image=photo)
        self.slice_label1.image = photo
        s = predict.a(sub_path)
        self.show_label2.config(text=s[0])
        tip = self.tips(s[0])
        self.tip_label.config(text=tip)

    def tips(self, tbs):
        if tbs=='疮痂病':
            t='(1)建立无病种子田，确保种子不带菌是杜绝病害传播的根本措施;种子用1%次氯酸钠溶液+云大-120500倍液,浸种20-30分钟，再用清水冲洗干净后催芽播种。\n(2)初发病时用"天达2116"800倍液+"天达诺杀"1000倍液、77%多宁可湿性粉剂600倍液、70%可杀得101可湿性粉剂800倍液、新植霉素4000倍液、50%琥胶肥酸铜可湿性粉剂500倍液喷雾，每隔7-10天喷1次，连喷3次。'
            return t
        elif tbs=='早疫病':
            t='可喷施代森锰锌、杀毒矾、百菌清、甲霜灵锰锌。也可用代森锰锌或百菌清，于发病后，涂抹茎秆病部。如果茎基部布满黑褐色病斑，可用高锰酸钾灌根。'
            return t
        elif tbs=='晚疫病':
            t='1、发病初期，及时摘除病叶、病果及严重病枝，然后根据作物该时期并发病害情况，40%乙磷锰锌可湿性粉剂300倍液加嘧啶核苷类抗菌素500倍液喷施加嘧菌酯百菌清800倍液， 5—7天用药1次，连用2-3次。\n 2.发病较重时，清除中心病株、病叶等，及时采用中西医结合的防治方法。'
            return t
        elif tbs=='叶霉病':
            t='重病区及时摘去植株下部病叶及老叶，轻病区适时整枝打杈，以利通风透光。发病初期，可选用25%嘧菌酯悬浮剂1500~2000倍液，或70%甲基托布津可湿性粉剂800倍液，或47%春雷霉素？王铜可湿性粉剂500~600倍液，或75%百菌清可湿性粉剂600~800倍液喷，连喷2~3次，隔7d天喷一次。'
            return t
        elif tbs=='斑枯病':
            t='1.选择抗病品系造林。\n 2.后清扫烧毁或深埋。\n3.8月份的发病初、中期喷洒1：2：200倍波尔多液或65%代森锌400~500倍液。每半月喷洒一次，共喷2〜3次即可控制病情。'
            return t
        elif tbs=='红蜘蛛损伤':
            t='及时清除田间杂草和作物的残枝败叶，减少虫源，在害虫初发时喷药保护，可采用20%好年冬2000倍，1%杀虫素2000倍液喷杀，每隔7～10天喷一次。'
            return t
        elif tbs=='疤斑病':
            t='1、用靶斑净稀释600倍液喷施，6天用药一次 \n 2、轻微发病时，按照靶斑净400-600倍液稀释喷雾，5-7天用药一次；病情较重时，按靶斑净100-300倍喷施，3天用药一次，具体施药次数视病情而定'
            return t
        elif tbs=='黄叶曲叶病毒感染':
            t='1，使用西红柿（番茄）黄花曲叶病治疗病剂，15克兑水15公斤全株喷施，严重者结合灌根治疗，15克兑水20-30公斤，苗期每株50-60ML，开花期以后每株60-100ML。初期感染使用1-2次即可治愈。 \n 2，喷施杀虫剂，如杀灭烟粉虱、白粉虱、蚜虫、蓟马的药物，预防再次感染。'
            return t
        elif tbs=='花叶病':
            t='1、选用抗病品种，之豇28-2、铁线青等。/n 2、建立无病留种田，加强田间管理，保证肥水供应，尽量避免高温干旱出现，促使植株生长健壮，可减轻发病 \n 3、生长间尽量避免蚜虫危害，发现蚜虫及时用药防治。常用药剂有40%乐果乳油1000倍液，或10%吡虫啉可湿性粉剂2000倍液，或20%杀灭菊酯乳油8000倍液喷雾。\n 4、加强营养调节。用药期间补喷抗病毒植物生长调节剂。'
            return t
        else:
            t='健康'
            return t

       

    # def detect(self):
    #     path = self.file_path
    #     with torch.no_grad():
    #         self.shiliu_num = detect(mysource=path)
    #     dir_path = os.path.join(os.getcwd(), 'output')
    #     image_name = os.listdir(dir_path)[0]
    #     self.image_path = os.path.join(dir_path, image_name)
    #     image_file = Image.open(self.image_path).resize((700, 393))
    #     photo = ImageTk.PhotoImage(image_file)
    #     self.label_image.config(image=photo)
    #     self.label_image.image = photo
    #     self.var.set("检测到石榴的个数为：" + str(self.shiliu_num))
    #
    # def savephoto(self):
    #     path = tk.filedialog.asksaveasfilename(filetypes=(('jpg文件', '*.jpg'),))
    #     shutil.move(self.image_path, path)
    #     tk.messagebox.showinfo(title='提示', message='保存成功，' + str(path))



####################
        #  1  斜拍系列，
        # # 打开45度斜拍图片文件按钮
        # but_add_xiepai_image = tk.Button(root, text='打开图片', font=self.font)
        # but_add_xiepai_image.place(x=500, y=35, width=110, height=35)
        # # 开始检测45度斜拍图片按钮
        # but_jiance_xiepai_image = tk.Button(root, text='开始检测', font=self.font)
        # but_jiance_xiepai_image.place(x=500+110+35, y=35, width=110, height=35)
        # # 保存45度斜拍图片检测结果按钮
        # but_save_xiepai_image = tk.Button(root, text='保存检测结果', font=self.font)
        # but_save_xiepai_image.place(x=500+110+35+110+35, y=35, width=120, height=35)
        # # 45度斜拍图片显示标签
        # # image_file = Image.open(self.file_path).resize((700, 393))
        # # photo = ImageTk.PhotoImage(image_file)
        # self.xiepai_img_show_label = tk.Label(root, text="请选45度斜拍小麦幼苗图片", font=self.font, bg='#BEBEBE')
        # self.xiepai_img_show_label.place(x=20, y=20, width=451, height=300)
        # # 45度斜拍图片检测结果
        # self.label_image = tk.Label(root, text="45度斜拍图片检测结果", font=self.font, bg='#BEBEBE')
        # self.label_image.place(x=500, y=35+35+20, width=300, height=200)
###############

#         # 分割线
#         self.label_image = tk.Label(root, bg='#BEBEBE')
#         self.label_image.place(x=10, y=335, width=990, height=3)
# ################
#
# ################
#         #   2  俯拍系列，
#         # 打开俯拍图片文件按钮
#         but_add_fupai_image = tk.Button(root, text='打开图片', command=self.addfile, font=self.font)
#         but_add_fupai_image.place(x=500, y=350+10, width=110, height=35)
#         # 开始检测俯拍图片按钮
#         but_jiance_fupai_image = tk.Button(root, text='开始检测', command=self.addfile, font=self.font)
#         but_jiance_fupai_image.place(x=500+110+35, y=350+10, width=110, height=35)
#         # 保存俯拍图片检测结果按钮
#         but_save_fupai_image = tk.Button(root, text='保存检测结果', command=self.addfile, font=self.font)
#         but_save_fupai_image.place(x=500+110+35+110+35, y=350+10, width=120, height=35)
#         # 45度图片显示标签
#         self.label_image = tk.Label(root, text="请选择俯拍图片", font=self.font, bg='#BEBEBE')
#         self.label_image.place(x=20, y=20+300+30, width=451, height=300)
#         # 俯拍图片检测结果
#         self.label_image = tk.Label(root, text="俯拍图片检测结果", font=self.font, bg='#BEBEBE')
#         self.label_image.place(x=500, y=350+10+35+20, width=300, height=200)
#
#         # 生成检测报告按钮
#         but_jiance_fupai_image = tk.Button(root, text='生成检测报告', command=self.addfile, font=self.font)
#         but_jiance_fupai_image.place(x=20+450+30, y=680-70, width=410, height=35)
# #############
#
#
#
#
# #############
#         #   3
#         # 检测按钮
#         but_det = tk.Button(root, text='开始检测', command=self.detect, font=self.font)
#         but_det.place(x=20, y=483, width=100, height=30)
#         # 结果显示按钮
#         self.label_info = tk.Label(root, textvariable=self.var, font=self.font)
#         self.label_info.place(x=160, y=483, width=300, height=30)
#         # 保存图片按钮
#         but_save = tk.Button(root, text='保存图片', font=self.font, command=self.savephoto)
#         but_save.place(x=550, y=683, width=100, height=30)

c = CountSoft()
