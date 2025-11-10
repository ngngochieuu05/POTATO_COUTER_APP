import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from PIL import Image, ImageTk
import time
from datetime import datetime
import json
import os


class He_Thong_Dem_Khoai_Tay_Pro:
    def __init__(self, root):
        self.root = root
        self.Khoi_Tao_Bien_Co_Ban()
        self.Tai_Cau_Hinh()
        self.Khoi_Tao_Thanh_Phan_AI()
        self.Thiet_Lap_Giao_Dien()
        self.Cap_Nhat_Thoi_Gian()

    def Khoi_Tao_Bien_Co_Ban(self):
        """Khoi tao cac bien cot loi"""
        # Trang thai xu ly
        self.model = None
        self.cap = None
        self.dang_xu_ly = False
        self.dang_tam_dung = False
        self.khung_hinh_hien_tai = None
        self.da_tai_du_lieu = False
        self.thoi_gian_hen_gio = None
        self.hen_gio_dang_hoat_dong = False
        self.luong_hen_gio = None
        self.cua_so_ho_tro = None

        # Che do xu ly
        self.che_do_xu_ly = "dem"  # "dem" hoac "chat_luong"

        # Du lieu dem
        self.tong_so_dem = 0
        self.danh_sach_id_da_dem = set()
        self.khoai_tay_da_phat_hien = {}

        # Du lieu chat luong - chi con 2 trang thai
        self.ket_qua_chat_luong = {
            'tot': 0,
            'kem': 0,
            'chi_tiet': {}  # {tracker_id: thong_tin_chat_luong}
        }

        # Nguon dau vao
        self.nguon_dau_vao = "file"  # "file", "camera", "image"
        self.chi_so_camera = 0

        # Thong ke
        self.thoi_gian_bat_dau_xu_ly = 0
        self.so_khung_hinh_da_xu_ly = 0
        self.phat_hien_cuoi = sv.Detections.empty()  # Khoi tao detection rong

        # Chon vung
        self.dang_chon_vung = False
        self.diem_vung = []
        self.diem_tam_thoi = []
        self.hien_thi_vung = None

        # Cau hinh mac dinh
        self.diem_vung_goc = [
            [272, 753], [301, 156], [1424, 176], [1395, 785],
            [1133, 778], [1133, 671], [1026, 673], [456, 666], [458, 768]
        ]

        self.cau_hinh = {
            "duong_dan_model": " ",
            "nguon_video": " ",
            "nguon_anh": " ",
            "nguong_tin_cay": 0.5,
            "nguong_iou": 0.5,
            "nguong_chat_luong": 0.50,  # Nguong phan chia tot/kem
            "bo_qua_khung_hinh": 2,
            "kich_thuoc_muc_tieu": (640, 640),
            "diem_vung": self.diem_vung_goc.copy(),
            "chi_so_camera": 0,
            "hien_toa_do": True,
            "duong_dan_xuat": ""  # Duong dan xuat file mac dinh
        }


    def Khoi_Tao_Thanh_Phan_AI(self):
        self.bo_theo_doi = sv.ByteTrack()
        
        # Tao 2 bo ve khung rieng cho tung loai chat luong
        self.bo_ve_khung_tot = sv.BoxAnnotator(
            thickness=3,
            color=sv.Color.from_hex("#00FF00")  # Mau xanh la cho chat luong tot
        )
        self.bo_ve_khung_kem = sv.BoxAnnotator(
            thickness=3,
            color=sv.Color.from_hex("#FF0000")  # Mau do cho chat luong kem
        )
        self.bo_ve_khung_mac_dinh = sv.BoxAnnotator(
            thickness=3,
            color=sv.Color.from_hex("#FFFF00")  # Mau vang cho chua xac dinh
        )
    
        self.bo_ve_nhan = sv.LabelAnnotator(text_thickness=2, text_scale=0.7)

    def Thiet_Lap_Giao_Dien(self):
        """Thiet lap giao dien nguoi dung"""
        self.Cau_Hinh_Cua_So()
        self.Cau_Hinh_Phong_Cach()

        # Khung chinh
        khung_chinh = tk.Frame(self.root, bg='#1e1e1e')
        khung_chinh.pack(fill='both', expand=True, padx=10, pady=10)

        self.Tao_Tieu_De(khung_chinh)

        khung_noi_dung = tk.Frame(khung_chinh, bg='#1e1e1e')
        khung_noi_dung.pack(fill='both', expand=True, pady=(10, 0))

        self.Tao_Bang_Dieu_Khien(khung_noi_dung)
        self.Tao_Bang_Video(khung_noi_dung)
        self.Tao_Thanh_Trang_Thai(khung_chinh)

    def Cau_Hinh_Cua_So(self):
        """Cau hinh cua so chinh"""
        self.root.title("HE THONG DEM & KIEM TRA CHAT LUONG KHOAI TAY THONG MINH (Nguyen_Ngoc_Hieu - NguyenTungDuong)")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        self.root.minsize(1000, 700)

    def Cau_Hinh_Phong_Cach(self):
        """Cau hinh phong cach cho giao dien"""
        style = ttk.Style()
        style.theme_use('clam')

        phong_cach = {
            'Title.TLabel': {'font': ('Segoe UI', 20, 'bold'), 'background': '#1e1e1e', 'foreground': '#ffffff'},
            'Subtitle.TLabel': {'font': ('Segoe UI', 10), 'background': '#1e1e1e', 'foreground': '#cccccc'},
            'Modern.TButton': {'font': ('Segoe UI', 9, 'bold'), 'padding': 8},
            'Count.TLabel': {'font': ('Segoe UI', 16, 'bold'), 'background': '#2d2d2d', 'foreground': '#00ff88',
                             'padding': 15}
        }

        for ten_phong_cach, cau_hinh in phong_cach.items():
            style.configure(ten_phong_cach, **cau_hinh)

    def Tao_Tieu_De(self, parent):
        """Tao tieu de"""
        khung_tieu_de = tk.Frame(parent, bg='#1e1e1e')
        khung_tieu_de.pack(fill='x', pady=(0, 10))

        khung_tieu_de_chinh = tk.Frame(khung_tieu_de, bg='#2d2d2d', relief='ridge', bd=2)
        khung_tieu_de_chinh.pack(fill='x', pady=5)

        ttk.Label(khung_tieu_de_chinh, text="HE THONG DEM KHOAI TAY & KIEM TRA CHAT LUONG PRO MAX",
                  style='Title.TLabel').pack(pady=10)
        ttk.Label(khung_tieu_de_chinh, text="-- He thong dem va kiem tra chat luong khoai tay --",
                  style='Subtitle.TLabel').pack(pady=(0, 10))

    def Tao_Bang_Dieu_Khien(self, parent):
        """Tao bang dieu khien"""
        khung_trai = tk.Frame(parent, bg='#2d2d2d', relief='raised', bd=2, width=340)
        khung_trai.pack(side='left', fill='y', padx=(0, 10))
        khung_trai.pack_propagate(False)

        # Canvas va thanh cuon
        canvas = tk.Canvas(khung_trai, bg='#2d2d2d', highlightthickness=0)
        thanh_cuon = ttk.Scrollbar(khung_trai, orient="vertical", command=canvas.yview)
        self.khung_cuon = tk.Frame(canvas, bg='#2d2d2d')

        def Gan_Cuon_Chuot(event):
            canvas.bind_all("<MouseWheel>", Xu_ly_Cuon_Chuot)

        def Bo_Gan_Cuon_Chuot(event):
            canvas.unbind_all("<MouseWheel>")

        def Xu_ly_Cuon_Chuot(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def Cau_Hinh_Vung_Cuon(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.khung_cuon.bind("<Configure>", Cau_Hinh_Vung_Cuon)
        canvas.bind('<Enter>', Gan_Cuon_Chuot)
        canvas.bind('<Leave>', Bo_Gan_Cuon_Chuot)

        canvas.create_window((0, 0), window=self.khung_cuon, anchor="nw")
        canvas.configure(yscrollcommand=thanh_cuon.set)

        canvas.pack(side="left", fill="both", expand=True)
        thanh_cuon.pack(side="right", fill="y")

        # Cac phan cua bang dieu khien
        self.Tao_Lua_Chon_Che_Do()
        self.Tao_Lua_Chon_Nguon()
        self.Tao_Lua_Chon_File()
        self.Tao_Phan_Tai_Du_Lieu()
        self.Tao_Lua_Chon_Vung()
        self.Tao_Phan_Tham_So()
        self.Tao_Cai_Dat_Xuat_File()
        self.Tao_Nut_Dieu_Khien()
        self.Tao_Phan_Thong_Ke()
        self.Tao_Phan_Ho_Tro_Khach_Hang()
        self.Tao_Phan_Hen_Gio()

        self.root.after(100, lambda: self.khung_cuon.event_generate('<Configure>'))

    def Tao_Lua_Chon_Che_Do(self):
        """Tao phan lua chon che do xu ly"""
        phan = tk.LabelFrame(self.khung_cuon, text="CHE DO XU LY",
                                font=('Segoe UI', 9, 'bold'), bg='#2d2d2d', fg='#ffffff', bd=2)
        phan.pack(fill='x', padx=10, pady=5)

        self.bien_che_do = tk.StringVar(value="dem")

        tk.Radiobutton(phan, text="DEM KHOAI TAY", variable=self.bien_che_do, value="dem",
                       bg='#2d2d2d', fg='#ffffff', selectcolor='#4d4d4d',
                       command=self.Xu_ly_Thay_Doi_Che_Do).pack(anchor='w', padx=8, pady=3)

        tk.Radiobutton(phan, text="KIEM TRA CHAT LUONG (TOT/KEM)", variable=self.bien_che_do, value="chat_luong",
                       bg='#2d2d2d', fg='#ffffff', selectcolor='#4d4d4d',
                       command=self.Xu_ly_Thay_Doi_Che_Do).pack(anchor='w', padx=8, pady=(0, 8))

    def Tao_Lua_Chon_Nguon(self):
        """Tao phan lua chon nguon dau vao"""
        phan = tk.LabelFrame(self.khung_cuon, text="NGUON DAU VAO",
                                font=('Segoe UI', 9, 'bold'), bg='#2d2d2d', fg='#ffffff', bd=2)
        phan.pack(fill='x', padx=10, pady=5)

        self.bien_nguon = tk.StringVar(value="file")

        khung_radio = tk.Frame(phan, bg='#2d2d2d')
        khung_radio.pack(fill='x', pady=8)

        tk.Radiobutton(khung_radio, text="VIDEO", variable=self.bien_nguon, value="file",
                       bg='#2d2d2d', fg='#ffffff', selectcolor='#4d4d4d',
                       command=self.Xu_ly_Thay_Doi_Nguon).pack(side='left', padx=(8, 10))

        tk.Radiobutton(khung_radio, text="CAMERA", variable=self.bien_nguon, value="camera",
                       bg='#2d2d2d', fg='#ffffff', selectcolor='#4d4d4d',
                       command=self.Xu_ly_Thay_Doi_Nguon).pack(side='left', padx=(0, 10))

        tk.Radiobutton(khung_radio, text="ANH", variable=self.bien_nguon, value="image",
                       bg='#2d2d2d', fg='#ffffff', selectcolor='#4d4d4d',
                       command=self.Xu_ly_Thay_Doi_Nguon).pack(side='left')

        # Cai dat camera
        self.khung_camera = tk.Frame(phan, bg='#2d2d2d')

        tk.Label(self.khung_camera, text="CHI SO CAMERA:", bg='#2d2d2d', fg='#cccccc',
                 font=('Segoe UI', 8)).pack(side='left', padx=(8, 0))
        self.bien_camera = tk.IntVar(value=0)
        tk.Spinbox(self.khung_camera, from_=0, to=10, width=4, textvariable=self.bien_camera,
                   bg='#3d3d3d', fg='#ffffff', font=('Segoe UI', 8)).pack(side='right', padx=(5, 8))

    def Tao_Lua_Chon_File(self):
        """Tao phan lua chon file"""
        phan = tk.LabelFrame(self.khung_cuon, text="CHON TAP TIN",
                                font=('Segoe UI', 9, 'bold'), bg='#2d2d2d', fg='#ffffff', bd=2)
        phan.pack(fill='x', padx=10, pady=5)

        # File model
        tk.Label(phan, text="TAP TIN MODEL (.pt):", bg='#2d2d2d', fg='#cccccc',
                 font=('Segoe UI', 8)).pack(anchor='w', padx=8, pady=(8, 2))
        khung_model = tk.Frame(phan, bg='#2d2d2d')
        khung_model.pack(fill='x', padx=8, pady=(0, 5))

        self.bien_duong_dan_model = tk.StringVar(value=self.cau_hinh["duong_dan_model"])
        tk.Entry(khung_model, textvariable=self.bien_duong_dan_model, bg='#3d3d3d', fg='#ffffff',
                 font=('Segoe UI', 8)).pack(side='left', expand=True, fill='x')
        ttk.Button(khung_model, text="...", width=3, command=self.Duyet_File_Model).pack(side='right', padx=(3, 0))

        # File video
        self.nhan_video = tk.Label(phan, text="TAP TIN VIDEO:", bg='#2d2d2d', fg='#cccccc',
                                    font=('Segoe UI', 8))
        self.nhan_video.pack(anchor='w', padx=8, pady=(5, 2))
        self.khung_video = tk.Frame(phan, bg='#2d2d2d')
        self.khung_video.pack(fill='x', padx=8, pady=(0, 8))

        self.bien_duong_dan_video = tk.StringVar(value=self.cau_hinh["nguon_video"])
        tk.Entry(self.khung_video, textvariable=self.bien_duong_dan_video, bg='#3d3d3d', fg='#ffffff',
                 font=('Segoe UI', 8)).pack(side='left', expand=True, fill='x')
        ttk.Button(self.khung_video, text="...", width=3, command=self.Duyet_File_Video).pack(side='right',
                                                                                               padx=(3, 0))

        # File anh
        self.nhan_anh = tk.Label(phan, text="TAP TIN ANH:", bg='#2d2d2d', fg='#cccccc',
                                    font=('Segoe UI', 8))
        self.khung_anh = tk.Frame(phan, bg='#2d2d2d')

        self.bien_duong_dan_anh = tk.StringVar(value=self.cau_hinh.get("nguon_anh", ""))
        tk.Entry(self.khung_anh, textvariable=self.bien_duong_dan_anh, bg='#3d3d3d', fg='#ffffff',
                 font=('Segoe UI', 8)).pack(side='left', expand=True, fill='x')
        ttk.Button(self.khung_anh, text="...", width=3, command=self.Duyet_File_Anh).pack(side='right',
                                                                                               padx=(3, 0))

    def Tao_Phan_Tai_Du_Lieu(self):
        """Tao phan tai du lieu"""
        phan = tk.LabelFrame(self.khung_cuon, text="TAI DU LIEU",
                                font=('Segoe UI', 9, 'bold'), bg='#2d2d2d', fg='#ffffff', bd=2)
        phan.pack(fill='x', padx=10, pady=5)

        self.nut_tai_du_lieu = ttk.Button(phan, text="TAI DU LIEU", command=self.Tai_Du_Lieu, style='Modern.TButton')
        self.nut_tai_du_lieu.pack(fill='x', padx=8, pady=8)

        self.nhan_trang_thai_tai = tk.Label(phan, text="CHUA TAI DU LIEU", bg='#2d2d2d', fg='#ffaa00',
                                          font=('Segoe UI', 8))
        self.nhan_trang_thai_tai.pack(pady=(0, 8))

    def Tao_Lua_Chon_Vung(self):
        """Tao phan lua chon vung dem"""
        phan = tk.LabelFrame(self.khung_cuon, text="CHON VUNG DEM",
                                font=('Segoe UI', 9, 'bold'), bg='#2d2d2d', fg='#ffffff', bd=2)
        phan.pack(fill='x', padx=10, pady=5)

        khung_nut = tk.Frame(phan, bg='#2d2d2d')
        khung_nut.pack(fill='x', padx=8, pady=8)

        self.nut_chon_vung = ttk.Button(khung_nut, text="CHON VUNG MOI", command=self.Bat_Dau_Chon_Vung,
                                            state='disabled')
        self.nut_chon_vung.pack(side='left', expand=True, fill='x', padx=(0, 3))

        ttk.Button(khung_nut, text="RESET", command=self.Reset_Vung).pack(side='right', expand=True, fill='x',
                                                                            padx=(3, 0))

        ttk.Button(phan, text="HIEN VUNG HIEN TAI", command=self.Hien_Vung_Hien_Tai).pack(fill='x', padx=8,
                                                                                              pady=(0, 5))

        self.bien_hien_toa_do = tk.BooleanVar(value=self.cau_hinh["hien_toa_do"])
        tk.Checkbutton(phan, text="HIEN TOA DO", variable=self.bien_hien_toa_do,
                       bg='#2d2d2d', fg='#ffffff', selectcolor='#4d4d4d',
                       font=('Segoe UI', 8)).pack(anchor='w', padx=8, pady=(0, 8))

    def Tao_Phan_Tham_So(self):
        """Tao phan tham so - them nguong chat luong"""
        phan = tk.LabelFrame(self.khung_cuon, text="THAM SO AI",
                                font=('Segoe UI', 9, 'bold'), bg='#2d2d2d', fg='#ffffff', bd=2)
        phan.pack(fill='x', padx=10, pady=5)

        tk.Label(phan, text="DO TIN CAY:", bg='#2d2d2d', fg='#cccccc',
                 font=('Segoe UI', 8)).pack(anchor='w', padx=8, pady=(8, 0))
        self.bien_tin_cay = tk.DoubleVar(value=self.cau_hinh["nguong_tin_cay"])
        tk.Scale(phan, from_=0.1, to=1.0, resolution=0.05, orient='horizontal', variable=self.bien_tin_cay,
                 bg='#2d2d2d', fg='#ffffff', highlightthickness=0, length=280).pack(fill='x', padx=8, pady=(0, 5))

        tk.Label(phan, text="IOU:", bg='#2d2d2d', fg='#cccccc',
                 font=('Segoe UI', 8)).pack(anchor='w', padx=8)
        self.bien_iou = tk.DoubleVar(value=self.cau_hinh["nguong_iou"])
        tk.Scale(phan, from_=0.1, to=1.0, resolution=0.05, orient='horizontal', variable=self.bien_iou,
                 bg='#2d2d2d', fg='#ffffff', highlightthickness=0, length=280).pack(fill='x', padx=8, pady=(0, 5))

        # Nguong chat luong (Tot/Kem)
        tk.Label(phan, text="NGUONG CHAT LUONG (TOT >= NGUONG >= KEM):", bg='#2d2d2d', fg='#cccccc',
                 font=('Segoe UI', 8)).pack(anchor='w', padx=8)
        self.bien_nguong_chat_luong = tk.DoubleVar(value=self.cau_hinh["nguong_chat_luong"])
        tk.Scale(phan, from_=0.1, to=1.0, resolution=0.05, orient='horizontal', variable=self.bien_nguong_chat_luong,
                 bg='#2d2d2d', fg='#ffffff', highlightthickness=0, length=280).pack(fill='x', padx=8, pady=(0, 8))

    def Tao_Cai_Dat_Xuat_File(self):
        """Tao phan cai dat xuat file"""
        phan = tk.LabelFrame(self.khung_cuon, text="CAI DAT XUAT FILE",
                                font=('Segoe UI', 9, 'bold'), bg='#2d2d2d', fg='#ffffff', bd=2)
        phan.pack(fill='x', padx=10, pady=5)

        # Chon thu muc xuat file
        tk.Label(phan, text="THU MUC XUAT FILE:", bg='#2d2d2d', fg='#cccccc',
                 font=('Segoe UI', 8)).pack(anchor='w', padx=8, pady=(8, 2))

        khung_xuat = tk.Frame(phan, bg='#2d2d2d')
        khung_xuat.pack(fill='x', padx=8, pady=(0, 5))

        self.bien_duong_dan_xuat = tk.StringVar(value=self.cau_hinh.get("duong_dan_xuat", ""))
        tk.Entry(khung_xuat, textvariable=self.bien_duong_dan_xuat, bg='#3d3d3d', fg='#ffffff',
                 font=('Segoe UI', 8)).pack(side='left', expand=True, fill='x')
        ttk.Button(khung_xuat, text="...", width=3, command=self.Duyet_Thu_Muc_Xuat).pack(side='right', padx=(3, 0))

        # Tu dong xuat
        self.bien_tu_dong_xuat = tk.BooleanVar(value=False)
        tk.Checkbutton(phan, text="TU DONG XUAT SAU KHI XU LY", variable=self.bien_tu_dong_xuat,
                       bg='#2d2d2d', fg='#ffffff', selectcolor='#4d4d4d',
                       font=('Segoe UI', 8)).pack(anchor='w', padx=8, pady=(0, 8))

    def Tao_Nut_Dieu_Khien(self):
        """Tao phan nut dieu khien"""
        phan = tk.LabelFrame(self.khung_cuon, text="DIEU KHIEN",
                                font=('Segoe UI', 9, 'bold'), bg='#2d2d2d', fg='#ffffff', bd=2)
        phan.pack(fill='x', padx=10, pady=5)

        ttk.Button(phan, text="TAI MODEL", command=self.Tai_Model, style='Modern.TButton').pack(fill='x', padx=8,
                                                                                                    pady=5)

        khung_nut = tk.Frame(phan, bg='#2d2d2d')
        khung_nut.pack(fill='x', padx=8, pady=5)

        self.nut_bat_dau = ttk.Button(khung_nut, text="BAT DAU", command=self.Bat_Dau_Xu_Ly)
        self.nut_bat_dau.pack(side='left', expand=True, fill='x', padx=(0, 2))

        self.nut_tam_dung = ttk.Button(khung_nut, text="TAM DUNG", command=self.Chuyen_Doi_Tam_Dung)
        self.nut_tam_dung.pack(side='right', expand=True, fill='x', padx=(2, 0))

        self.nut_dung = ttk.Button(phan, text="DUNG", command=self.Dung_Xu_Ly)
        self.nut_dung.pack(fill='x', padx=8, pady=(5, 8))

    def Tao_Phan_Thong_Ke(self):
        """Tao phan thong ke - cap nhat cho 2 trang thai"""
        phan = tk.LabelFrame(self.khung_cuon, text="THONG KE",
                                font=('Segoe UI', 9, 'bold'), bg='#2d2d2d', fg='#ffffff', bd=2)
        phan.pack(fill='x', padx=10, pady=5)

        # Hien thi so luong
        self.nhan_so_luong = ttk.Label(phan, text="0", style='Count.TLabel')
        self.nhan_so_luong.pack(pady=8)

        self.nhan_mo_ta_so_luong = tk.Label(phan, text="TONG SO DOI TUONG", bg='#2d2d2d', fg='#cccccc',
                                         font=('Segoe UI', 8))
        self.nhan_mo_ta_so_luong.pack()

        # Hien thi chat luong (chi 2 trang thai)
        self.khung_chat_luong = tk.Frame(phan, bg='#2d2d2d')

        nhan_chat_luong = ['TOT:', 'KEM:']
        mau_chat_luong = ['#00ff00', '#ff4444']  # Xanh la, Do
        self.bien_chat_luong_list = []

        for i, (noi_dung_nhan, mau_sac) in enumerate(zip(nhan_chat_luong, mau_chat_luong)):
            khung_dong = tk.Frame(self.khung_chat_luong, bg='#2d2d2d')
            khung_dong.pack(fill='x', pady=2)

            tk.Label(khung_dong, text=noi_dung_nhan, bg='#2d2d2d', fg='#cccccc', width=10, anchor='w',
                     font=('Segoe UI', 8)).pack(side='left')

            bien = tk.StringVar(value="0")
            self.bien_chat_luong_list.append(bien)

            # Tao label voi background color de mo phong den nhay
            nhan_chat_luong = tk.Label(khung_dong, textvariable=bien, bg='#2d2d2d',
                                     fg=mau_sac, font=('Segoe UI', 9, 'bold'))
            nhan_chat_luong.pack(side='left')

            # Luu reference de co the lam hieu ung nhap nhay
            if i == 0:
                self.nhan_chat_luong_tot = nhan_chat_luong
            else:
                self.nhan_chat_luong_kem = nhan_chat_luong

        # Thong ke bo sung
        self.nhan_fps = tk.Label(phan, text="FPS: 0", bg='#2d2d2d', fg='#cccccc',
                                  font=('Segoe UI', 8))
        self.nhan_fps.pack(pady=2)

        # Nut xuat bao cao
        khung_xuat = tk.Frame(phan, bg='#2d2d2d')
        khung_xuat.pack(fill='x', padx=8, pady=8)

        ttk.Button(khung_xuat, text="XUAT TXT", command=self.Xuat_Bao_Cao_TXT).pack(side='left', expand=True,
                                                                                       fill='x', padx=(0, 2))
        ttk.Button(khung_xuat, text="XUAT JSON", command=self.Xuat_Bao_Cao).pack(side='right', expand=True, fill='x',
                                                                                    padx=(2, 0))


    def Tao_Phan_Ho_Tro_Khach_Hang(self):
        """Tao phan ho tro khach hang"""
        phan = tk.LabelFrame(self.khung_cuon, text="HO TRO KHACH HANG",
                                font=('Segoe UI', 9, 'bold'), bg='#2d2d2d', fg='#ffffff', bd=2)
        phan.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(phan, text="LIEN HE HO TRO", 
                command=self.Mo_Hop_Thoai_Ho_Tro,
                style='Modern.TButton').pack(fill='x', padx=8, pady=8)
        
        # Thong tin lien he nhanh
        thong_tin_frame = tk.Frame(phan, bg='#2d2d2d')
        thong_tin_frame.pack(fill='x', padx=8, pady=(0, 8))
        
        tk.Label(thong_tin_frame, text="Support@khoaitaychatluongcao.vn", 
                bg='#2d2d2d', fg='#00bfff', 
                font=('Segoe UI', 8)).pack(anchor='w')
        tk.Label(thong_tin_frame, text="Hotline: 1900-090909", 
                bg='#2d2d2d', fg='#00ff88', 
                font=('Segoe UI', 8)).pack(anchor='w')


    def Tao_Phan_Hen_Gio(self):
        """Tao phan hen gio tat ung dung"""
        phan = tk.LabelFrame(self.khung_cuon, text="HEN GIO TAT UNG DUNG",
                                font=('Segoe UI', 9, 'bold'), bg='#2d2d2d', fg='#ffffff', bd=2)
        phan.pack(fill='x', padx=10, pady=5)
        
        # Chon thoi gian
        khung_chon = tk.Frame(phan, bg='#2d2d2d')
        khung_chon.pack(fill='x', padx=8, pady=8)
        
        tk.Label(khung_chon, text="THOI GIAN (PHUT):", 
                bg='#2d2d2d', fg='#cccccc',
                font=('Segoe UI', 8)).pack(side='left', padx=(0, 5))
        
        self.bien_phut_hen_gio = tk.IntVar(value=30)
        tk.Spinbox(khung_chon, from_=1, to=480, width=8, 
                textvariable=self.bien_phut_hen_gio,
                bg='#3d3d3d', fg='#ffffff', 
                font=('Segoe UI', 9)).pack(side='left')
        
        # Nut bat/tat hen gio
        khung_nut = tk.Frame(phan, bg='#2d2d2d')
        khung_nut.pack(fill='x', padx=8, pady=(0, 5))
        
        self.nut_hen_gio = ttk.Button(khung_nut, text="⏰ BAT HEN GIO", 
                                    command=self.Bat_Hen_Gio)
        self.nut_hen_gio.pack(side='left', expand=True, fill='x', padx=(0, 2))
        
        ttk.Button(khung_nut, text="⏹ HUY", 
                command=self.Huy_Hen_Gio).pack(side='right', expand=True, fill='x', padx=(2, 0))
        
        # Hien thi trang thai
        self.nhan_trang_thai_hen_gio = tk.Label(phan, text="CHUA HEN GIO", 
                                                bg='#2d2d2d', fg='#ffaa00',
                                                font=('Segoe UI', 8))
        self.nhan_trang_thai_hen_gio.pack(pady=(0, 8))

    def Mo_Hop_Thoai_Ho_Tro(self):
        """Mo hop thoai ho tro khach hang"""
        if self.cua_so_ho_tro and self.cua_so_ho_tro.winfo_exists():
            self.cua_so_ho_tro.lift()
            return
        
        self.cua_so_ho_tro = tk.Toplevel(self.root)
        self.cua_so_ho_tro.title("HO TRO KHACH HANG")
        self.cua_so_ho_tro.geometry("500x650")
        self.cua_so_ho_tro.configure(bg='#2d2d2d')
        self.cua_so_ho_tro.resizable(False, False)
        
        # Tieu de
        tk.Label(self.cua_so_ho_tro, text="📞 LIEN HE HO TRO KHACH HANG",
                font=('Segoe UI', 16, 'bold'), 
                bg='#2d2d2d', fg='#00ff88').pack(pady=15)
        
        # Khung noi dung
        khung_noi_dung = tk.Frame(self.cua_so_ho_tro, bg='#2d2d2d')
        khung_noi_dung.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Ho va ten
        tk.Label(khung_noi_dung, text="HO VA TEN:", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.nhap_ten = tk.Entry(khung_noi_dung, bg='#3d3d3d', fg='#ffffff',
                                font=('Segoe UI', 10), relief='flat', bd=5)
        self.nhap_ten.pack(fill='x', pady=(0, 15))
        
        # So dien thoai
        tk.Label(khung_noi_dung, text="SO DIEN THOAI:", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.nhap_sdt = tk.Entry(khung_noi_dung, bg='#3d3d3d', fg='#ffffff',
                                font=('Segoe UI', 10), relief='flat', bd=5)
        self.nhap_sdt.pack(fill='x', pady=(0, 15))
        
        # Email
        tk.Label(khung_noi_dung, text="EMAIL:", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.nhap_email = tk.Entry(khung_noi_dung, bg='#3d3d3d', fg='#ffffff',
                                    font=('Segoe UI', 10), relief='flat', bd=5)
        self.nhap_email.pack(fill='x', pady=(0, 15))
        
        # Noi dung gop y
        tk.Label(khung_noi_dung, text="NOI DUNG GOP Y / BAO LOI:", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        
        khung_text = tk.Frame(khung_noi_dung, bg='#3d3d3d', bd=5)
        khung_text.pack(fill='both', expand=True, pady=(0, 10))
        
        self.nhap_gop_y = tk.Text(khung_text, bg='#3d3d3d', fg='#ffffff',
                                font=('Segoe UI', 10), relief='flat',
                                wrap=tk.WORD, height=8)
        self.nhap_gop_y.pack(fill='both', expand=True)
        
        # NUT GUI
        khung_nut_gui = tk.Frame(khung_noi_dung, bg='#2d2d2d')
        khung_nut_gui.pack(fill='x', pady=(10, 0))
        
        self.nut_gui_gop_y = tk.Button(
            khung_nut_gui,
            text="📨 GUI GOP Y",
            command=self.Gui_Gop_Y,
            bg='#00aa44',
            fg='#ffffff',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2',
            activebackground='#00cc55',
            activeforeground='#ffffff'
        )
        self.nut_gui_gop_y.pack(fill='x')
        
        # Hieu ung hover
        def on_enter_gui(e):
            self.nut_gui_gop_y.config(bg='#00cc55')
        
        def on_leave_gui(e):
            self.nut_gui_gop_y.config(bg='#00aa44')
        
        self.nut_gui_gop_y.bind('<Enter>', on_enter_gui)
        self.nut_gui_gop_y.bind('<Leave>', on_leave_gui)
        
        # Nut dong
        khung_nut_dong = tk.Frame(self.cua_so_ho_tro, bg='#2d2d2d')
        khung_nut_dong.pack(fill='x', padx=20, pady=(10, 5))
        
        nut_dong = tk.Button(
            khung_nut_dong,
            text="✖ DONG",
            command=self.cua_so_ho_tro.destroy,
            bg='#666666',
            fg='#ffffff',
            font=('Segoe UI', 9),
            relief='flat',
            bd=0,
            padx=15,
            pady=8,
            cursor='hand2',
            activebackground='#777777',
            activeforeground='#ffffff'
        )
        nut_dong.pack(side='right')
        
        # Hieu ung hover nut dong
        def on_enter_dong(e):
            nut_dong.config(bg='#777777')
        
        def on_leave_dong(e):
            nut_dong.config(bg='#666666')
        
        nut_dong.bind('<Enter>', on_enter_dong)
        nut_dong.bind('<Leave>', on_leave_dong)
        
        # Thong tin lien he
        khung_lien_he = tk.LabelFrame(self.cua_so_ho_tro, text="THONG TIN LIEN HE",
                                        bg='#2d2d2d', fg='#00bfff',
                                        font=('Segoe UI', 9, 'bold'))
        khung_lien_he.pack(fill='x', padx=20, pady=(0, 15))
        
        tk.Label(khung_lien_he, text="📧 Email: support@khoaitaychatluongcao.vn", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 9)).pack(anchor='w', padx=10, pady=3)
        
        tk.Label(khung_lien_he, text="☎ Hotline: 1900-0909", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 9)).pack(anchor='w', padx=10, pady=3)
        
        tk.Label(khung_lien_he, text="🌐 Website: www.khoaitaychatluongcao.vn", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 9)).pack(anchor='w', padx=10, pady=(3, 8))
    
    def Tao_Bang_Video(self, parent):
        """Tao bang hien thi video"""
        bang_phai = tk.Frame(parent, bg='#2d2d2d', relief='raised', bd=2)
        bang_phai.pack(side='right', fill='both', expand=True)

        tk.Label(bang_phai, text="HIEN THI VIDEO & CHON VUNG", font=('Segoe UI', 12, 'bold'),
                 bg='#2d2d2d', fg='#ffffff').pack(pady=10)

        self.nhan_huong_dan = tk.Label(bang_phai, text="VUI LONG TAI DU LIEU TRUOC KHI CHON VUNG DEM",
                                          font=('Segoe UI', 9), bg='#2d2d2d', fg='#ffaa00')
        self.nhan_huong_dan.pack(pady=(0, 8))

        khung_video = tk.Frame(bang_phai, bg='#1e1e1e', relief='sunken', bd=3)
        khung_video.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(khung_video, bg='#000000', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        self.canvas.bind("<Button-1>", self.Xu_ly_Nhan_Chuot_Trai)
        self.canvas.bind("<Motion>", self.Xu_ly_Di_Chuyen_Chuot)
        self.canvas.bind("<Button-3>", self.Xu_ly_Nhan_Chuot_Phai)

        self.Hien_Thong_Bao_Mac_Dinh()

    def Tao_Thanh_Trang_Thai(self, parent):
        """Tao thanh trang thai"""
        khung_trang_thai = tk.Frame(parent, bg='#3d3d3d', height=30)
        khung_trang_thai.pack(fill='x', side='bottom')
        khung_trang_thai.pack_propagate(False)

        self.bien_trang_thai = tk.StringVar(value="SAN SANG - CHON CHE DO VA TAI MODEL DE BAT DAU")
        tk.Label(khung_trang_thai, textvariable=self.bien_trang_thai, bg='#3d3d3d', fg='#ffffff',
                 anchor='w', font=('Segoe UI', 9)).pack(side='left', padx=8, pady=3)

        self.bien_thoi_gian = tk.StringVar()
        tk.Label(khung_trang_thai, textvariable=self.bien_thoi_gian, bg='#3d3d3d', fg='#cccccc',
                 anchor='e', font=('Segoe UI', 9)).pack(side='right', padx=8, pady=3)

    # Xu ly su kien
    def Xu_ly_Thay_Doi_Che_Do(self):
        """Xu ly khi thay doi che do xu ly"""
        self.che_do_xu_ly = self.bien_che_do.get()

        if self.che_do_xu_ly == "chat_luong":
            self.khung_chat_luong.pack(fill='x', pady=8)
            self.nhan_mo_ta_so_luong.config(text="TONG SO DOI TUONG KIEM TRA")
        else:
            self.khung_chat_luong.pack_forget()
            self.nhan_mo_ta_so_luong.config(text="TONG SO KHOAI TAY DA DEM")

        self.bien_trang_thai.set(
            f"DA CHUYEN SANG CHE DO: {'KIEM TRA CHAT LUONG (TOT/KEM)' if self.che_do_xu_ly == 'chat_luong' else 'DEM KHOAI TAY'}")

    def Xu_ly_Thay_Doi_Nguon(self):
        """Xu ly khi thay doi nguon input"""
        nguon = self.bien_nguon.get()
        self.nguon_dau_vao = nguon

        # An tat ca khung cua nguon cu
        self.khung_camera.pack_forget()
        self.nhan_anh.pack_forget()
        self.khung_anh.pack_forget()

        if nguon == "camera":
            self.khung_camera.pack(fill='x', padx=8, pady=(0, 8))
            self.nhan_video.config(text="TAP TIN VIDEO:")
            self.khung_video.pack(fill='x', padx=8, pady=(0, 8))
        elif nguon == "image":
            self.nhan_anh.pack(anchor='w', padx=8, pady=(5, 2))
            self.khung_anh.pack(fill='x', padx=8, pady=(0, 8))
            self.khung_video.pack_forget()
        else:  # file
            self.nhan_video.config(text="TAP TIN VIDEO:")
            self.khung_video.pack(fill='x', padx=8, pady=(0, 8))

        self.da_tai_du_lieu = False
        self.nhan_trang_thai_tai.config(text="CHUA TAI DU LIEU", fg='#ffaa00')
        self.nut_chon_vung.config(state='disabled')
        self.Hien_Thong_Bao_Mac_Dinh()

    def Duyet_Thu_Muc_Xuat(self):
        """Chon thu muc xuat file"""
        thu_muc = filedialog.askdirectory(title="CHON THU MUC LUU FILE BAO CAO")
        if thu_muc:
            self.bien_duong_dan_xuat.set(thu_muc)
            self.cau_hinh["duong_dan_xuat"] = thu_muc

    def Hien_Vung_Hien_Tai(self):
        """Hien thi vung dem hien tai tren canvas"""
        if not self.da_tai_du_lieu or self.khung_hinh_hien_tai is None:
            messagebox.showwarning("CANH BAO", "VUI LONG TAI DU LIEU TRUOC!")
            return

        # Tao frame voi vung da ve
        khung_hinh_co_chu_thich = self.khung_hinh_hien_tai.copy()
        da_giac = np.array(self.cau_hinh["diem_vung"])

        # Ve vung
        cv2.polylines(khung_hinh_co_chu_thich, [da_giac], isClosed=True, color=(0, 255, 255), thickness=3)

        # Ve cac diem
        for i, diem in enumerate(self.cau_hinh["diem_vung"]):
            cv2.circle(khung_hinh_co_chu_thich, tuple(diem), 8, (0, 0, 255), -1)
            cv2.putText(khung_hinh_co_chu_thich, str(i + 1), (diem[0] - 5, diem[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Hien thi toa do
        y_chu = 30
        for i, diem in enumerate(self.cau_hinh["diem_vung"]):
            chu = f"DIEM{i + 1}: ({diem[0]}, {diem[1]})"
            cv2.putText(khung_hinh_co_chu_thich, chu, (10, y_chu),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_chu += 25

        self.Cap_Nhat_Hien_Thi(khung_hinh_co_chu_thich)
        self.nhan_huong_dan.config(text=f"HIEN THI VUNG HIEN TAI VOI {len(self.cau_hinh['diem_vung'])} DIEM")

    def Xu_ly_Nhan_Chuot_Trai(self, event):
        """Xu ly nhan chuot trai tren canvas"""
        if not self.dang_chon_vung:
            return

        # Chuyen doi toa do canvas sang toa do khung hinh
        if hasattr(self, 'kich_thuoc_khung_hinh_hien_thi'):
            chieu_rong_canvas = self.canvas.winfo_width()
            chieu_cao_canvas = self.canvas.winfo_height()
            chieu_rong_khung, chieu_cao_khung = self.kich_thuoc_khung_hinh_hien_thi

            if chieu_rong_canvas / chieu_cao_canvas > chieu_rong_khung / chieu_cao_khung:
                ti_le = chieu_cao_canvas / chieu_cao_khung
                offset_x = (chieu_rong_canvas - chieu_rong_khung * ti_le) / 2
                offset_y = 0
            else:
                ti_le = chieu_rong_canvas / chieu_rong_khung
                offset_x = 0
                offset_y = (chieu_cao_canvas - chieu_cao_khung * ti_le) / 2

            x = int((event.x - offset_x) / ti_le)
            y = int((event.y - offset_y) / ti_le)
            x = max(0, min(x, chieu_rong_khung - 1))
            y = max(0, min(y, chieu_cao_khung - 1))
        else:
            x, y = event.x, event.y

        self.diem_tam_thoi.append([x, y])
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3,
                                fill='red', outline='white', width=2, tags="diem_vung")

        so_diem_can = max(0, 3 - len(self.diem_tam_thoi))
        if so_diem_can > 0:
            self.nhan_huong_dan.config(text=f"DA CHON {len(self.diem_tam_thoi)} DIEM. CON {so_diem_can} DIEM")
        else:
            self.nhan_huong_dan.config(text="DU DIEM! CHUOT PHAI DE HOAN THANH")

    def Xu_ly_Nhan_Chuot_Phai(self, event):
        """Xu ly nhan chuot phai de hoan thanh chon vung"""
        if self.dang_chon_vung and len(self.diem_tam_thoi) >= 3:
            self.Hoan_Thanh_Chon_Vung()

    def Xu_ly_Di_Chuyen_Chuot(self, event):
        """Xu ly di chuyen chuot"""
        if self.dang_chon_vung and len(self.diem_tam_thoi) > 0:
            self.canvas.delete("duong_tam")
            diem_cuoi = self.diem_tam_thoi[-1]

            if hasattr(self, 'kich_thuoc_khung_hinh_hien_thi'):
                chieu_rong_canvas = self.canvas.winfo_width()
                chieu_cao_canvas = self.canvas.winfo_height()
                chieu_rong_khung, chieu_cao_khung = self.kich_thuoc_khung_hinh_hien_thi

                if chieu_rong_canvas / chieu_cao_canvas > chieu_rong_khung / chieu_cao_khung:
                    ti_le = chieu_cao_canvas / chieu_cao_khung
                    offset_x = (chieu_rong_canvas - chieu_rong_khung * ti_le) / 2
                    offset_y = 0
                else:
                    ti_le = chieu_rong_canvas / chieu_rong_khung
                    offset_x = 0
                    offset_y = (chieu_cao_canvas - chieu_cao_khung * ti_le) / 2

                x_cuoi = diem_cuoi[0] * ti_le + offset_x
                y_cuoi = diem_cuoi[1] * ti_le + offset_y
            else:
                x_cuoi, y_cuoi = diem_cuoi

            self.canvas.create_line(x_cuoi, y_cuoi, event.x, event.y,
                                    fill='yellow', width=2, tags="duong_tam")

    # Thao tac voi file
    def Duyet_File_Model(self):
        """Chon file model"""
        ten_file = filedialog.askopenfilename(
            title="CHON FILE MODEL YOLO",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if ten_file:
            self.bien_duong_dan_model.set(ten_file)
            self.cau_hinh["duong_dan_model"] = ten_file

    def Duyet_File_Video(self):
        """Chon file video"""
        ten_file = filedialog.askopenfilename(
            title="CHON FILE VIDEO",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if ten_file:
            self.bien_duong_dan_video.set(ten_file)
            self.cau_hinh["nguon_video"] = ten_file

    def Duyet_File_Anh(self):
        """Chon file anh"""
        ten_file = filedialog.askopenfilename(
            title="CHON FILE ANH",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if ten_file:
            self.bien_duong_dan_anh.set(ten_file)
            self.cau_hinh["nguon_anh"] = ten_file

    # Chuc nang chinh
    def Tai_Du_Lieu(self):
        """Tai du lieu video/camera/anh"""
        try:
            self.bien_trang_thai.set("DANG TAI DU LIEU...")

            if self.nguon_dau_vao == "camera":
                cap_test = cv2.VideoCapture(self.bien_camera.get())
                if not cap_test.isOpened():
                    raise Exception(f"KHONG THE MO CAMERA {self.bien_camera.get()}")
                ret, khung_hinh = cap_test.read()
                if not ret:
                    raise Exception("KHONG THE DOC KHUNG HINH TU CAMERA")
                cap_test.release()
            elif self.nguon_dau_vao == "image":
                duong_dan_anh = self.bien_duong_dan_anh.get()
                if not os.path.exists(duong_dan_anh):
                    raise Exception("FILE ANH KHONG TON TAI!")
                khung_hinh = cv2.imread(duong_dan_anh)
                if khung_hinh is None:
                    raise Exception("KHONG THE DOC FILE ANH")
            else:
                duong_dan_video = self.bien_duong_dan_video.get()
                if not os.path.exists(duong_dan_video):
                    raise Exception("FILE VIDEO KHONG TON TAI!")
                cap_test = cv2.VideoCapture(duong_dan_video)
                if not cap_test.isOpened():
                    raise Exception("KHONG THE MO VIDEO")
                ret, khung_hinh = cap_test.read()
                if not ret:
                    raise Exception("KHONG THE DOC KHUNG HINH TU VIDEO")
                cap_test.release()

            self.khung_hinh_hien_tai = khung_hinh
            self.Cap_Nhat_Hien_Thi(khung_hinh)

            self.da_tai_du_lieu = True
            self.nhan_trang_thai_tai.config(text="DU LIEU DA TAI", fg='#00ff88')
            self.nut_chon_vung.config(state='normal')

            if self.nguon_dau_vao == "camera":
                thong_tin_nguon = f"CAMERA {self.bien_camera.get()}"
            elif self.nguon_dau_vao == "image":
                thong_tin_nguon = os.path.basename(self.bien_duong_dan_anh.get())
            else:
                thong_tin_nguon = os.path.basename(self.bien_duong_dan_video.get())

            self.bien_trang_thai.set(f"DA TAI DU LIEU: {thong_tin_nguon}")
            self.nhan_huong_dan.config(text="NHAN 'CHON VUNG MOI' DE CHON VUNG DEM HOAC 'HIEN VUNG HIEN TAI' DE XEM")

        except Exception as e:
            self.da_tai_du_lieu = False
            self.nhan_trang_thai_tai.config(text="LOI TAI DU LIEU", fg='#ff4444')
            self.nut_chon_vung.config(state='disabled')
            messagebox.showerror("LOI", f"KHONG THE TAI DU LIEU:\n{str(e)}")

    def Bat_Dau_Chon_Vung(self):
        """Bat dau chon vung"""
        if not self.da_tai_du_lieu:
            messagebox.showwarning("CANH BAO", "VUI LONG TAI DU LIEU TRUOC!")
            return

        self.dang_chon_vung = True
        self.diem_tam_thoi = []

        self.canvas.delete("diem_vung")
        self.canvas.delete("duong_tam")

        self.nhan_huong_dan.config(text="NHAN CHUOT TRAI DE CHON DIEM. CHUOT PHAI DE HOAN THANH (TOI THIEU 3 DIEM)")
        self.nut_chon_vung.config(text="HUY CHON", command=self.Huy_Chon_Vung)

    def Huy_Chon_Vung(self):
        """Huy chon vung"""
        self.dang_chon_vung = False
        self.diem_tam_thoi = []

        self.canvas.delete("diem_vung")
        self.canvas.delete("duong_tam")

        self.nhan_huong_dan.config(text="NHAN 'CHON VUNG MOI' DE CHON VUNG DEM")
        self.nut_chon_vung.config(text="CHON VUNG MOI", command=self.Bat_Dau_Chon_Vung)

    def Hoan_Thanh_Chon_Vung(self):
        """Hoan thanh chon vung"""
        if len(self.diem_tam_thoi) < 3:
            messagebox.showwarning("CANH BAO", "CAN IT NHAT 3 DIEM!")
            return

        self.cau_hinh["diem_vung"] = self.diem_tam_thoi.copy()
        self.nhan_huong_dan.config(text="DA CHON VUNG MOI THANH CONG!")

        self.dang_chon_vung = False
        self.nut_chon_vung.config(text="CHON VUNG MOI", command=self.Bat_Dau_Chon_Vung)

        # Hien thi vung moi ngay lap tuc
        self.Hien_Vung_Hien_Tai()

        self.Luu_Cau_Hinh()
        messagebox.showinfo("THANH CONG", f"DA CAP NHAT VUNG DEM VOI {len(self.diem_tam_thoi)} DIEM!")

    def Reset_Vung(self):
        """Reset vung dem ve mac dinh"""
        self.cau_hinh["diem_vung"] = self.diem_vung_goc.copy()

        self.canvas.delete("diem_vung")
        self.canvas.delete("duong_tam")

        if self.dang_chon_vung:
            self.Huy_Chon_Vung()

        # Hien thi vung mac dinh neu co du lieu
        if self.da_tai_du_lieu and self.khung_hinh_hien_tai is not None:
            self.Hien_Vung_Hien_Tai()

        self.Luu_Cau_Hinh()
        self.bien_trang_thai.set("DA RESET VUNG DEM VE MAC DINH")

    def Tai_Model(self):
        """Tai model YOLO"""
        try:
            self.bien_trang_thai.set("DANG TAI MODEL...")
            duong_dan_model = self.bien_duong_dan_model.get()

            if not os.path.exists(duong_dan_model):
                raise FileNotFoundError(f"KHONG TIM THAY FILE MODEL: {duong_dan_model}")

            self.model = YOLO(duong_dan_model)
            self.bien_trang_thai.set("MODEL DA DUOC TAI THANH CONG!")
            messagebox.showinfo("THANH CONG", "MODEL DA DUOC TAI THANH CONG!")

        except Exception as e:
            messagebox.showerror("LOI", f"KHONG THE TAI MODEL:\n{str(e)}")

    def Bat_Dau_Xu_Ly(self):
        """Bat dau xu ly"""
        if self.model is None:
            messagebox.showwarning("CANH BAO", "VUI LONG TAI MODEL TRUOC!")
            return

        if not self.da_tai_du_lieu:
            messagebox.showwarning("CANH BAO", "VUI LONG TAI DU LIEU TRUOC!")
            return

        if self.nguon_dau_vao == "file" and not os.path.exists(self.bien_duong_dan_video.get()):
            messagebox.showerror("LOI", "FILE VIDEO KHONG TON TAI!")
            return

        if self.nguon_dau_vao == "image" and not os.path.exists(self.bien_duong_dan_anh.get()):
            messagebox.showerror("LOI", "FILE ANH KHONG TON TAI!")
            return

        self.Reset_Trang_Thai_Xu_Ly()
        self.Cap_Nhat_Trang_Thai_Nut(dang_xu_ly=True)

        if self.nguon_dau_vao == "image":
            # Xu ly anh truc tiep trong main thread
            self.Xu_Ly_Anh()
        else:
            # Xu ly video/camera trong thread rieng
            luong_xu_ly = threading.Thread(target=self.Xu_Ly_Video, daemon=True)
            luong_xu_ly.start()

    def Xu_Ly_Anh(self):
        """Xu ly anh don"""
        try:
            duong_dan_anh = self.bien_duong_dan_anh.get()
            khung_hinh = cv2.imread(duong_dan_anh)

            if khung_hinh is None:
                raise Exception("KHONG THE DOC FILE ANH")

            # Cap nhat cau hinh
            self.cau_hinh["nguong_tin_cay"] = self.bien_tin_cay.get()
            self.cau_hinh["nguong_iou"] = self.bien_iou.get()
            self.cau_hinh["nguong_chat_luong"] = self.bien_nguong_chat_luong.get()

            # Tao vung
            da_giac = np.array(self.cau_hinh["diem_vung"])
            vung = sv.PolygonZone(polygon=da_giac)

            self.kich_thuoc_khung_hinh_goc = (khung_hinh.shape[1], khung_hinh.shape[0])

            # Xu ly khung hinh
            self.Xu_Ly_Khung_Hinh(khung_hinh, vung, da_giac)
            self.so_khung_hinh_da_xu_ly = 1

            # Cap nhat UI trong main thread
            self.root.after(0, self.Cap_Nhat_UI_Sau_Xu_Ly)

            # Hien thi ket qua sau mot chut delay
            self.root.after(1000, self.Hien_Ket_Qua_Hoan_Thanh)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("LOI", f"LOI XU LY ANH:\n{str(e)}"))
        finally:
            self.root.after(0, self.Dung_Xu_Ly)

    def Cap_Nhat_UI_Sau_Xu_Ly(self):
        """Cap nhat UI sau khi xu ly - duoc goi trong main thread"""
        # Cap nhat so dem
        self.nhan_so_luong.config(text=str(self.tong_so_dem))

        # Cap nhat hien thi chat luong neu can
        if self.che_do_xu_ly == "chat_luong":
            self.Cap_Nhat_Hien_Thi_Chat_Luong()

        # Tao khung hinh co chu thich de hien thi
        if hasattr(self, 'khung_hinh_hien_tai') and self.khung_hinh_hien_tai is not None:
            da_giac = np.array(self.cau_hinh["diem_vung"])
            khung_hinh_co_chu_thich = self.Them_Chu_Thich_Khung_Hinh(self.khung_hinh_hien_tai, self.phat_hien_cuoi, da_giac)
            self.Cap_Nhat_Hien_Thi(khung_hinh_co_chu_thich)

    def Reset_Trang_Thai_Xu_Ly(self):
        """Reset trang thai xu ly"""
        self.dang_xu_ly = True
        self.dang_tam_dung = False
        self.danh_sach_id_da_dem.clear()
        self.tong_so_dem = 0
        self.khoai_tay_da_phat_hien.clear()

        # Reset du lieu chat luong - chi con 2 trang thai
        self.ket_qua_chat_luong = {
            'tot': 0, 'kem': 0,
            'chi_tiet': {}
        }

        self.thoi_gian_bat_dau_xu_ly = time.time()
        self.so_khung_hinh_da_xu_ly = 0

    def Cap_Nhat_Trang_Thai_Nut(self, dang_xu_ly=False):
        """Cap nhat trang thai cac nut"""
        trang_thai = 'disabled' if dang_xu_ly else 'normal'
        trang_thai_nguoc = 'normal' if dang_xu_ly else 'disabled'

        self.nut_bat_dau.config(state=trang_thai)
        self.nut_tai_du_lieu.config(state=trang_thai)

        if self.nguon_dau_vao != "image":
            self.nut_tam_dung.config(state=trang_thai_nguoc)
            self.nut_dung.config(state=trang_thai_nguoc)

    def Chuyen_Doi_Tam_Dung(self):
        """Tam dung/tiep tuc"""
        if self.nguon_dau_vao == "image":
            return

        self.dang_tam_dung = not self.dang_tam_dung
        self.nut_tam_dung.config(text="TIEP TUC" if self.dang_tam_dung else "TAM DUNG")
        self.bien_trang_thai.set("DA TAM DUNG" if self.dang_tam_dung else "DANG XU LY...")

    def Dung_Xu_Ly(self):
        """Dung xu ly"""
        self.dang_xu_ly = False
        self.dang_tam_dung = False
        self.Cap_Nhat_Trang_Thai_Nut(dang_xu_ly=False)
        self.bien_trang_thai.set("DA DUNG XU LY")

    def Xu_Ly_Video(self):
        """Xu ly video/camera"""
        try:
            # Khoi tao bo doc
            if self.nguon_dau_vao == "camera":
                self.cap = cv2.VideoCapture(self.bien_camera.get())
                fps = 30
            else:
                self.cap = cv2.VideoCapture(self.bien_duong_dan_video.get())
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 25

            if not self.cap.isOpened():
                raise Exception("KHONG THE MO NGUON VIDEO")

            # Cap nhat cau hinh
            self.cau_hinh["nguong_tin_cay"] = self.bien_tin_cay.get()
            self.cau_hinh["nguong_iou"] = self.bien_iou.get()
            self.cau_hinh["nguong_chat_luong"] = self.bien_nguong_chat_luong.get()

            # Tao vung
            da_giac = np.array(self.cau_hinh["diem_vung"])
            vung = sv.PolygonZone(polygon=da_giac)

            so_thu_tu_khung_hinh = 0
            thoi_gian_fps_cuoi = time.time()
            dem_fps = 0

            while self.dang_xu_ly:
                if self.dang_tam_dung:
                    time.sleep(0.1)
                    continue

                ret, khung_hinh = self.cap.read()
                if not ret:
                    if self.nguon_dau_vao == "file":
                        self.root.after(0, self.Hien_Ket_Qua_Hoan_Thanh)
                        break
                    else:
                        continue

                self.kich_thuoc_khung_hinh_goc = (khung_hinh.shape[1], khung_hinh.shape[0])

                if so_thu_tu_khung_hinh % self.cau_hinh["bo_qua_khung_hinh"] == 0:
                    self.Xu_Ly_Khung_Hinh(khung_hinh, vung, da_giac)
                    self.so_khung_hinh_da_xu_ly += 1

                so_thu_tu_khung_hinh += 1
                dem_fps += 1

                # Cap nhat FPS
                thoi_gian_hien_tai = time.time()
                if thoi_gian_hien_tai - thoi_gian_fps_cuoi >= 1.0:
                    fps_hien_tai = dem_fps / (thoi_gian_hien_tai - thoi_gian_fps_cuoi)
                    self.root.after(0, lambda f=fps_hien_tai: self.nhan_fps.config(text=f"FPS: {f:.1f}"))
                    dem_fps = 0
                    thoi_gian_fps_cuoi = thoi_gian_hien_tai

                time.sleep(max(0, 1 / fps - 0.001))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("LOI", f"LOI XU LY:\n{str(e)}"))
        finally:
            if self.cap:
                self.cap.release()
            self.root.after(0, self.Dung_Xu_Ly)

    def Xu_Ly_Khung_Hinh(self, khung_hinh, vung, da_giac):
        """Xu ly tung khung hinh"""
        try:
            chieu_cao_goc, chieu_rong_goc = khung_hinh.shape[:2]
            kich_thuoc_muc_tieu = self.cau_hinh["kich_thuoc_muc_tieu"]
            he_so_ti_le_w = chieu_rong_goc / kich_thuoc_muc_tieu[0]
            he_so_ti_le_h = chieu_cao_goc / kich_thuoc_muc_tieu[1]

            khung_hinh_da_resize = cv2.resize(khung_hinh, kich_thuoc_muc_tieu)

            # YOLO phat hien
            ket_qua = self.model.predict(
                khung_hinh_da_resize,
                conf=self.cau_hinh["nguong_tin_cay"],
                iou=self.cau_hinh["nguong_iou"],
                verbose=False
            )

            cac_phat_hien = self.Chuyen_Doi_Ket_Qua_YOLO(ket_qua, he_so_ti_le_w, he_so_ti_le_h)

            # Theo doi
            cac_phat_hien_da_theo_doi = self.bo_theo_doi.update_with_detections(cac_phat_hien)
            mat_na = vung.trigger(detections=cac_phat_hien_da_theo_doi)
            cac_phat_hien_trong_vung = cac_phat_hien_da_theo_doi[mat_na]

            # Xu ly theo che do
            if self.che_do_xu_ly == "dem":
                self.Xu_Ly_Che_Do_Dem(cac_phat_hien_trong_vung)
            else:
                self.Xu_Ly_Kiem_Chat_Luong(cac_phat_hien_trong_vung)

            # Luu phat hien cuoi de hien thi
            self.phat_hien_cuoi = cac_phat_hien_trong_vung

            # Them chu thich va cap nhat hien thi
            khung_hinh_co_chu_thich = self.Them_Chu_Thich_Khung_Hinh(khung_hinh, cac_phat_hien_trong_vung, da_giac)
            self.root.after(0, lambda: self.Cap_Nhat_Hien_Thi(khung_hinh_co_chu_thich))

            # Cap nhat bo dem UI
            self.root.after(0, lambda: self.nhan_so_luong.config(text=str(self.tong_so_dem)))
            if self.che_do_xu_ly == "chat_luong":
                self.root.after(0, self.Cap_Nhat_Hien_Thi_Chat_Luong)

        except Exception as e:
            print(f"LOI XU LY KHUNG HINH: {e}")

    def Xu_Ly_Che_Do_Dem(self, cac_phat_hien):
        """Xu ly che do dem"""
        if len(cac_phat_hien) > 0:
            for id_theo_doi in cac_phat_hien.tracker_id:
                self.danh_sach_id_da_dem.add(id_theo_doi)

        self.tong_so_dem = len(self.danh_sach_id_da_dem)
        self.Cap_Nhat_Toa_Do_Khoai_Tay(cac_phat_hien)

    def Xu_Ly_Kiem_Chat_Luong(self, cac_phat_hien):
        """Xu ly che do kiem tra chat luong - chi 2 trang thai: Tot/Kem"""
        if len(cac_phat_hien) == 0:
            return

        for i, id_theo_doi in enumerate(cac_phat_hien.tracker_id):
            if id_theo_doi not in self.ket_qua_chat_luong['chi_tiet']:
                # Danh gia chat luong dua tren confidence va cac yeu to khac
                do_tin_cay = cac_phat_hien.confidence[i] if len(cac_phat_hien.confidence) > i else 0.5

                # Lay thong tin bounding box de phan tich them
                bbox = cac_phat_hien.xyxy[i] if len(cac_phat_hien.xyxy) > i else None

                # Tinh toan diem chat luong dua tren nhieu yeu to
                diem_chat_luong = self.Tinh_Diem_Chat_Luong(do_tin_cay, bbox)

                # Chuyen doi diem_chat_luong thanh phan tram
                phan_tram_chat_luong = diem_chat_luong * 100

                # Phan loai dua tren nguong 70%
                if phan_tram_chat_luong >= 70.0:
                    chat_luong = 'tot'
                    self.Kich_Hoat_Chi_Bao_Chat_Luong('tot')
                else:
                    chat_luong = 'kem'
                    self.Kich_Hoat_Chi_Bao_Chat_Luong('kem')

                self.ket_qua_chat_luong['chi_tiet'][id_theo_doi] = {
                    'chat_luong': chat_luong,
                    'do_tin_cay': do_tin_cay,
                    'diem_chat_luong': diem_chat_luong,
                    'phan_tram_chat_luong': phan_tram_chat_luong,
                    'thoi_gian': time.time()
                }

                self.ket_qua_chat_luong[chat_luong] += 1

        self.tong_so_dem = len(self.ket_qua_chat_luong['chi_tiet'])
        self.Cap_Nhat_Toa_Do_Khoai_Tay(cac_phat_hien)

    def Tinh_Diem_Chat_Luong(self, do_tin_cay, bbox):
        """Tinh toan diem chat luong dua tren nhieu yeu to"""
        diem_chat_luong = do_tin_cay  # Diem co ban tu confidence

        if bbox is not None:
            # Them cac yeu to phan tich khac
            x1, y1, x2, y2 = bbox
            chieu_rong = x2 - x1
            chieu_cao = y2 - y1
            dien_tich = chieu_rong * chieu_cao
            ti_le_khung_hinh = chieu_rong / max(chieu_cao, 1)

            # Dieu chinh diem so dua tren kich thuoc (khoai tay qua nho hoac qua lon co the kem chat luong)
            if dien_tich > 1000 and dien_tich < 50000:  # Kich thuoc hop ly
                diem_chat_luong += 0.1
            else:
                diem_chat_luong -= 0.1

            # Dieu chinh dua tren ti le khung hinh (khoai tay co ti le hop ly)
            if 0.5 < ti_le_khung_hinh < 2.0:  # Ti le hop ly
                diem_chat_luong += 0.05
            else:
                diem_chat_luong -= 0.05

        return min(1.0, max(0.0, diem_chat_luong))  # Gioi han trong khoang [0, 1]

    def Kich_Hoat_Chi_Bao_Chat_Luong(self, chat_luong):
        """Kich hoat hieu ung den nhay cho chat luong"""

        def Nhay_Chi_Bao():
            if chat_luong == 'tot' and hasattr(self, 'nhan_chat_luong_tot'):
                # Hieu ung nhay xanh
                mau_goc = self.nhan_chat_luong_tot.cget('bg')
                self.nhan_chat_luong_tot.config(bg='#00ff00')
                self.root.after(200, lambda: self.nhan_chat_luong_tot.config(bg=mau_goc))
            elif chat_luong == 'kem' and hasattr(self, 'nhan_chat_luong_kem'):
                # Hieu ung nhay do
                mau_goc = self.nhan_chat_luong_kem.cget('bg')
                self.nhan_chat_luong_kem.config(bg='#ff0000')
                self.root.after(200, lambda: self.nhan_chat_luong_kem.config(bg=mau_goc))

        self.root.after(0, Nhay_Chi_Bao)

    def Cap_Nhat_Hien_Thi_Chat_Luong(self):
        """Cap nhat hien thi chat luong - chi 2 loai"""
        cac_chat_luong = ['tot', 'kem']
        for i, chat_luong in enumerate(cac_chat_luong):
            if i < len(self.bien_chat_luong_list):
                self.bien_chat_luong_list[i].set(str(self.ket_qua_chat_luong[chat_luong]))

    def Cap_Nhat_Toa_Do_Khoai_Tay(self, cac_phat_hien):
        """Cap nhat toa do doi tuong"""
        if not self.cau_hinh["hien_toa_do"]:
            return

        if len(cac_phat_hien) > 0:
            for i, id_theo_doi in enumerate(cac_phat_hien.tracker_id):
                if i < len(cac_phat_hien.xyxy):
                    xyxy = cac_phat_hien.xyxy[i]
                    x1, y1, x2, y2 = xyxy
                    tam_x = int((x1 + x2) / 2)
                    tam_y = int((y1 + y2) / 2)

                    self.khoai_tay_da_phat_hien[id_theo_doi] = {
                        'tam': (tam_x, tam_y),
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'thoi_gian': time.time()
                    }

    def Them_Chu_Thich_Khung_Hinh(self, khung_hinh, cac_phat_hien, da_giac):
        khung_hinh_co_chu_thich = khung_hinh.copy()

        # Ve vung
        cv2.polylines(khung_hinh_co_chu_thich, [da_giac], isClosed=True, color=(0, 255, 255), thickness=3)

        # Ve cac phat hien voi mau sac theo chat luong
        if len(cac_phat_hien) > 0:
            if self.che_do_xu_ly == "chat_luong":
                # Phan loai cac phat hien theo chat luong
                cac_phat_hien_tot = []
                cac_phat_hien_kem = []
                cac_phat_hien_chua_xac_dinh = []
                
                chi_so_tot = []
                chi_so_kem = []
                chi_so_chua_xac_dinh = []
                
                for i, id_theo_doi in enumerate(cac_phat_hien.tracker_id):
                    if id_theo_doi in self.ket_qua_chat_luong['chi_tiet']:
                        chat_luong = self.ket_qua_chat_luong['chi_tiet'][id_theo_doi]['chat_luong']
                        if chat_luong == 'tot':
                            chi_so_tot.append(i)
                        else:
                            chi_so_kem.append(i)
                    else:
                        chi_so_chua_xac_dinh.append(i)
                
                # Ve tung loai voi mau tuong ung
                if chi_so_tot:
                    phat_hien_tot = cac_phat_hien[chi_so_tot]
                    khung_hinh_co_chu_thich = self.bo_ve_khung_tot.annotate(
                        scene=khung_hinh_co_chu_thich, 
                        detections=phat_hien_tot
                    )
                
                if chi_so_kem:
                    phat_hien_kem = cac_phat_hien[chi_so_kem]
                    khung_hinh_co_chu_thich = self.bo_ve_khung_kem.annotate(
                        scene=khung_hinh_co_chu_thich, 
                        detections=phat_hien_kem
                    )
                
                if chi_so_chua_xac_dinh:
                    phat_hien_chua_xac_dinh = cac_phat_hien[chi_so_chua_xac_dinh]
                    khung_hinh_co_chu_thich = self.bo_ve_khung_mac_dinh.annotate(
                        scene=khung_hinh_co_chu_thich, 
                        detections=phat_hien_chua_xac_dinh
                    )
            else:
                # Che do dem: su dung mau mac dinh
                khung_hinh_co_chu_thich = self.bo_ve_khung_mac_dinh.annotate(
                    scene=khung_hinh_co_chu_thich, 
                    detections=cac_phat_hien
                )

            # Them nhan voi mau sac tuong ung
            cac_nhan = []
            for i, id_theo_doi in enumerate(cac_phat_hien.tracker_id):
                if self.che_do_xu_ly == "chat_luong" and id_theo_doi in self.ket_qua_chat_luong['chi_tiet']:
                    thong_tin_chat_luong = self.ket_qua_chat_luong['chi_tiet'][id_theo_doi]
                    chat_luong = thong_tin_chat_luong['chat_luong']
                    do_tin_cay = thong_tin_chat_luong.get('do_tin_cay', 0.0)
                    phan_tram = thong_tin_chat_luong.get('phan_tram_chat_luong', do_tin_cay * 100)
                    chu_chat_luong = "TOT" if chat_luong == 'tot' else "KEM"
                    cac_nhan.append(f"ID:{id_theo_doi} {chu_chat_luong} ({phan_tram:.1f}%)")
                else:
                    if self.cau_hinh["hien_toa_do"] and id_theo_doi in self.khoai_tay_da_phat_hien:
                        tam = self.khoai_tay_da_phat_hien[id_theo_doi]['tam']
                        cac_nhan.append(f"ID:{id_theo_doi} ({tam[0]},{tam[1]})")
                    else:
                        cac_nhan.append(f"ID:{id_theo_doi}")

            khung_hinh_co_chu_thich = self.bo_ve_nhan.annotate(
                scene=khung_hinh_co_chu_thich, 
                detections=cac_phat_hien, 
                labels=cac_nhan
            )

        # Them lop thong ke
        self.Them_Lop_Thong_Ke(khung_hinh_co_chu_thich)

        return khung_hinh_co_chu_thich    

    def Them_Lop_Thong_Ke(self, khung_hinh):
        """Them thong tin thong ke len khung hinh"""
        lop_phu = khung_hinh.copy()

        # Nen ban trong suot
        cv2.rectangle(lop_phu, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(lop_phu, 0.7, khung_hinh, 0.3, 0, khung_hinh)

        # Thong tin che do
        chu_che_do = "KIEM_TRA_CHAT_LUONG" if self.che_do_xu_ly == "chat_luong" else "DEM_KHOAI_TAY"
        cv2.putText(khung_hinh, chu_che_do, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # So dem
        cv2.putText(khung_hinh, f"TONG SO: {self.tong_so_dem}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Thong ke chat luong neu o che do chat luong
        if self.che_do_xu_ly == "chat_luong":
            so_tot = self.ket_qua_chat_luong['tot']
            so_kem = self.ket_qua_chat_luong['kem']

            cv2.putText(khung_hinh, f"TOT: {so_tot}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(khung_hinh, f"KEM: {so_kem}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Ti le chat luong
            if self.tong_so_dem > 0:
                ti_le_tot = (so_tot / self.tong_so_dem) * 100
                cv2.putText(khung_hinh, f"TI LE TOT: {ti_le_tot:.1f}%", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

    def Chuyen_Doi_Ket_Qua_YOLO(self, ket_qua, he_so_ti_le_w, he_so_ti_le_h):
        """Chuyen doi ket qua YOLO thanh format supervision"""
        if not ket_qua or len(ket_qua) == 0:
            return sv.Detections.empty()

        ket_qua_dau = ket_qua[0]
        if ket_qua_dau.boxes is None or len(ket_qua_dau.boxes) == 0:
            return sv.Detections.empty()

        # Ti le toa do ve kich thuoc goc
        cac_hop = ket_qua_dau.boxes.xyxy.cpu().numpy()
        cac_hop[:, [0, 2]] *= he_so_ti_le_w
        cac_hop[:, [1, 3]] *= he_so_ti_le_h

        cac_do_tin_cay = ket_qua_dau.boxes.conf.cpu().numpy() if ket_qua_dau.boxes.conf is not None else []
        cac_id_lop = ket_qua_dau.boxes.cls.cpu().numpy().astype(int) if ket_qua_dau.boxes.cls is not None else []

        return sv.Detections(
            xyxy=cac_hop,
            confidence=cac_do_tin_cay,
            class_id=cac_id_lop
        )

    def Hien_Ket_Qua_Hoan_Thanh(self):
        """Hien thi ket qua sau khi hoan thanh xu ly"""
        if self.so_khung_hinh_da_xu_ly == 0:
            return

        thoi_gian_xu_ly = time.time() - self.thoi_gian_bat_dau_xu_ly if self.thoi_gian_bat_dau_xu_ly > 0 else 0

        # Cap nhat UI cuoi cung
        self.nhan_so_luong.config(text=str(self.tong_so_dem))
        if self.che_do_xu_ly == "chat_luong":
            self.Cap_Nhat_Hien_Thi_Chat_Luong()

        # Tao bao cao chi tiet
        bao_cao = self.Tao_Bao_Cao_Chi_Tiet(thoi_gian_xu_ly)

        # Hien thi popup ket qua
        cua_so_ket_qua = tk.Toplevel(self.root)
        cua_so_ket_qua.title("KET QUA XU LY")
        cua_so_ket_qua.geometry("500x400")
        cua_so_ket_qua.configure(bg='#2d2d2d')
        cua_so_ket_qua.grab_set()

        # Tieu de
        tk.Label(cua_so_ket_qua, text="KET QUA XU LY HOAN TAT",
                 font=('Segoe UI', 14, 'bold'), bg='#2d2d2d', fg='#ffffff').pack(pady=10)

        # Noi dung ket qua
        widget_chu = tk.Text(cua_so_ket_qua, wrap=tk.WORD, bg='#3d3d3d', fg='#ffffff',
                              font=('Segoe UI', 10), height=15)
        widget_chu.pack(fill='both', expand=True, padx=10, pady=10)

        widget_chu.insert('1.0', bao_cao)
        widget_chu.config(state='disabled')

        # Cac nut
        khung_nut = tk.Frame(cua_so_ket_qua, bg='#2d2d2d')
        khung_nut.pack(fill='x', pady=10)

        ttk.Button(khung_nut, text="XUAT JSON",
                   command=lambda: self.Xuat_Bao_Cao(hien_hop_thoai=True)).pack(side='left', padx=(10, 5))
        ttk.Button(khung_nut, text="XUAT TXT",
                   command=lambda: self.Xuat_Bao_Cao_TXT(hien_hop_thoai=True)).pack(side='left', padx=5)
        ttk.Button(khung_nut, text="DONG",
                   command=cua_so_ket_qua.destroy).pack(side='right', padx=(5, 10))

        # Tu dong xuat neu kich hoat
        if self.bien_tu_dong_xuat.get() and self.bien_duong_dan_xuat.get():
            self.Tu_Dong_Xuat_Ket_Qua()

    def Tao_Bao_Cao_Chi_Tiet(self, thoi_gian_xu_ly):
        """Tao bao cao chi tiet"""
        dong_bao_cao = []
        dong_bao_cao.append("=" * 50)
        dong_bao_cao.append("BAO CAO PHAN TICH KHOAI TAY")
        dong_bao_cao.append("=" * 50)
        dong_bao_cao.append(f"THOI GIAN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        dong_bao_cao.append("")

        # Thong tin nguon
        if self.nguon_dau_vao == "camera":
            thong_tin_nguon = f"CAMERA {self.bien_camera.get()}"
        elif self.nguon_dau_vao == "image":
            thong_tin_nguon = f"ANH: {os.path.basename(self.bien_duong_dan_anh.get())}"
        else:
            thong_tin_nguon = f"VIDEO: {os.path.basename(self.bien_duong_dan_video.get())}"

        dong_bao_cao.append(f"NGUON DU LIEU: {thong_tin_nguon}")
        dong_bao_cao.append(
            f"CHE DO XU LY: {'KIEM TRA CHAT LUONG' if self.che_do_xu_ly == 'chat_luong' else 'DEM KHOAI TAY'}")
        dong_bao_cao.append("")

        # Thong ke xu ly
        dong_bao_cao.append("THONG KE XU LY:")
        dong_bao_cao.append(f"- TONG SO DOI TUONG: {self.tong_so_dem}")
        dong_bao_cao.append(f"- KHUNG HINH DA XU LY: {self.so_khung_hinh_da_xu_ly}")
        dong_bao_cao.append(f"- THOI GIAN XU LY: {thoi_gian_xu_ly:.2f} GIAY")

        if self.so_khung_hinh_da_xu_ly > 0 and thoi_gian_xu_ly > 0:
            fps_trung_binh = self.so_khung_hinh_da_xu_ly / thoi_gian_xu_ly
            dong_bao_cao.append(f"- FPS TRUNG BINH: {fps_trung_binh:.2f}")

        dong_bao_cao.append("")

        # Phan tich chat luong
        if self.che_do_xu_ly == "chat_luong":
            so_tot = self.ket_qua_chat_luong['tot']
            so_kem = self.ket_qua_chat_luong['kem']

            dong_bao_cao.append("PHAN TICH CHAT LUONG:")
            dong_bao_cao.append(f"- CHAT LUONG TOT: {so_tot} ({so_tot / max(self.tong_so_dem, 1) * 100:.1f}%)")
            dong_bao_cao.append(f"- CHAT LUONG KEM: {so_kem} ({so_kem / max(self.tong_so_dem, 1) * 100:.1f}%)")
            dong_bao_cao.append("")

            # Thong tin nguong chat luong
            dong_bao_cao.append("NGUONG CHAT LUONG: 70%")
            dong_bao_cao.append("(CONFIDENCE >= 70% = TOT, < 70% = KEM)")
            dong_bao_cao.append("")

            # De xuat chat luong
            if self.tong_so_dem > 0:
                ti_le_tot = so_tot / self.tong_so_dem
                if ti_le_tot >= 0.8:
                    dong_bao_cao.append("✓ DANH GIA: CHAT LUONG TONG THE TOT")
                else:
                    dong_bao_cao.append("✗ DANH GIA: CHAT LUONG TONG THE KEM")
                dong_bao_cao.append("")

        # Tham so ky thuat
        dong_bao_cao.append("THAM SO KY THUAT:")
        dong_bao_cao.append(f"- MODEL: {os.path.basename(self.bien_duong_dan_model.get())}")
        dong_bao_cao.append(f"- NGUONG TIN CAY: {self.cau_hinh['nguong_tin_cay']:.2f}")
        dong_bao_cao.append(f"- NGUONG IOU: {self.cau_hinh['nguong_iou']:.2f}")
        dong_bao_cao.append(f"- VUNG DEM: {len(self.cau_hinh['diem_vung'])} DIEM")

        if hasattr(self, 'kich_thuoc_khung_hinh_goc') and self.kich_thuoc_khung_hinh_goc:
            w, h = self.kich_thuoc_khung_hinh_goc
            dong_bao_cao.append(f"- KICH THUOC KHUNG HINH: {w}x{h}")

        return "\n".join(dong_bao_cao)

    def Tu_Dong_Xuat_Ket_Qua(self):
        """Tu dong xuat ket qua"""
        if not self.bien_duong_dan_xuat.get():
            return

        try:
            dau_thoi_gian = datetime.now().strftime("%Y%m%d_%H%M%S")
            che_do = "chat_luong" if self.che_do_xu_ly == "chat_luong" else "dem"

            # Xuat JSON
            ten_file_json = f"phan_tich_khoai_tay_{che_do}_{dau_thoi_gian}.json"
            duong_dan_json = os.path.join(self.bien_duong_dan_xuat.get(), ten_file_json)
            self.Xuat_Bao_Cao_JSON(duong_dan_json)

            # Xuat TXT
            ten_file_txt = f"bao_cao_khoai_tay_{che_do}_{dau_thoi_gian}.txt"
            duong_dan_txt = os.path.join(self.bien_duong_dan_xuat.get(), ten_file_txt)
            self.Xuat_Bao_Cao_TXT_File(duong_dan_txt)

            self.bien_trang_thai.set(f"DA TU DONG XUAT: {ten_file_json}, {ten_file_txt}")

        except Exception as e:
            print(f"LOI TU DONG XUAT FILE: {e}")

    def Xuat_Bao_Cao(self, hien_hop_thoai=False):
        """Xuat bao cao JSON"""
        try:
            if hien_hop_thoai or not self.bien_duong_dan_xuat.get():
                dau_thoi_gian = datetime.now().strftime("%Y%m%d_%H%M%S")
                che_do = "chat_luong" if self.che_do_xu_ly == "chat_luong" else "dem"

                duong_dan_file = filedialog.asksaveasfilename(
                    title="LUU BAO CAO JSON",
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
                )

                if duong_dan_file and not duong_dan_file.endswith('.json'):
                    duong_dan_file += f"_phan_tich_khoai_tay_{che_do}_{dau_thoi_gian}.json"

            else:
                dau_thoi_gian = datetime.now().strftime("%Y%m%d_%H%M%S")
                che_do = "chat_luong" if self.che_do_xu_ly == "chat_luong" else "dem"
                ten_file = f"phan_tich_khoai_tay_{che_do}_{dau_thoi_gian}.json"
                duong_dan_file = os.path.join(self.bien_duong_dan_xuat.get(), ten_file)

            if duong_dan_file:
                self.Xuat_Bao_Cao_JSON(duong_dan_file)
                if hien_hop_thoai:
                    messagebox.showinfo("THANH CONG", f"DA XUAT BAO CAO: {os.path.basename(duong_dan_file)}")

        except Exception as e:
            messagebox.showerror("LOI", f"KHONG THE XUAT BAO CAO JSON:\n{str(e)}")

    def Xuat_Bao_Cao_JSON(self, duong_dan_file):
        """Xuat bao cao JSON chi tiet voi thong tin phan tram"""
        thoi_gian_xu_ly = time.time() - self.thoi_gian_bat_dau_xu_ly if self.thoi_gian_bat_dau_xu_ly > 0 else 0

        du_lieu_bao_cao = {
            "metadata": {
                "thoi_gian_xuat": datetime.now().isoformat(),
                "phan_mem": "HE THONG DEM KHOAI TAY & KIEM TRA CHAT LUONG PRO",
                "phien_ban": "2.1 DA SUA LOGIC CHAT LUONG",
                "che_do_xu_ly": "kiem_tra_chat_luong" if self.che_do_xu_ly == "chat_luong" else "dem_so_luong"
            },
            "thong_tin_nguon": {
                "loai_dau_vao": self.nguon_dau_vao,
                "duong_dan_nguon": self.Lay_Duong_Dan_Nguon_Hien_Tai(),
                "kich_thuoc_khung_hinh": self.kich_thuoc_khung_hinh_goc if hasattr(self, 'kich_thuoc_khung_hinh_goc') else None
            },
            "tham_so_xu_ly": {
                "duong_dan_model": self.bien_duong_dan_model.get(),
                "nguong_tin_cay": self.cau_hinh["nguong_tin_cay"],
                "nguong_iou": self.cau_hinh["nguong_iou"],
                "nguong_chat_luong_phan_tram": 70.0,  # Co dinh 70%
                "diem_vung": self.cau_hinh["diem_vung"],
                "bo_qua_khung_hinh": self.cau_hinh["bo_qua_khung_hinh"]
            },
            "ket_qua": {
                "tong_so_dem": self.tong_so_dem,
                "khung_hinh_da_xu_ly": self.so_khung_hinh_da_xu_ly,
                "thoi_gian_xu_ly_giay": round(thoi_gian_xu_ly, 2),
                "fps_trung_binh": round(self.so_khung_hinh_da_xu_ly / max(thoi_gian_xu_ly, 0.001), 2) if thoi_gian_xu_ly > 0 else 0
            }
        }

        # Them ket qua theo che do
        if self.che_do_xu_ly == "chat_luong":
            # Tinh phan tram chat luong trung binh
            phan_tram_chat_luong_trung_binh = 0
            if self.ket_qua_chat_luong['chi_tiet']:
                tong_phan_tram = 0
                for chi_tiet in self.ket_qua_chat_luong['chi_tiet'].values():
                    if 'phan_tram_chat_luong' in chi_tiet:
                        tong_phan_tram += chi_tiet['phan_tram_chat_luong']
                    else:
                        phan_tram = chi_tiet.get('diem_chat_luong', chi_tiet.get('do_tin_cay', 0.5)) * 100
                        tong_phan_tram += phan_tram

                phan_tram_chat_luong_trung_binh = tong_phan_tram / len(self.ket_qua_chat_luong['chi_tiet'])

            du_lieu_bao_cao["phan_tich_chat_luong"] = {
                "tom_tat": {
                    "chat_luong_tot": self.ket_qua_chat_luong['tot'],
                    "chat_luong_kem": self.ket_qua_chat_luong['kem'],
                    "ti_le_tot_phan_tram": round((self.ket_qua_chat_luong['tot'] / max(self.tong_so_dem, 1)) * 100, 1),
                    "ti_le_kem_phan_tram": round((self.ket_qua_chat_luong['kem'] / max(self.tong_so_dem, 1)) * 100, 1),
                    "phan_tram_chat_luong_trung_binh": round(phan_tram_chat_luong_trung_binh, 1)
                },
                "phuong_phap_phan_loai": {
                    "nguong_phan_tram": 70.0,
                    "tieu_chi": "PHAN TRAM CHAT LUONG = TIN CAY × 100, >=70% = TOT, <70% = KEM"
                },
                "ket_qua_chi_tiet": []
            }

            # Them ket qua chat luong chi tiet voi phan tram
            for id_theo_doi, chi_tiet in self.ket_qua_chat_luong['chi_tiet'].items():
                if 'phan_tram_chat_luong' not in chi_tiet:
                    chi_tiet['phan_tram_chat_luong'] = chi_tiet.get('diem_chat_luong', chi_tiet.get('do_tin_cay', 0.5)) * 100

                muc_chi_tiet = {
                    "id_theo_doi": int(id_theo_doi),
                    "chat_luong": chi_tiet['chat_luong'],
                    "do_tin_cay": round(chi_tiet['do_tin_cay'], 3),
                    "phan_tram_chat_luong": round(chi_tiet['phan_tram_chat_luong'], 1),
                    "thoi_gian": chi_tiet['thoi_gian']
                }

                # Them toa do neu co
                if id_theo_doi in self.khoai_tay_da_phat_hien:
                    thong_tin_toa_do = self.khoai_tay_da_phat_hien[id_theo_doi]
                    muc_chi_tiet["toa_do"] = {
                        "tam": thong_tin_toa_do['tam'],
                        "bbox": thong_tin_toa_do['bbox']
                    }

                du_lieu_bao_cao["phan_tich_chat_luong"]["ket_qua_chi_tiet"].append(muc_chi_tiet)

        # Them chi tiet phat hien neu co
        if self.khoai_tay_da_phat_hien:
            du_lieu_bao_cao["chi_tiet_phat_hien"] = []
            for id_theo_doi, thong_tin_khoai_tay in self.khoai_tay_da_phat_hien.items():
                chi_tiet = {
                    "id_theo_doi": int(id_theo_doi),
                    "toa_do_tam": thong_tin_khoai_tay['tam'],
                    "hop_bao": thong_tin_khoai_tay['bbox'],
                    "thoi_gian_phat_hien_cuoi": thong_tin_khoai_tay['thoi_gian']
                }

                # Them thong tin chat luong neu o che do chat luong
                if self.che_do_xu_ly == "chat_luong" and id_theo_doi in self.ket_qua_chat_luong['chi_tiet']:
                    thong_tin_chat_luong = self.ket_qua_chat_luong['chi_tiet'][id_theo_doi]
                    if 'phan_tram_chat_luong' not in thong_tin_chat_luong:
                        thong_tin_chat_luong['phan_tram_chat_luong'] = thong_tin_chat_luong.get('diem_chat_luong',
                                                                              thong_tin_chat_luong.get('do_tin_cay', 0.5)) * 100

                    chi_tiet["thong_tin_chat_luong"] = {
                        "chat_luong": thong_tin_chat_luong['chat_luong'],
                        "phan_tram_chat_luong": round(thong_tin_chat_luong['phan_tram_chat_luong'], 1)
                    }

                du_lieu_bao_cao["chi_tiet_phat_hien"].append(chi_tiet)

        # Luu JSON
        with open(duong_dan_file, 'w', encoding='utf-8') as f:
            json.dump(du_lieu_bao_cao, f, indent=2, ensure_ascii=False)

    def Xuat_Bao_Cao_TXT(self, hien_hop_thoai=False):
        """Xuat bao cao TXT"""
        try:
            if hien_hop_thoai or not self.bien_duong_dan_xuat.get():
                dau_thoi_gian = datetime.now().strftime("%Y%m%d_%H%M%S")
                che_do = "chat_luong" if self.che_do_xu_ly == "chat_luong" else "dem"

                duong_dan_file = filedialog.asksaveasfilename(
                    title="LUU BAO CAO TXT",
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                )

                if duong_dan_file and not duong_dan_file.endswith('.txt'):
                    duong_dan_file += f"_bao_cao_khoai_tay_{che_do}_{dau_thoi_gian}.txt"

            else:
                dau_thoi_gian = datetime.now().strftime("%Y%m%d_%H%M%S")
                che_do = "chat_luong" if self.che_do_xu_ly == "chat_luong" else "dem"
                ten_file = f"bao_cao_khoai_tay_{che_do}_{dau_thoi_gian}.txt"
                duong_dan_file = os.path.join(self.bien_duong_dan_xuat.get(), ten_file)

            if duong_dan_file:
                self.Xuat_Bao_Cao_TXT_File(duong_dan_file)
                if hien_hop_thoai:
                    messagebox.showinfo("THANH CONG", f"DA XUAT BAO CAO: {os.path.basename(duong_dan_file)}")

        except Exception as e:
            messagebox.showerror("LOI", f"KHONG THE XUAT BAO CAO TXT:\n{str(e)}")

    def Xuat_Bao_Cao_TXT_File(self, duong_dan_file):
        """Xuat bao cao TXT chi tiet voi thong tin phan tram"""
        thoi_gian_xu_ly = time.time() - self.thoi_gian_bat_dau_xu_ly if self.thoi_gian_bat_dau_xu_ly > 0 else 0
        noi_dung_bao_cao = self.Tao_Bao_Cao_Chi_Tiet(thoi_gian_xu_ly)

        # Them phan chi tiet chat luong neu o che do chat luong
        if self.che_do_xu_ly == "chat_luong" and self.ket_qua_chat_luong['chi_tiet']:
            noi_dung_bao_cao += "\n\n" + "=" * 60
            noi_dung_bao_cao += "\nCHI TIET KIEM TRA CHAT LUONG:"
            noi_dung_bao_cao += "\n" + "=" * 60
            noi_dung_bao_cao += f"\n{'ID':<6} {'CHAT LUONG':<12} {'PHAN TRAM':<10} {'TIN CAY':<12} {'TOA DO':<15}"
            noi_dung_bao_cao += "\n" + "-" * 60

            for id_theo_doi, chi_tiet in sorted(self.ket_qua_chat_luong['chi_tiet'].items()):
                chu_chat_luong = "TOT" if chi_tiet['chat_luong'] == 'tot' else "KEM"
                do_tin_cay = chi_tiet['do_tin_cay']

                if 'phan_tram_chat_luong' in chi_tiet:
                    phan_tram_chat_luong = chi_tiet['phan_tram_chat_luong']
                else:
                    phan_tram_chat_luong = chi_tiet.get('diem_chat_luong', do_tin_cay) * 100

                chu_toa_do = "KHONG CO"
                if id_theo_doi in self.khoai_tay_da_phat_hien:
                    tam = self.khoai_tay_da_phat_hien[id_theo_doi]['tam']
                    chu_toa_do = f"({tam[0]},{tam[1]})"

                noi_dung_bao_cao += f"\n{id_theo_doi:<6} {chu_chat_luong:<12} {phan_tram_chat_luong:<10.1f}% {do_tin_cay:<12.3f} {chu_toa_do:<15}"

        with open(duong_dan_file, 'w', encoding='utf-8') as f:
            f.write(noi_dung_bao_cao)

    def Lay_Duong_Dan_Nguon_Hien_Tai(self):
        """Lay duong dan nguon hien tai"""
        if self.nguon_dau_vao == "camera":
            return f"CAMERA {self.bien_camera.get()}"
        elif self.nguon_dau_vao == "image":
            return self.bien_duong_dan_anh.get()
        else:
            return self.bien_duong_dan_video.get()

    # Hien thi va cap nhat UI
    def Cap_Nhat_Hien_Thi(self, khung_hinh):
        """Cap nhat hien thi khung hinh len canvas"""
        try:
            # Them kiem tra khung hinh None
            if khung_hinh is None:
                print("CANH BAO: CO GANG HIEN THI KHUNG HINH NULL")
                return

            chieu_cao, chieu_rong = khung_hinh.shape[:2]
            self.kich_thuoc_khung_hinh_hien_thi = (chieu_rong, chieu_cao)

            chieu_rong_canvas = self.canvas.winfo_width()
            chieu_cao_canvas = self.canvas.winfo_height()

            if chieu_rong_canvas <= 1 or chieu_cao_canvas <= 1:
                self.root.after(50, lambda: self.Cap_Nhat_Hien_Thi(khung_hinh))
                return

            # Tinh toan ti le
            ti_le_w = chieu_rong_canvas / chieu_rong
            ti_le_h = chieu_cao_canvas / chieu_cao
            ti_le = min(ti_le_w, ti_le_h)

            chieu_rong_moi = int(chieu_rong * ti_le)
            chieu_cao_moi = int(chieu_cao * ti_le)

            # Thay doi kich thuoc khung hinh
            khung_hinh_da_resize = cv2.resize(khung_hinh, (chieu_rong_moi, chieu_cao_moi))

            # Chuyen sang RGB
            khung_hinh_rgb = cv2.cvtColor(khung_hinh_da_resize, cv2.COLOR_BGR2RGB)
            anh_pil = Image.fromarray(khung_hinh_rgb)
            anh_tk = ImageTk.PhotoImage(anh_pil)

            # Dat giua tren canvas
            offset_x = (chieu_rong_canvas - chieu_rong_moi) // 2
            offset_y = (chieu_cao_canvas - chieu_cao_moi) // 2

            self.canvas.delete("all")
            self.canvas.create_image(offset_x, offset_y, anchor='nw', image=anh_tk)
            self.canvas.image = anh_tk

            # Cap nhat hien thi so dem
            if hasattr(self, 'tong_so_dem'):
                self.nhan_so_luong.config(text=str(self.tong_so_dem))

        except Exception as e:
            print(f"LOI CAP NHAT HIEN THI: {e}")

    def Mo_Hop_Thoai_Ho_Tro(self):
        """Mo hop thoai ho tro khach hang"""
        if self.cua_so_ho_tro and self.cua_so_ho_tro.winfo_exists():
            self.cua_so_ho_tro.lift()
            return
        
        self.cua_so_ho_tro = tk.Toplevel(self.root)
        self.cua_so_ho_tro.title("HO TRO KHACH HANG")
        self.cua_so_ho_tro.geometry("500x650")
        self.cua_so_ho_tro.configure(bg='#2d2d2d')
        self.cua_so_ho_tro.resizable(False, False)
        
        # Tieu de
        tk.Label(self.cua_so_ho_tro, text="📞 LIEN HE HO TRO KHACH HANG",
                font=('Segoe UI', 16, 'bold'), 
                bg='#2d2d2d', fg='#00ff88').pack(pady=15)
        
        # Khung noi dung
        khung_noi_dung = tk.Frame(self.cua_so_ho_tro, bg='#2d2d2d')
        khung_noi_dung.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Ho va ten
        tk.Label(khung_noi_dung, text="HO VA TEN:", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.nhap_ten = tk.Entry(khung_noi_dung, bg='#3d3d3d', fg='#ffffff',
                                font=('Segoe UI', 10), relief='flat', bd=5)
        self.nhap_ten.pack(fill='x', pady=(0, 15))
        
        # So dien thoai
        tk.Label(khung_noi_dung, text="SO DIEN THOAI:", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.nhap_sdt = tk.Entry(khung_noi_dung, bg='#3d3d3d', fg='#ffffff',
                                font=('Segoe UI', 10), relief='flat', bd=5)
        self.nhap_sdt.pack(fill='x', pady=(0, 15))
        
        # Email
        tk.Label(khung_noi_dung, text="EMAIL:", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.nhap_email = tk.Entry(khung_noi_dung, bg='#3d3d3d', fg='#ffffff',
                                    font=('Segoe UI', 10), relief='flat', bd=5)
        self.nhap_email.pack(fill='x', pady=(0, 15))
        
        # Noi dung gop y
        tk.Label(khung_noi_dung, text="NOI DUNG GOP Y / BAO LOI:", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        
        khung_text = tk.Frame(khung_noi_dung, bg='#3d3d3d', bd=5)
        khung_text.pack(fill='both', expand=True, pady=(0, 10))
        
        self.nhap_gop_y = tk.Text(khung_text, bg='#3d3d3d', fg='#ffffff',
                                font=('Segoe UI', 10), relief='flat',
                                wrap=tk.WORD, height=8)
        self.nhap_gop_y.pack(fill='both', expand=True)
        
        # NUT GUI - THEM VAO DAY
        khung_nut_gui = tk.Frame(khung_noi_dung, bg='#2d2d2d')
        khung_nut_gui.pack(fill='x', pady=(10, 0))
        
        self.nut_gui_gop_y = tk.Button(
            khung_nut_gui,
            text="📨 GUI GOP Y",
            command=self.Gui_Gop_Y,
            bg='#00aa44',
            fg='#ffffff',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2',
            activebackground='#00cc55',
            activeforeground='#ffffff'
        )
        self.nut_gui_gop_y.pack(fill='x')
        
        # Hieu ung hover cho nut
        def on_enter_gui(e):
            self.nut_gui_gop_y.config(bg='#00cc55')
        
        def on_leave_gui(e):
            self.nut_gui_gop_y.config(bg='#00aa44')
        
        self.nut_gui_gop_y.bind('<Enter>', on_enter_gui)
        self.nut_gui_gop_y.bind('<Leave>', on_leave_gui)
        
        # Nut dong (duoi cung)
        khung_nut_dong = tk.Frame(self.cua_so_ho_tro, bg='#2d2d2d')
        khung_nut_dong.pack(fill='x', padx=20, pady=(10, 5))
        
        nut_dong = tk.Button(
            khung_nut_dong,
            text="✖ DONG",
            command=self.cua_so_ho_tro.destroy,
            bg='#666666',
            fg='#ffffff',
            font=('Segoe UI', 9),
            relief='flat',
            bd=0,
            padx=15,
            pady=8,
            cursor='hand2',
            activebackground='#777777',
            activeforeground='#ffffff'
        )
        nut_dong.pack(side='right')
        
        # Hieu ung hover cho nut dong
        def on_enter_dong(e):
            nut_dong.config(bg='#777777')
        
        def on_leave_dong(e):
            nut_dong.config(bg='#666666')
        
        nut_dong.bind('<Enter>', on_enter_dong)
        nut_dong.bind('<Leave>', on_leave_dong)
        
        # Thong tin lien he (duoi cung)
        khung_lien_he = tk.LabelFrame(self.cua_so_ho_tro, text="THONG TIN LIEN HE",
                                        bg='#2d2d2d', fg='#00bfff',
                                        font=('Segoe UI', 9, 'bold'))
        khung_lien_he.pack(fill='x', padx=20, pady=(0, 15))
        
        tk.Label(khung_lien_he, text="📧 Email: support@khoaitaychatluongcao.vn", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 9)).pack(anchor='w', padx=10, pady=3)
        
        tk.Label(khung_lien_he, text="☎ Hotline: 1900-0909", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 9)).pack(anchor='w', padx=10, pady=3)
        
        tk.Label(khung_lien_he, text="🌐 Website: www.khoaitaychatluongcao.vn", 
                bg='#2d2d2d', fg='#ffffff',
                font=('Segoe UI', 9)).pack(anchor='w', padx=10, pady=(3, 8))


    def Gui_Gop_Y(self):
        """Gui gop y cua khach hang"""
        ten = self.nhap_ten.get().strip()
        sdt = self.nhap_sdt.get().strip()
        email = self.nhap_email.get().strip()
        gop_y = self.nhap_gop_y.get('1.0', 'end').strip()
        
        # Kiem tra thong tin
        if not ten:
            messagebox.showwarning("CANH BAO", "VUI LONG NHAP HO VA TEN!")
            self.nhap_ten.focus()
            return
        
        if not sdt and not email:
            messagebox.showwarning("CANH BAO", "VUI LONG NHAP IT NHAT SO DIEN THOAI HOAC EMAIL!")
            self.nhap_sdt.focus()
            return
        
        if not gop_y:
            messagebox.showwarning("CANH BAO", "VUI LONG NHAP NOI DUNG GOP Y!")
            self.nhap_gop_y.focus()
            return
        
        # Kiem tra dinh dang email neu co
        if email and '@' not in email:
            messagebox.showwarning("CANH BAO", "EMAIL KHONG HOP LE!")
            self.nhap_email.focus()
            return
        
        # Kiem tra dinh dang so dien thoai neu co
        if sdt and (not sdt.replace('+', '').replace(' ', '').replace('-', '').isdigit() or len(sdt.replace('+', '').replace(' ', '').replace('-', '')) < 9):
            messagebox.showwarning("CANH BAO", "SO DIEN THOAI KHONG HOP LE! (IT NHAT 9 CHU SO)")
            self.nhap_sdt.focus()
            return
        
        # Xac nhan gui
        xac_nhan = messagebox.askyesno(
            "XAC NHAN GUI GOP Y",
            f"BAN CO CHAC CHAN MUON GUI GOP Y?\n\n"
            f"Ho ten: {ten}\n"
            f"SDT: {sdt if sdt else '(Khong co)'}\n"
            f"Email: {email if email else '(Khong co)'}"
        )
        
        if not xac_nhan:
            return
        
        try:
            # Luu gop y vao file
            dau_thoi_gian = datetime.now().strftime("%Y%m%d_%H%M%S")
            ten_file = f"GopY_{dau_thoi_gian}.txt"
            
            # Lay duong dan mac dinh (cung thu muc voi chuong trinh)
            duong_dan_mac_dinh = os.path.dirname(os.path.abspath(__file__))
            thu_muc_gop_y = os.path.join(duong_dan_mac_dinh, "GopY_KhachHang")
            
            # Tao thu muc luu gop y neu chua co
            if not os.path.exists(thu_muc_gop_y):
                os.makedirs(thu_muc_gop_y)
            
            duong_dan_file = os.path.join(thu_muc_gop_y, ten_file)
            
            noi_dung = f"""
    {'='*70}
                        GOP Y KHACH HANG
            HE THONG DEM & KIEM TRA CHAT LUONG KHOAI TAY AI
    {'='*70}

    THONG TIN KHACH HANG:
    {'-'*70}
    Thoi gian gui     : {datetime.now().strftime('%d/%m/%Y - %H:%M:%S')}
    Ho va ten         : {ten}
    So dien thoai     : {sdt if sdt else 'Khong cung cap'}
    Email             : {email if email else 'Khong cung cap'}
    Ma so gop y       : {dau_thoi_gian}

    NOI DUNG GOP Y / BAO LOI:
    {'-'*70}
    {gop_y}

    {'='*70}
    Trang thai        : DA TIEP NHAN
    Phan hoi du kien  : TRONG 24H
    {'='*70}

    * Ghi chu: File nay duoc tao tu dong boi he thong.
    * Lien he: support@khoaitaychatluongcao.vn | Hotline: 1900-0909
    """
            
            # Luu file
            with open(duong_dan_file, 'w', encoding='utf-8') as f:
                f.write(noi_dung)
            
            # Thong bao thanh cong
            messagebox.showinfo(
                "✓ GUI THANH CONG", 
                f"DA GUI GOP Y THANH CONG!\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📝 Ma so gop y: {dau_thoi_gian}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"✉ Cam on ban da gop y!\n"
                f"☎ Chung toi se lien he lai trong 24h."
            )
            
            # Xoa noi dung sau khi gui thanh cong
            self.nhap_ten.delete(0, 'end')
            self.nhap_sdt.delete(0, 'end')
            self.nhap_email.delete(0, 'end')
            self.nhap_gop_y.delete('1.0', 'end')
            
            # Dong cua so
            self.cua_so_ho_tro.destroy()
            
            # Cap nhat trang thai
            self.bien_trang_thai.set(f"DA TIEP NHAN GOP Y - MA SO: {dau_thoi_gian}")
            
        except Exception as e:
            messagebox.showerror("✗ LOI", f"KHONG THE LUU GOP Y:\n\n{str(e)}\n\nVui long thu lai!")

    def Bat_Hen_Gio(self):
        """Bat hen gio tat ung dung"""
        if self.hen_gio_dang_hoat_dong:
            messagebox.showwarning("CANH BAO", "HEN GIO DA DANG HOAT DONG!")
            return
        
        phut = self.bien_phut_hen_gio.get()
        
        if phut <= 0:
            messagebox.showwarning("CANH BAO", "THOI GIAN PHAI LON HON 0 PHUT!")
            return
        
        # Xac nhan
        xac_nhan = messagebox.askyesno(
            "XAC NHAN HEN GIO",
            f"UNG DUNG SE TU DONG TAT SAU {phut} PHUT.\n\n"
            f"BAN CO CHAC CHAN MUON BAT HEN GIO?"
        )
        
        if not xac_nhan:
            return
        
        self.hen_gio_dang_hoat_dong = True
        self.thoi_gian_hen_gio = time.time() + (phut * 60)
        
        # Chay thread theo doi
        self.luong_hen_gio = threading.Thread(target=self.Theo_Doi_Hen_Gio, daemon=True)
        self.luong_hen_gio.start()
        
        self.nut_hen_gio.config(state='disabled')
        self.Cap_Nhat_Trang_Thai_Hen_Gio()
        
        messagebox.showinfo("THANH CONG", 
                        f"DA BAT HEN GIO!\n\n"
                        f"Ung dung se tu dong tat sau {phut} phut.")


    def Huy_Hen_Gio(self):
        """Huy hen gio"""
        if not self.hen_gio_dang_hoat_dong:
            messagebox.showinfo("THONG BAO", "KHONG CO HEN GIO NAO DANG HOAT DONG!")
            return
        
        xac_nhan = messagebox.askyesno("XAC NHAN", "BAN CO CHAC CHAN MUON HUY HEN GIO?")
        
        if xac_nhan:
            self.hen_gio_dang_hoat_dong = False
            self.thoi_gian_hen_gio = None
            self.nut_hen_gio.config(state='normal')
            self.nhan_trang_thai_hen_gio.config(text="DA HUY HEN GIO", fg='#ffaa00')
            messagebox.showinfo("THANH CONG", "DA HUY HEN GIO!")


    def Theo_Doi_Hen_Gio(self):
        """Theo doi va xu ly hen gio"""
        while self.hen_gio_dang_hoat_dong:
            if self.thoi_gian_hen_gio:
                thoi_gian_con_lai = self.thoi_gian_hen_gio - time.time()
                
                if thoi_gian_con_lai <= 0:
                    # Het gio - tat ung dung
                    self.root.after(0, self.Tat_Ung_Dung_Hen_Gio)
                    break
                
                # Cap nhat trang thai
                self.root.after(0, self.Cap_Nhat_Trang_Thai_Hen_Gio)
            
            time.sleep(1)

    def Cap_Nhat_Trang_Thai_Hen_Gio(self):
        """Cap nhat hien thi trang thai hen gio"""
        if self.hen_gio_dang_hoat_dong and self.thoi_gian_hen_gio:
            thoi_gian_con_lai = int(self.thoi_gian_hen_gio - time.time())
            
            if thoi_gian_con_lai > 0:
                phut = thoi_gian_con_lai // 60
                giay = thoi_gian_con_lai % 60
                
                chu_trang_thai = f"CON LAI: {phut:02d}:{giay:02d}"
                self.nhan_trang_thai_hen_gio.config(text=chu_trang_thai, fg='#00ff88')
                
                # Canh bao khi con 1 phut
                if thoi_gian_con_lai == 60:
                    self.root.after(0, lambda: messagebox.showwarning(
                        "CANH BAO HEN GIO",
                        "UNG DUNG SE TAT SAU 1 PHUT!\n\nVUI LONG LUU CONG VIEC!"))


    def Tat_Ung_Dung_Hen_Gio(self):
        """Tat ung dung khi het gio"""
        try:
            # Dung xu ly neu dang chay
            if self.dang_xu_ly:
                self.Dung_Xu_Ly()
            
            # Luu cau hinh
            self.Luu_Cau_Hinh()
            
            # Thong bao
            messagebox.showinfo("HEN GIO", 
                            "DA HET THOI GIAN HEN!\n\n"
                            "UNG DUNG SE DONG SAU 3 GIAY...")
            
            # Cho 3 giay roi dong
            self.root.after(3000, self.root.destroy)
            
        except Exception as e:
            print(f"LOI TAT UNG DUNG: {e}")
            self.root.destroy()

    def Hien_Thong_Bao_Mac_Dinh(self):
        """Hien thi thong bao mac dinh tren canvas"""
        self.canvas.delete("all")
        chieu_rong_canvas = self.canvas.winfo_width()
        chieu_cao_canvas = self.canvas.winfo_height()

        if chieu_rong_canvas > 1 and chieu_cao_canvas > 1:
            self.canvas.create_text(
                chieu_rong_canvas // 2, chieu_cao_canvas // 2,
                text="CHUA TAI DU LIEU\n\nVUI LONG CHON NGUON VA NHAN 'TAI DU LIEU'",
                fill='white', font=('Segoe UI', 14), justify='center'
            )

    def Cap_Nhat_Thoi_Gian(self):
        """Cap nhat thoi gian"""
        thoi_gian_hien_tai = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
        self.bien_thoi_gian.set(thoi_gian_hien_tai)
        self.root.after(1000, self.Cap_Nhat_Thoi_Gian)

    # Quan ly cau hinh
    def Tai_Cau_Hinh(self):
        """Tai cau hinh"""
        try:
            if os.path.exists("cau_hinh_khoai_tay.json"):
                with open("cau_hinh_khoai_tay.json", 'r', encoding='utf-8') as f:
                    cau_hinh_da_luu = json.load(f)
                    self.cau_hinh.update(cau_hinh_da_luu)
        except Exception as e:
            print(f"LOI TAI CAU HINH: {e}")

    def Luu_Cau_Hinh(self):
        """Luu cau hinh"""
        try:
            self.cau_hinh.update({
                "duong_dan_model": self.bien_duong_dan_model.get(),
                "nguon_video": self.bien_duong_dan_video.get(),
                "nguon_anh": self.bien_duong_dan_anh.get() if hasattr(self, 'bien_duong_dan_anh') else "",
                "nguong_tin_cay": self.bien_tin_cay.get(),
                "nguong_iou": self.bien_iou.get(),
                "nguong_chat_luong": 0.70,  # Co dinh 70%
                "chi_so_camera": self.bien_camera.get(),
                "hien_toa_do": self.bien_hien_toa_do.get(),
                "duong_dan_xuat": self.bien_duong_dan_xuat.get()
            })

            with open("cau_hinh_khoai_tay.json", 'w', encoding='utf-8') as f:
                json.dump(self.cau_hinh, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"LOI LUU CAU HINH: {e}")

    def __del__(self):
        """Ham huy"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()


def Ham_Chinh():
    """Ham chinh"""
    root = tk.Tk()
    ung_dung = He_Thong_Dem_Khoai_Tay_Pro(root)

    def Xu_Ly_Dong_Ung_Dung():
        """Xu ly khi dong ung dung"""
        if ung_dung.dang_xu_ly:
            ung_dung.Dung_Xu_Ly()
        ung_dung.Luu_Cau_Hinh()
        if hasattr(ung_dung, 'cap') and ung_dung.cap:
            ung_dung.cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", Xu_Ly_Dong_Ung_Dung)
    root.mainloop()


if __name__ == "__main__":
    Ham_Chinh()