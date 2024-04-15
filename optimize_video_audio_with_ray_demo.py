# Databricks notebook source
# MAGIC %pip install -U numba
# MAGIC %pip install numpy==1.23.5
# MAGIC %pip install torch
# MAGIC %pip install tqdm
# MAGIC %pip install more-itertools
# MAGIC %pip install tiktoken==0.3.3
# MAGIC %pip install openai-whisper
# MAGIC %pip install shapely
# MAGIC %pip install scikit-image
# MAGIC %pip install imgaug
# MAGIC %pip install pyclipper
# MAGIC %pip install lmdb
# MAGIC %pip install tqdm
# MAGIC %pip install visualdl
# MAGIC %pip install rapidfuzz
# MAGIC %pip install opencv-contrib-python
# MAGIC %pip install cython
# MAGIC %pip install lxml
# MAGIC %pip install premailer
# MAGIC %pip install openpyxl
# MAGIC %pip install attrdict
# MAGIC %pip install PyMuPDF
# MAGIC %pip install Pillow>=10.0.0
# MAGIC %pip install pyyaml
# MAGIC %pip install paddlepaddle #-i https://pypi.tuna.tsinghua.edu.cn/simple
# MAGIC %pip install paddleocr==2.7.0.1 # Recommend to use version 2.0.1+
# MAGIC %pip install fuzzywuzzy
# MAGIC %pip install requests
# MAGIC %pip install ffmpeg-python
# MAGIC %pip install imagehash
# MAGIC %pip install decord
# MAGIC %pip install imageio
# MAGIC %pip install imageio[pyav]
# MAGIC %pip install imageio[ffmpeg]
# MAGIC %pip install pyacoustid
# MAGIC %pip install pydub
# MAGIC %pip install ray[default,tune,client]==2.10.0
# MAGIC %pip install retry
# MAGIC %pip install scipy
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

# !!! Need to make sure the set up matches cluster definition???
num_cpu_cores_per_worker =4 # total cpu's present in each node
num_cpus_head_node = 3
num_gpu_per_worker = 1
num_gpus_head_node = 1
max_worker_nodes = 4

ray_conf = setup_ray_cluster(
  min_worker_nodes=1,
  max_worker_nodes=max_worker_nodes,
  num_cpus_head_node= num_cpus_head_node,
  num_gpus_head_node= num_gpus_head_node,
  num_cpus_per_node=num_cpu_cores_per_worker,
  num_gpus_per_node=num_gpu_per_worker
  # autoscale = True
  # num_cpus_worker_node=num_cpu_cores_per_worker,
  # num_gpus_worker_node =num_gpu_per_worker,
  # ray_temp_root_dir = '/dbfs/pj/vivvex'
  # collect_log_to_path="/dbfs/path/to/ray_collected_logs"
  )

# COMMAND ----------

import ast
import errno
import glob
import hashlib
import io
import json
import logging
import os
import shutil
import signal
import ssl
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Iterator, List, Tuple
from urllib.request import urlopen

import audioread
import base64
import cv2
import imagehash
import imageio
import imageio.v3 as iio
import numpy as np
import pandas as pd
import psycopg2
import torch
import urllib.error
import whisper
from PIL import Image
from acoustid import audioread
from chromaprint import Fingerprinter, decode_fingerprint, hash_fingerprint
from decord import VideoReader, cpu, gpu
from delta.tables import *
from fuzzywuzzy import fuzz
from paddleocr import PaddleOCR
from pydub import AudioSegment, scipy_effects
from pyspark.sql import functions as fn, types as T
import retry



# COMMAND ----------

from pydub import AudioSegment
from scipy.ndimage import maximum_filter
from scipy.signal import spectrogram
import hashlib
def cryptoHashCreative(assetLocalPath):
	A=hashlib.blake2b()
	with open(assetLocalPath,'rb')as B:
		while(C:=B.read(8192)):A.update(C)
	D=A.hexdigest();return D
 
class AudioPeaksFingerprinter:
	SAMPLE_RATE=44100;PEAK_BOX_SIZE=30;POINT_EFFICIENCY=.8;FFT_WINDOW_SIZE=.2;TARGET_START=.05;TARGET_T=1.8;TARGET_F=4000;SILENCE_THRESHOLD=-5e1;MIN_SILENCE_LEN=500;FREQUENCY_IDX=0;TIME_OFFSET_IDX=1
	def target_zone(A,anchor,points,width,height,t):
		D=height;C=anchor;E=[];F=C[A.TIME_OFFSET_IDX]+t;H=F+width;G=C[A.FREQUENCY_IDX]-D*.5;I=G+D
		for B in points:
			if B[A.FREQUENCY_IDX]<G or B[A.FREQUENCY_IDX]>I:continue
			if B[A.TIME_OFFSET_IDX]<F or B[A.TIME_OFFSET_IDX]>H:continue
			E.append(B)
		return E
	def fingerprint_audio_file(A,filename):
		F=[];G=.0
		try:
			C=None
			try:C=AudioSegment.from_file(filename).set_channels(1).set_frame_rate(A.SAMPLE_RATE).normalize()
			except Exception as E:
				if str(E)!='list index out of range':raise
			if C is not None:
				G=C.duration_seconds;L=np.frombuffer(C.raw_data,np.int16);M=int(A.SAMPLE_RATE*A.FFT_WINDOW_SIZE);N,O,B=spectrogram(L,A.SAMPLE_RATE,nperseg=M);P=maximum_filter(B,size=A.PEAK_BOX_SIZE,mode='constant',cval=.0);Q=B==P;H,I=Q.nonzero();R=B[H,I];S=R.argsort()[::-1];T=[(H[A],I[A])for A in S];U=B.shape[0]*B.shape[1];V=int(U/A.PEAK_BOX_SIZE**2*A.POINT_EFFICIENCY);J=np.array([(N[B[A.FREQUENCY_IDX]],O[B[A.TIME_OFFSET_IDX]])for B in T[:V]])
				for D in J:
					for K in A.target_zone(D,J,A.TARGET_T,A.TARGET_F,A.TARGET_START):F.append((hash((D[A.FREQUENCY_IDX],K[A.FREQUENCY_IDX],K[A.TIME_OFFSET_IDX]-D[A.TIME_OFFSET_IDX])),D[A.TIME_OFFSET_IDX]))
		except Exception as E:print('Exception occurred: ',E);print(traceback.format_exc())
		finally:return F,G
  
_A=None
def transcribe(audio_filename,duration_in_seconds,whisper_model):
	C=_A;A=_A
	try:
		D,E=whisper_model.transcribe(audio_filename);A=''
		for B in D:
			if B.start==.0 and B.end>=duration_in_seconds-.5:A='(MUSIC)';break
			else:A+=B.text
		C=E.language;A=A.strip()
	except Exception as F:print(F)
	finally:return C,A
def translate(audio_filename,whisper_model):
	A=_A
	try:
		B,E=whisper_model.transcribe(audio_filename,task='translate');A=''
		for C in B:A+=C.text
	except Exception as D:print(D);A=_A
	finally:return A
def get_ocr_results(pil_image,paddle_ocr):
	B=paddle_ocr;E=[];C=_A
	if B is _A:return _A,_A
	try:A=np.array(pil_image);A=A[:,:,::-1].copy();C=B.ocr(A,cls=True)
	except Exception as D:print(D);print(traceback.format_exc())
	finally:return C
class VideoFrames:
	def __init__(A,filepath):
		I='VideoReader';H='imageio';G='audio_codec';A.filepath=filepath;A.index=0;A.first_frame_offset=0;A.frames=[];A.width=_A;A.height=_A;A.vr=imageio.get_reader(A.filepath);B=A.vr.get_meta_data()
		if G in B:E=B[G]
		else:E=''
		if E=='aac':A.mode=H;A.fps=B['fps'];A.durationInSeconds=round(B['duration']);A.frame_count=A.vr.count_frames()
		else:A.mode=I;A.vr=VideoReader(A.filepath,ctx=cpu(0));A.fps=A.vr.get_avg_fps();A.durationInSeconds=round(len(A.vr)/A.fps);A.frame_count=len(A.vr)
		A.frame_increment=int(round(A.fps/2))if A.durationInSeconds>=10 else int(round(A.fps/10))
		if A.frame_increment==0:A.frame_increment=1
		J=0;A.first_frame_offset=0
		for F in range(A.first_frame_offset,A.frame_count,A.frame_increment):
			if A.mode==H:C=A.vr.get_data(F);D=Image.fromarray(C)
			elif A.mode==I:C=A.vr[F];D=Image.fromarray(C.asnumpy())
			A.frames.append(D)
		A.width,A.height=D.size
	def __iter__(A):A.index=0;return A
	def __next__(A):
		if A.index>=len(A.frames):raise StopIteration
		B=A.frames[A.index];A.index+=1;return B
def video_hash(vfr,paddle_ocr,perform_ocr=False,bookend_secs=5.):
	'Extract frames from video using VideoReader and return array of hashes for use in creative matching';B=vfr
	try:
		A=_A;D=B.first_frame_offset
		for E in B:
			if perform_ocr:
				try:
					C=get_ocr_results(E,paddle_ocr)
					if C is not _A and C!='':
						if A is _A:A=[]
						A.append({'frame_index':D,'text':C})
				except Exception as F:print(F);print(traceback.format_exc())
			D+=B.frame_increment
	finally:return A
def extract_audio(video_file,audio_file,normalize=True,loHz=8000,hiHz=200,format_w='wav'):
	'Extracts audio from a video file and applies lo/high frequency filters'
	try:
		A=AudioSegment.from_file(video_file);C=-26;D=A.dBFS
		if A.channels>1:A=A.set_channels(1)
		if abs(D-C)>1:A=A.apply_gain(C-D)
		if normalize:A=A.normalize();A=A.set_frame_rate(44100)
		try:A=AudioSegment.low_pass_filter(A,loHz)
		except Exception as B:print(B)
		try:A=AudioSegment.high_pass_filter(A,hiHz)
		except Exception as B:print(B)
	except:A=AudioSegment.empty()
	A.export(audio_file,format=format_w).close()


# COMMAND ----------

@retry.retry(tries=5, delay=10)
def get_paddle():
    for metric in  ['cls','det', 'rec']:
        dirpath = Path(f'/root/.paddleocr/whl/{metric}/') / '"en'
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
    return PaddleOCR(use_angle_cls=True, det_limit_side_len=1080, det_db_box_thresh= 0.5, lang='en', 
                    show_log=False, debug=True, use_gpu = False, ocr_version = 'PP-OCRv3',
                    det_model_dir ='/dbfs/usr/pj/whl/det/',cls_model_dir ='/dbfs/usr/pj/whl/cls/',rec_model_dir ='/dbfs/usr/pj/whl/rec/')



class VideoOCR:
    '''
    Class to finger print the audio
    '''
    def __init__(self):
        self.unverified_context = ssl._create_unverified_context()
        self.perform_ocr = True
        self.paddle_ocr = get_paddle()

    def __call__(self, row: dict) -> dict:
        if row['creative_found']:
                splits = row['combo'].split("\t")
                url = splits[3]
                audio_hash_dict = json.loads(row['audio_hash'])
                durationInSeconds = audio_hash_dict['audio_durationInSeconds']
                id = url.split('/')[-1].split('-')[2].split('.')[0]
                file_path = f'/dbfs/user/tmp/'+ id +'.mp4'
                start = time.time()
                print(datetime.now(),f"get video frames {file_path}")
                vfr = VideoFrames(file_path)
                print(datetime.now(),f"get video cropbox {file_path}")
                ocrTexts = video_hash(
                    vfr, self.paddle_ocr, self.perform_ocr
                )
                if durationInSeconds == 0:
                    durationInSeconds = vfr.durationInSeconds
                end = time.time()
                ocr_video_hashing_duration =  end - start
                creativeJson = json.dumps({
                                                "durationInSeconds": durationInSeconds, 
                                                "size": (vfr.width, vfr.height), 
                                                "ocr_texts": ocrTexts,
                                                "ocr_video_hashing_duration": ocr_video_hashing_duration,
                                            })
                row['final_dict'] = {"video_hash" :creativeJson,
                                    "audio_transcription" : row['audio_transcription'],
                                    "audio_hash"  : row['audio_hash'] }
                row['final_dict'] = json.dumps(row['final_dict'])
        else:
                row['final_dict'] = {
                                    "audio_hash"  : row['audio_hash'] }
                row['final_dict'] = json.dumps(row['final_dict'])

        return row
    

class WhisperTranscription:
    '''
    Class to finger print the audio
    '''
    def __init__(self):
        self.perform_transcription = True
        self.perform_ocr = True
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model_type ='medium'
        self.transcribe_whisper_model = whisper.load_model(whisper_model_type, device = DEVICE)

    def __call__(self, row: dict) -> dict:
        if row['creative_found']:
            splits = row['combo'].split("\t")
            audio_hash_dict = json.loads(row['audio_hash'])
            durationInSeconds = audio_hash_dict['audio_durationInSeconds']
            id = url.split('/')[-1].split('-')[2].split('.')[0]
            file_path = f'/dbfs/user/tmp/'+ id +'.mp4'
            file_path_a = file_path.replace(".mp4", ".wav")
            start = time.time()
 
            if durationInSeconds > 300:
                print(datetime.now(),f"creative duration exceeds 5 minutes\tIGNORING\t{nm}")
                row['transcription'] = '{}'
                return row
            transcription = {"language": None, "text": None}
            translation = None
            print(datetime.now(),f"transcribe audio {file_path}\tlength: {durationInSeconds} secs")
            if self.perform_transcription:
                transcription = transcribe(file_path_a, durationInSeconds, self.transcribe_whisper_model)
            end = time.time()
            print("transcribe audio " ,end - start)  
            transcribe_duration = end - start
            start = time.time()
            translate_duration = 0
            print(datetime.now(),f"{transcription=}\t{file_path}")
            if 'language' in transcription:
                print(datetime.now(),f"transcribed language {transcription['language']}\t{file_path}")
                if transcription['language'] != 'en' and transcription['text'] != "(MUSIC)":
                    translation = translate(file_path_a, self.transcribe_whisper_model)
                    end = time.time()
                    translate_duration = end - start
        
            creativeJson = json.dumps({
                                        "predicted_language": transcription["language"] if transcription is not None and "language" in transcription else "en",
                                        "transcription": transcription["text"] if transcription is not None and "text" in transcription else "(MUSIC)",
                                        "transcription_en": translation if translation is not None else "",
                                        "transcribe_duration": transcribe_duration,
                                        "translate_duration": translate_duration
                                        })
            row['audio_transcription'] = creativeJson
        return row
    

class FingerprintAudio:
    '''
    Class to finger print the audio
    '''
    def __init__(self):
        self.unverified_context = ssl._create_unverified_context()
        self.unverified_context.set_ciphers('DEFAULT@SECLEVEL=1')
        self.fg = AudioPeaksFingerprinter()
        perform_transcription = True
        self.perform_ocr = True
        self.creative_iso2code=None
        self.creative_provider=None
        self.creative_type=None
        self.creative_subtype=None
        self.url=None
    def __call__(self, row: dict) -> dict:
        splits = row['combo'].split("\t")
        self.creative_iso2code= splits[0]
        self.creative_provider= splits[1]
        self.creative_type = splits[2].split("/")[0].strip().title()
        if self.creative_provider.lower() in ("deeplisten","mediawatch"):
            self.creative_subtype = "TV"
        elif self.creative_provider.lower() == "edo":
            self.creative_subtype = "AVOD"
        self.url = splits[3]
        start = time.time()
        id = self.url.split('/')[-1].split('-')[2].split('.')[0]
        nm = f'/dbfs/user/tmp/'+ id +'.mp4'
        if os.path.exists(nm):
            os.remove(nm)
        print(datetime.now(),f"processing {nm} / {self.url}")
        try:
            with urlopen(url, context=self.unverified_context) as rsp:
                with open(nm, 'wb') as mp4:
                    mp4.write(rsp.read())
            row['creative_found'] = True
        except:
            print(datetime.now(),f"could not download {nm} / {self.url}")
            row['creative_found'] = False
        end = time.time()
        print("Download the file" ,end - start)  
        download_duration = end - start
        if row['creative_found']:
            with open(nm, "rb") as f:
                fileHash = hashlib.blake2b()
                while chunk := f.read(8192):
                    fileHash.update(chunk)
            cryptoHash = fileHash.hexdigest()
            end = time.time()
            print("crypto hashing end" ,end - start)
            crypto_hash_duration = end - start  
            file_path_a = nm.replace(".mp4", ".wav")
            extract_audio(nm, file_path_a)
            end = time.time()
            print("extract audio" ,end - start)
            audio_extract_duration = end - start   
            audio_hashes, durationInSeconds = self.fg.fingerprint_audio_file(file_path_a)
            end = time.time()
            print("hashed audio duration" ,end - start)  
            audio_hash_duration = end - start 
            creativeJson = json.dumps({
                                        "country_iso_2_code": self.creative_iso2code,
                                        "creative_source": self.creative_provider,
                                        "creative_type": self.creative_type,
                                        "creative_subtype": self.creative_subtype,
                                        "id": id, 
                                        "url": self.url,
                                        "cryptoHash": cryptoHash, 
                                        "audio_hashes": audio_hashes,
                                        "audio_durationInSeconds" : durationInSeconds,
                                        "download_duration": download_duration,
                                        "crypto_hash_duration": crypto_hash_duration,
                                        "audio_extract_duration": audio_extract_duration,
                                        "audio_hash_duration": audio_hash_duration
                                        })
        else:
            creativeJson = json.dumps({
                                        "country_iso_2_code": self.creative_iso2code,
                                        "creative_source": self.creative_provider,
                                        "creative_type": self.creative_type,
                                        "creative_subtype": self.creative_subtype,
                                        "id": id, 
                                        "url": self.url,
                                        "error": "creative not found!"
                                        })
        row['audio_hash'] = creativeJson
        return row

# COMMAND ----------

@fn.pandas_udf(T.StringType())
def parse_creatives(urls: pd.Series) -> pd.Series:
    start = time.time()
    import ray
    import ray.data

    ray.init(ray_conf[1])


    @ray.remote
    def ray_data_task(ds = None):
        ds = ray.data.from_pandas(pd.DataFrame(urls.to_list(),columns = ['combo']))

        print("shape:",urls.shape[0])
        preds = (
        ds.repartition(urls.shape[0])
        .map(
            FingerprintAudio,
            compute=ray.data.ActorPoolStrategy(min_size=1,max_size=18),
            num_cpus=1,
        )
        .map(
            WhisperTranscription,
            compute=ray.data.ActorPoolStrategy(min_size=1,max_size=10),
            num_gpus=.5,
        )
        .map(
            VideoOCR,
            compute=ray.data.ActorPoolStrategy(min_size=1,max_size=18),
            num_cpus=1,
        ))
        end = time.time()
        print("Loaded model dependencies" ,end - start)  

        final_df = preds.to_pandas()

        return final_df['final_dict']  
    
    return ray.get(ray_data_task.remote(urls)) 
 


# COMMAND ----------

# ray_conf = f"{spark.conf.get('spark.driver.host')}:9339"
silver_events_schema = T.StructType([ 
  StructField('Url', T.StringType()),  
  StructField('CountryISO2Code', T.StringType()),  
  StructField('Provider', T.StringType()),  
  StructField('MimeType', T.StringType()),  
  ])


# COMMAND ----------

json_payload = []
for url in ['https://stcommonmediasvcdeu2.blob.core.windows.net/mediaservices/KMAdCreatives/deeplisten-US-1040399071.mp4?sv=2020-08-04&st=2024-04-05T03%3A47%3A14Z&se=2024-05-20T03%3A47%3A14Z&sr=b&sp=r&sig=Susr590sMEBsmMzyRyGYK3fYg64cE%2Beyd42qQb0D3Gw%3D',
'https://stcommonmediasvcdeu2.blob.core.windows.net/mediaservices/KMAdCreatives/deeplisten-US-1043596658.mp4?sv=2020-08-04&st=2024-04-05T04%3A02%3A19Z&se=2024-05-20T04%3A02%3A19Z&sr=b&sp=r&sig=mtk2AgJx%2B2DfGRUGU14sv4E9jrn%2BgFeejHoc3NE35uo%3D',
'https://stcommonmediasvcdeu2.blob.core.windows.net/mediaservices/KMAdCreatives/deeplisten-US-1043876928.mp4?sv=2020-08-04&st=2024-04-05T04%3A06%3A57Z&se=2024-05-20T04%3A06%3A57Z&sr=b&sp=r&sig=1KBd%2BOXxcroZy%2FhQe%2Fjo02yp3v92cnma4oTtqqewiOM%3D',
'https://stcommonmediasvcdeu2.blob.core.windows.net/mediaservices/KMAdCreatives/deeplisten-US-1045718999.mp4?sv=2020-08-04&st=2024-04-05T05%3A02%3A54Z&se=2024-05-20T05%3A02%3A54Z&sr=b&sp=r&sig=4%2BsCTO%2FQZVY3SCM%2FyGkb7MwHdOLrnUT22OtaMc5pwnU%3D',
'https://stcommonmediasvcdeu2.blob.core.windows.net/mediaservices/KMAdCreatives/deeplisten-US-1046069863.mp4?sv=2020-08-04&st=2024-04-05T05%3A11%3A59Z&se=2024-05-20T05%3A11%3A59Z&sr=b&sp=r&sig=YxRvTEHEPL5%2Fl2NoGlq2mPsnBA8xn6MauXpTAzMGoo0%3D',
'https://stcommonmediasvcdeu2.blob.core.windows.net/mediaservices/KMAdCreatives/deeplisten-US-1046489435.mp4?sv=2020-08-04&st=2024-04-05T05%3A21%3A39Z&se=2024-05-20T05%3A21%3A39Z&sr=b&sp=r&sig=YJVezX4e34OoAEOUHz%2FBq3Yccuc%2FEkpv1MLO5ZM9PbM%3D',
'https://stcommonmediasvcdeu2.blob.core.windows.net/mediaservices/KMAdCreatives/deeplisten-US-1047743724.mp4?sv=2020-08-04&st=2024-04-05T05%3A46%3A56Z&se=2024-05-20T05%3A46%3A56Z&sr=b&sp=r&sig=N7EGrcm7bJD0FeKs6x26afaenYYvpO0H0HgK%2FsbfyR0%3D',
'https://stcommonmediasvcdeu2.blob.core.windows.net/mediaservices/KMAdCreatives/deeplisten-US-1048204017.mp4?sv=2020-08-04&st=2024-04-05T06%3A11%3A41Z&se=2024-05-20T06%3A11%3A41Z&sr=b&sp=r&sig=so7wG%2B4A2XzcEaOtyQDmVRglbXEa%2B4bDYp5H7Y2DPHs%3D',
'https://stcommonmediasvcdeu2.blob.core.windows.net/mediaservices/KMAdCreatives/deeplisten-US-1051202355.mp4?sv=2020-08-04&st=2024-04-05T07%3A11%3A47Z&se=2024-05-20T07%3A11%3A47Z&sr=b&sp=r&sig=X3ekDMD0ICYAZnIgzdSpjeoH7BCXE0%2B4MOLG2zTRg98%3D',
'https://stcommonmediasvcdeu2.blob.core.windows.net/mediaservices/KMAdCreatives/deeplisten-US-1051449715.mp4?sv=2020-08-04&st=2024-04-05T07%3A16%3A45Z&se=2024-05-20T07%3A16%3A45Z&sr=b&sp=r&sig=w5pyjpLe77HmE19GoZcnW83gLhBI4FTJV7ctksBOx8c%3D']:

    json_payload.append({"Url":url,"Provider":"deeplisten","CountryISO2Code":"US","MimeType":"video/mp4"})
    

# COMMAND ----------

df = (spark
    .read
    .json(sc.parallelize(json_payload), schema=silver_events_schema)
)
# df=df.select("*").limit(5)
display(df)

# COMMAND ----------

ray_conf

# COMMAND ----------

df.count()

# COMMAND ----------

df = df.repartition(1)
df = df.withColumn("combo", fn.concat(fn.col("CountryISO2Code"), fn.lit("\t"), fn.col("Provider"), fn.lit("\t"), fn.col("MimeType"), fn.lit("\t"), fn.col("Url")))
# display(df)
df = df.withColumn("creativejson", parse_creatives(fn.col("combo")))#.drop("combo")
df.write.mode('overwrite').option("mergeSchema", "true").saveAsTable('default.fingerprint_temp')

# COMMAND ----------

shutdown_ray_cluster()

# COMMAND ----------

# MAGIC %sql
# MAGIC select* from 
# MAGIC -- truncate table
# MAGIC default.fingerprint_temp
# MAGIC -- order by 1

# COMMAND ----------

dbutils.fs.rm("/user/tmp", True)
dbutils.fs.mkdirs("/user/tmp")

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /user/tmp
