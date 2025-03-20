import glob,random
import shutil,os,torchaudio,torch
from torchmetrics.functional.audio import signal_noise_ratio
trainset_num = num
speech_source_path = "XXX"
noise_source_path = "XXX"

sp_ls = sorted(glob.glob(os.path.join(speech_source_path, "**/*.flac"), recursive=True))
ns_ls = sorted(glob.glob(noise_source_path+"/*.CHX.wav"))
print("%d,%d"%(len(sp_ls),len(ns_ls)))

des = "XXX"

if os.path.exists(des):
    shutil.rmtree(des)
os.makedirs(des)

noise_tuple = ['BUS','CAF','PED','STR']

tag = 0
for tag in range(trainset_num):
    npath = random.sample(ns_ls,1)[0]
    while not any(nt in npath for nt in noise_tuple):
        npath = random.sample(ns_ls,1)[0]
    noise_type = npath.split("/")[-1].split("_")[-1].split(".")[0]
    ns,fs = torchaudio.load(npath)
    snr = random.uniform(-5, 15)
    sp_path = sp_ls[tag]
    sp,_ = torchaudio.load(sp_ls[tag])
    start = random.randint(0,ns.shape[-1]-sp.shape[-1]-1)
    ns_slice = ns[...,start:start+sp.shape[-1]]
    snr_des = snr
    snr = torch.pow(10, -torch.tensor(snr)/20)
    noisy = sp + torch.sqrt(torch.sum(sp**2)/(torch.sum(ns_slice**2) + 1e-8)) * snr * ns_slice

    norm = max(noisy.abs().max(),sp.abs().max())
    sp = 0.999*sp/norm
    noisy = 0.999*noisy/norm
    if noisy.abs().max()>1 or sp.abs().max()>1:
        print('error')
        raise KeyboardInterrupt
            
    file_prefix = os.path.basename(sp_path).split('.')[0]
    npath = os.path.join(des, '%s_%s_%d_noisy.flac' % (file_prefix, noise_type, snr_des)) #audio
    torchaudio.save(npath, noisy, fs)
    tpath = os.path.join(des,'%s_%s_%d_trans.txt'%(file_prefix, noise_type, snr_des)) #text
    print("%.2f"%(tag/trainset_num),end='\r')
    tag+=1
