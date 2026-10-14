"""Render report-only figures from saved outputs and original prepared data.

Run with Python from any directory. This script does not train or run inference,
modify notebooks/checkpoints, or crop existing raster figures.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
from PIL import Image, ImageOps

root = Path(__file__).resolve().parents[3]
assets = root / "report/source_latex/assets"
manifest = []


def mdrows(text):
    lines = [line for line in text.splitlines() if line.startswith("|")]
    return [[value.strip() for value in line.strip().strip("|").split("|")] for line in lines[2:]]


def save(fig, relative, source, method):
    path = assets / relative
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    manifest.append(dict(asset=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source=source, method=method))


def build_eda():
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12,
        "figure.dpi": 180, "axes.spines.top": False, "axes.spines.right": False})
    countfmt=FuncFormatter(lambda x,p:f'{x:,.0f}')
    nb=json.loads((root/'source_code/A1.ipynb').read_text(encoding='utf-8'))
    output=nb['cells'][8]['outputs'][2]['data']['text/markdown']
    rows=mdrows(''.join(output));assert len(rows)==43
    fig,ax=plt.subplots(figsize=(10,5.4),layout='constrained');pos=np.arange(43)
    for offset,j,label,color in [(-.26,2,'Train','#0072B2'),(0,3,'Validation','#E69F00'),(.26,4,'Test','#009E73')]:
        vals=[int(r[j].replace(',','')) for r in rows]
        ax.bar(pos+offset,vals,.25,label=label,color=color)
    ax.set(xticks=pos,xlabel='Class ID',ylabel='Số ảnh',title='GTSRB: phân bố 43 lớp theo split')
    ax.tick_params(axis='x',labelsize=9);ax.yaxis.set_major_formatter(countfmt);ax.legend()
    save(fig,'a1/class_distribution_report.png',dict(notebook='source_code/A1.ipynb',cell=8,output=2),'Bar heights copied exactly from saved 43-class count table; formatted ticks only.')

    nb=json.loads((root/'source_code/A2.kaggle.ipynb').read_text(encoding='utf-8'))
    rows=mdrows(''.join(nb['cells'][6]['outputs'][1]['data']['text/markdown']))
    names=[r[0] for r in rows];assert len(rows)==20
    fig,ax=plt.subplots(figsize=(10,5.8),layout='constrained');pos=np.arange(20)
    for offset,j,label,color in [(-.26,1,'Train','#0072B2'),(0,2,'Validation','#E69F00'),(.26,3,'Test','#009E73')]:
        ax.bar(pos+offset,[int(r[j].replace(',','')) for r in rows],.25,label=label,color=color)
    ax.set(xticks=pos,xticklabels=names,xlabel='Lớp VOC2012',ylabel='Số bbox thường',title='VOC2012: phân bố 20 lớp theo split')
    ax.tick_params(axis='x',rotation=65,labelsize=10);ax.yaxis.set_major_formatter(countfmt);ax.legend()
    save(fig,'a2/class_distribution_report.png',dict(notebook='source_code/A2.kaggle.ipynb',cell=6,output=1),'Grouped bars from saved class counts, preserving all three splits.')

    base=root/'datasets/voc';data={}
    sources=[]
    for split,col in [('train',1),('val',2),('test',3)]:
        p=base/'voc2012'/f'{split}.csv'
        with p.open(encoding='utf-8',newline='') as f:
            records=list(csv.DictReader(f))
        for r in records:
            for k in ['boxes','labels','difficult','truncated']:r[k]=json.loads(r[k])
        counts=Counter(label for r in records for label,d in zip(r['labels'],r['difficult']) if not d)
        assert [counts[i] for i in range(1,21)]==[int(r[col].replace(',','')) for r in rows]
        data[split]=records
        sources.append(dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    figs=[plt.subplots(figsize=(8.5,5.4),layout='constrained') for _ in range(2)]
    for split,records in data.items():
        boxes=np.array([b for r in records for b,d in zip(r['boxes'],r['difficult']) if not d]).reshape(-1,4)
        wh=boxes[:,2:]-boxes[:,:2]
        figs[0][1].hist(np.sqrt(wh.prod(1)),bins=40,histtype='step',label=split)
        figs[1][1].hist(np.log2(wh[:,0]/wh[:,1]),bins=40,histtype='step',label=split)
    figs[0][1].axvline(32,color='#D55E00',ls='--',label='small < 32²')
    figs[0][1].axvline(96,color='#009E73',ls='--',label='large > 96²')
    for (fig,ax),name,title,xlabel in zip(figs,['bbox_area_report','bbox_ratio_report'],['Kích thước bbox ảnh gốc','Tỷ lệ bbox'],['√area (px)','log₂(width/height)']):
        ax.set(title=title,xlabel=xlabel,ylabel='Số bbox thường');ax.yaxis.set_major_formatter(countfmt);ax.legend(fontsize=11)
        save(fig,'a2/'+name+'.png',sources,'Same regular-bbox arrays and 40-bin histogram calculation as notebook EDA; all 60 class counts verified against saved Kaggle table.')

    # Recreate only the deterministic preprocessing sample, without model inference.
    r=data['train'][0];image=Image.open(base/r['image']).convert('RGB');boxes=np.array(r['boxes'],dtype=np.float32)
    def draw(im,bxs,title):
        fig,ax=plt.subplots(figsize=(8.5,6),layout='constrained');ax.imshow(im);ax.axis('off');ax.set_title(title,fontsize=13)
        for b,label,diff in zip(bxs,r['labels'],r['difficult']):
            color='#777777' if diff else plt.get_cmap('tab20')((label-1)%20)
            ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,linewidth=1.5,linestyle='--' if diff else '-'))
            ax.text(b[0],b[1],names[label-1]+(' · difficult' if diff else ''),fontsize=10,color='white',bbox=dict(facecolor=color,alpha=.85,pad=1))
        return fig
    src=sources+[dict(path=(base/r['image']).relative_to(root).as_posix(),sha256=hashlib.sha256((base/r['image']).read_bytes()).hexdigest(),key=r['key'])]
    save(draw(image,boxes,'Ảnh gốc + nhãn XML'),'a2/preprocess_original_report.png',src,'Original CSV sample and annotation; no inference.')
    scale=min(600/min(image.size),1000/max(image.size));resized=image.resize((int(image.width*scale),int(image.height*scale)))
    factors=np.array([resized.width/image.width,resized.height/image.height]*2,dtype=np.float32)
    save(draw(resized,boxes*factors,'Resize: short 600, max 1,000'),'a2/preprocess_resize_report.png',src,'Deterministic resize and bbox factors exactly as notebook preprocessing example.')
    flipped=boxes.copy();flipped[:,[0,2]]=image.width-flipped[:,[2,0]]
    save(draw(ImageOps.mirror(image),flipped,'Horizontal flip: p=0.5 (train)'),'a2/preprocess_flip_report.png',src,'Deterministic demonstration of the notebook flip transform; no random augmentation run.')


def save_sample(fig, relative, source, method):
    path = assets / relative
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    manifest.append(dict(asset=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source=source, method=method))


def build_samples():
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "figure.dpi": 200})
    nb=json.loads((root/'source_code/A1.ipynb').read_text(encoding='utf-8'))
    md=''.join(nb['cells'][8]['outputs'][2]['data']['text/markdown'])
    rows=[[x.strip() for x in l.strip().strip('|').split('|')] for l in md.splitlines() if l.startswith('|')][2:]
    names=[r[1] for r in rows]
    base=root/'datasets/gtsrb/train'
    with (base/'samples.csv').open(encoding='utf-8',newline='') as f: records=list(csv.DictReader(f))
    for first,last,name in [(0,22,'a1/class_gallery_0_21_report.png'),(22,43,'a1/class_gallery_22_42_report.png')]:
        fig,axes=plt.subplots(6,4,figsize=(6.1,8.5),layout='constrained')
        sources=[]
        for ax in axes.flat:ax.axis('off')
        for label,ax in zip(range(first,last),axes.flat):
            r=next(r for r in records if int(r['label'])==label);p=base/r['image']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
            with Image.open(p) as im:ax.imshow(im)
            ax.set_title(f'ID {label}:\n'+textwrap.fill(names[label],18),fontsize=8.5,pad=2)
            sources.append(dict(path=p.relative_to(root).as_posix(),sha256=r['sha256'],class_id=label))
        save_sample(fig,name,sources,'Same first sample per class as notebook gallery; 43 names from saved class-count output; redrawn independently with report-sized text.')

    imfile=root/'datasets/fashion_mnist/val/images.npy';imgs=np.load(imfile,mmap_mode='r')
    for index,name in [(0,'shirt'),(31,'boot')]:
        fig,ax=plt.subplots(figsize=(2.3,2.3),layout='constrained');ax.imshow(imgs[index],cmap='gray',vmin=0,vmax=255);ax.axis('off')
        save_sample(fig,'e2/sample_'+name+'_compact.png',dict(path=imfile.relative_to(root).as_posix(),index=index,sha256=hashlib.sha256(imfile.read_bytes()).hexdigest()),'Original validation pixels; sample label/prediction are kept in the LaTeX subcaption.')

    # Draw the same image and scan rules with readable typography in compact panels.
    img=np.array(imgs[0]);flat=img.reshape(7,4,7,4).transpose(0,2,1,3).reshape(49,16)
    for kind,seq in [('rows',img),('columns',img.T),('patch',flat)]:
        fig,axes=plt.subplots(1,2,figsize=(4.6,3.4),layout='constrained')
        ax=axes[0];ax.imshow(img,cmap='gray',vmin=0,vmax=255);ax.set_title('Ảnh: hướng quét',fontsize=13)
        if kind=='rows':ax.annotate('',xy=(27,25),xytext=(27,2),arrowprops=dict(arrowstyle='->',color='#D55E00',lw=2));ax.axhline(8,color='#E69F00',lw=2)
        elif kind=='columns':ax.annotate('',xy=(25,27),xytext=(2,27),arrowprops=dict(arrowstyle='->',color='#D55E00',lw=2));ax.axvline(8,color='#E69F00',lw=2)
        else:
            for k in range(0,29,4):ax.axvline(k-.5,color='#E69F00',lw=.7);ax.axhline(k-.5,color='#E69F00',lw=.7)
            ax.annotate('',xy=(26,2),xytext=(2,2),arrowprops=dict(arrowstyle='->',color='#D55E00',lw=2))
        ax.set_xticks([]);ax.set_yticks([])
        ax=axes[1];ax.imshow(seq,cmap='gray',vmin=0,vmax=255,aspect='auto');ax.set(xlabel='Feature F',ylabel='Bước T');ax.set_title(f'T={seq.shape[0]}, F={seq.shape[1]}',fontsize=13);ax.tick_params(labelsize=11)
        save_sample(fig,'e3/representation_'+kind+'_compact.png',dict(path=imfile.relative_to(root).as_posix(),index=0,sha256=hashlib.sha256(imfile.read_bytes()).hexdigest()),'Original Fashion-MNIST validation image; rows/columns and non-overlapping 4x4 raster patch reshape exactly match notebook, no inference.')


if __name__ == "__main__":
    build_eda()
    build_samples()
    (assets / "report_figures.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(manifest)} independent static report figures.")
