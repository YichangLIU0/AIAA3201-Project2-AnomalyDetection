"""Produce normal-training-only examples; never label them formal real-defect results."""
import argparse
from pathlib import Path
from PIL import Image
from shared_protocol.data import load_manifest
from shared_protocol.protocol import CATEGORIES,inside,save_json,sha256
from methods.part1.template import read_rgb,preprocess
from .synthetic import CONFIG,generate,estimate_foreground


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/part3_c_preparation'))
    a=p.parse_args(); records=[]
    for ci,category in enumerate(CATEGORIES):
        manifest=load_manifest(category)
        # Fixed first training-pool item, after removing all calibration images.
        relative=manifest['normal_training_pool'][0]
        if relative in manifest['calibration'] or relative in {r['path'] for r in manifest['test']}:
            raise ValueError('Training-only preview selection failed')
        original=preprocess(read_rgb(inside(a.data_root,relative)))
        folder=a.output/'examples'/category;folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(original).save(folder/'normal.png')
        Image.fromarray(estimate_foreground(original).astype('uint8')*255).save(folder/'estimated_foreground.png')
        for mode in CONFIG['modes']:
            modified,mask,meta=generate(original,20261111+ci,mode)
            Image.fromarray(modified).save(folder/f'{mode}.png')
            Image.fromarray(mask.astype('uint8')*255).save(folder/f'{mode}_mask.png')
            records.append({'category':category,'normal_training_path':relative,
                'source_sha256':sha256(inside(a.data_root,relative)),
                'example':(folder/f'{mode}.png').relative_to(a.output).as_posix(),
                'mask':(folder/f'{mode}_mask.png').relative_to(a.output).as_posix(),
                'synthetic_sha256':sha256(folder/f'{mode}.png'),**meta})
    save_json(a.output/'preparation.json',{'status':'preparation_only_not_formal_experiment',
        'config':CONFIG,'source_sha256':sha256(Path(__file__).with_name('synthetic.py')),
        'backbone_loaded':False,'head_trained_on_dino':False,'real_defect_transfer_evaluated':False,
        'examples':records})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(6,6,figsize=(15,15),layout='constrained')
    names=['normal','estimated_foreground','cutpaste','cutpaste_mask','texture','texture_mask']
    for category,row in zip(CATEGORIES,axes):
        for ax,name in zip(row,names):
            with Image.open(a.output/'examples'/category/f'{name}.png') as image:
                ax.imshow(image,cmap='gray',vmin=0,vmax=255)
            ax.set_title(category+' '+name,fontsize=8);ax.axis('off')
    fig.savefig(a.output/'synthetic_contact.png',dpi=110);plt.close(fig)
    print('SYNTHETIC_PREVIEWS_READY',len(records))


if __name__=='__main__':main()
