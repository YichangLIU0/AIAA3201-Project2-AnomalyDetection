from pathlib import Path
import fitz
from PIL import Image,ImageDraw
r=Path(__file__).resolve().parents[1];d=fitz.open(r/'reports/main.pdf');out=r/'reports/preview';out.mkdir(exist_ok=True);thumbs=[]
for i,p in enumerate(d):
 pix=p.get_pixmap(matrix=fitz.Matrix(1.2,1.2));pix.save(out/f'page_{i+1}.png')
 im=Image.open(out/f'page_{i+1}.png').convert('RGB');im.thumbnail((306,396));tile=Image.new('RGB',(326,426),'#cccccc');tile.paste(im,(10,20));ImageDraw.Draw(tile).text((10,4),str(i+1),fill='black');thumbs.append(tile)
 print(i+1,len(p.get_text()),p.get_text()[:65].replace('\n',' '))
canvas=Image.new('RGB',(978,426*((len(thumbs)+2)//3)),'white')
for i,im in enumerate(thumbs):canvas.paste(im,((i%3)*326,(i//3)*426))
canvas.save(out/'contact.png');print('PAGES',len(d))
