from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
r=Path(__file__).resolve().parents[1]
fig,ax=plt.subplots(figsize=(3.4,1.75));ax.set_xlim(0,3);ax.set_ylim(0,3);ax.axis('off')
labels=[['8 refs x 8 views\n224 or 448','Frozen DINOv2','4096 vectors\n6 MiB bank'],['Original query','Resize + DINO\n16x16 / 32x32','Nearest match\nscore + map'],['40 disjoint\nnormal images','Normal quantiles\nNG / review','OK / REVIEW\nor NG']]
for row in range(3):
 for col in range(3):
  x=col+.06;y=2.3-row
  ax.add_patch(FancyBboxPatch((x,y),.86,.52,boxstyle='round,pad=.025',linewidth=.6,edgecolor='#37667b',facecolor='#eef6fa'))
  ax.text(x+.43,y+.26,labels[row][col],ha='center',va='center',fontsize=6.2)
  if col<2:ax.annotate('',xy=(col+1.025,y+.26),xytext=(x+.9,y+.26),arrowprops={'arrowstyle':'->','lw':.7})
for row in [0,1]:ax.annotate('',xy=(2.49,1.3-row+.58),xytext=(2.49,2.3-row-.03),arrowprops={'arrowstyle':'->','lw':.7})
fig.subplots_adjust(left=0,right=1,bottom=0,top=1);fig.savefig(r/'reports/figures/lowmiss_pipeline.pdf');plt.close(fig)
