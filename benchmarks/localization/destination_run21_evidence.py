"""Render the fresh-lap transfer result without treating lap offsets as errors."""
import json
from pathlib import Path
import cv2
import numpy as np
from benchmarks.localization.destination_run21 import ROOT, TARGETS

def main():
    rows=json.loads((ROOT/'transfer-rows.json').read_text());frames=json.loads((ROOT/'frames.json').read_text());times=np.array([r['time_s'] for r in frames]);images=np.load(ROOT/'gray.npy',mmap_mode='r');views=json.loads((ROOT/'first-lap-views.json').read_text())
    intervals=[(41,82),(82,124),(124,166),(166,208),(208,240)];summary=[]
    canvas=np.full((220+4*200,1600,3),247,np.uint8)
    cv2.putText(canvas,'Session 21: first-lap memory on later laps',(25,38),cv2.FONT_HERSHEY_SIMPLEX,.85,(30,30,30),2)
    for j,text in enumerate(['Four manually identified landmark groups; 15 prepared views; no neural training.',
        'Each trace uses its own min/max horizontal image bearing. Gaps = unavailable, visibility unreviewed.',
        'This is an OBSERVATION AVAILABILITY chart, not verified target lifetime or steering accuracy.',
        'All later frames are evaluated. Scene and lighting change; target absence also contributes to gaps.']):
        cv2.putText(canvas,text,(25,70+j*24),cv2.FONT_HERSHEY_SIMPLEX,.50,(40,40,40),1)
    x0,x1=260,1550
    def xx(t):return int(x0+(t-41)/(240-41)*(x1-x0))
    for k,c in enumerate(TARGETS):
        rr=[r for r in rows if r['target']==c['name']];valid=[r for r in rr if r['aim'] is not None];y0=190+k*200;yb=y0+138;yt=y0+28
        b=[r['bearing'] for r in valid];lo=min(b) if b else 0;hi=max(b) if b else 0
        cv2.putText(canvas,c['name'],(20,y0+50),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1)
        cv2.putText(canvas,'bearing %.3f .. %.3f'%(lo,hi),(20,y0+76),cv2.FONT_HERSHEY_SIMPLEX,.38,(70,70,70),1)
        cv2.putText(canvas,'%d views; %d/%d accepted'%(len(views[c['name']]),len(valid),len(rr)),(20,y0+99),cv2.FONT_HERSHEY_SIMPLEX,.38,(70,70,70),1)
        cv2.rectangle(canvas,(x0,yt),(x1,yb),(220,220,220),1)
        for t in np.arange(41,240,2):cv2.line(canvas,(xx(t),yb+7),(xx(t)+3,yb+7),(145,145,145),1)
        prev=None
        for r in rr:
            if r['aim'] is None:prev=None;continue
            pos=(xx(r['time_s']),int(yb-(r['bearing']-lo)/max(hi-lo,1e-9)*(yb-yt)))
            if prev is not None:cv2.line(canvas,prev,pos,(105,132,0),2)
            cv2.circle(canvas,pos,1,(105,132,0),-1);prev=pos
        for a,z in intervals:
            selected=[r for r in rr if a<=r['time_s']<z];accepted=[r for r in selected if r['aim'] is not None]
            # Unavailability between first/last accepted: explicitly not a physical-visibility truth.
            longest=0.;start=None
            if accepted:
                middle=[r for r in selected if accepted[0]['time_s']<=r['time_s']<=accepted[-1]['time_s']]
                for r in middle:
                    if r['aim'] is None and start is None:start=r['time_s']
                    if r['aim'] is not None and start is not None:longest=max(longest,r['time_s']-start);start=None
            summary.append(dict(target=c['name'],interval_s=[a,z],frames=len(selected),accepted=len(accepted),first_s=accepted[0]['time_s'] if accepted else None,last_s=accepted[-1]['time_s'] if accepted else None,longest_internal_unavailable_s=longest if accepted else None))
            cv2.line(canvas,(xx(a),yt),(xx(a),yb),(195,195,195),1)
            cv2.putText(canvas,'%d/%d'%(len(accepted),len(selected)),(xx(a)+5,yb+33),cv2.FONT_HERSHEY_SIMPLEX,.40,(50,50,50),1)
            cv2.putText(canvas,'%d s'%a,(xx(a)+5,yb+56),cv2.FONT_HERSHEY_SIMPLEX,.40,(50,50,50),1)
    cv2.imwrite(str(ROOT/'availability-bearing.png'),canvas)
    sheet=np.full((4*205,6*320,3),247,np.uint8)
    for k,c in enumerate(TARGETS):
        for j,offset in enumerate([0,41,82,124,166,208]):
            target=c['seed']+offset
            if target>times[-1]:continue
            i=int(np.argmin(abs(times-target)));im=cv2.cvtColor(images[i],cv2.COLOR_GRAY2BGR)
            if j==0:
                a,b,d,e=c['roi'];cv2.rectangle(im,(a,b),(d,e),(0,200,255),2)
            elif rows:
                r=next((r for r in rows if r['target']==c['name'] and r['frame']==i),None)
                if r and r['aim'] is not None:cv2.drawMarker(im,tuple(np.rint(r['aim']).astype(int)),(0,200,255),cv2.MARKER_CROSS,20,2)
            sheet[k*205+25:(k+1)*205,j*320:(j+1)*320]=cv2.resize(im,(320,180))
            cv2.putText(sheet,'%s %.2fs'%('TEACH' if j==0 else 'QUERY',times[i]),(j*320+5,k*205+17),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1)
    cv2.imwrite(str(ROOT/'lap-comparison.jpg'),sheet)
    (ROOT/'transfer-summary.json').write_text(json.dumps(summary,indent=2))
    lifetime(rows, times)
    print(json.dumps(summary,indent=2))

def lifetime(rows,times):
    # Native-frame review recorded before rendering; bounds are not estimator exits.
    reviewed=[('market_canopy',57,63,65),('stair_top',70,75,77),('corner_lamp',76,79,80),('after_turn_arch',79.1,None,None)]
    episodes=[];lane_ends=[]
    for name,start,present,absent in reviewed:
        accepted=[r for r in rows if r['target']==name and start<=r['time_s']<86 and r['aim'] is not None]
        if not accepted:continue
        end_present=None if present is None else float(times[np.argmin(abs(times-present))]);end_absent=None if absent is None else float(times[np.argmin(abs(times-absent))]);end=end_absent or 86
        accepted=[r for r in accepted if r['time_s']<=end];begin=accepted[0]['time_s']
        lane=next((i for i,z in enumerate(lane_ends) if z<begin),len(lane_ends))
        if lane==len(lane_ends):lane_ends.append(end)
        else:lane_ends[lane]=end
        episodes.append(dict(target=name,initial_acquisition_s=begin,last_success_s=accepted[-1]['time_s'],last_verified_present_s=end_present,first_verified_absent_s=end_absent,final_appearance_s=None,lane=lane,review='Manual native-scene neighborhood review; exact exit and individual feature identities unresolved',bearing_min=min(r['bearing'] for r in accepted),bearing_max=max(r['bearing'] for r in accepted)))
    (ROOT/'reviewed-episodes.json').write_text(json.dumps(episodes,indent=2))
    out=np.full((570,1600,3),250,np.uint8);x0,x1=80,1550;yt,yb=180,335
    def x(t):return int(x0+(t-57)/29*(x1-x0))
    cv2.putText(out,'Target lifetime timeline | reviewed second-lap passage and turn',(25,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(25,25,25),2)
    for j,text in enumerate(['One lane reused only after a reviewed first-absent frame. Target names move with their episodes.',
        'Teal = image observation; red hatching = lost; gray boundary = exit bracket / unverified end.',
        'Each target uses its own bearing min/max (image half-width units). No lap residual or XY error.']):cv2.putText(out,text,(25,78+j*25),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
    for e in episodes:
        a=e['initial_acquisition_s'];b=e['first_verified_absent_s'] or 86;known=e['last_verified_present_s'];xa,xb=x(a),x(b)
        cv2.rectangle(out,(xa,yt),(xb,yb),(220,235,245),-1)
        for xx in range(xa,xb,10):cv2.line(out,(xx,yb),(min(xx+12,xb),yt),(170,170,230),1)
        if known is not None:cv2.rectangle(out,(x(known),yt),(xb,yb),(223,223,223),-1)
        rr=[r for r in rows if r['target']==e['target'] and a<=r['time_s']<=b];prev=None
        for index,r in enumerate(rr):
            if r['aim'] is not None:
                end=rr[index+1]['time_s'] if index+1<len(rr) else r['time_s']+.04
                cv2.rectangle(out,(x(r['time_s']),yt+1),(x(min(end,b)),yb-1),(226,241,228),-1)
        for r in rr:
            if r['aim'] is None:prev=None;continue
            pos=(x(r['time_s']),int(yb-10-(r['bearing']-e['bearing_min'])/max(e['bearing_max']-e['bearing_min'],1e-9)*(yb-yt-20)))
            if prev is not None:cv2.line(out,prev,pos,(95,130,0),2)
            cv2.circle(out,pos,2,(95,130,0),-1);prev=pos
        cv2.rectangle(out,(xa,yt),(xb,yb),(70,70,70),1);cv2.circle(out,(xa,yb),5,(95,130,0),-1)
        cv2.putText(out,e['target'],(xa,yt-12),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1)
        cv2.putText(out,'%.3f..%.3f'%(e['bearing_min'],e['bearing_max']),(xa,yb+23),cv2.FONT_HERSHEY_SIMPLEX,.39,(30,30,30),1)
        if known is None:cv2.putText(out,'end unknown >',(max(xa,xb-145),yb+48),cv2.FONT_HERSHEY_SIMPLEX,.40,(40,40,40),1)
    for t in range(57,87,2):cv2.putText(out,str(t)+'s',(x(t)-8,418),cv2.FONT_HERSHEY_SIMPLEX,.43,(30,30,30),1)
    cv2.putText(out,'Visible after tracking failed: canopy at 63.04 s; stair-top neighborhood at 75.03 s; lamp at 79.01 s.',(25,467),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
    cv2.putText(out,'First absent brackets: canopy by 64.98 s; stair top by 76.97 s; lamp by 80.01 s. Facade end unverified.',(25,497),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
    cv2.putText(out,'Review is sparse and scene-group based. Exact disappearance and per-feature identity are not certified.',(25,527),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
    cv2.imwrite(str(ROOT/'target-lifetime.png'),out)

if __name__=='__main__':main()
