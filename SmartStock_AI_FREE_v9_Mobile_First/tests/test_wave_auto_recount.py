import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'SmartStock_AI_FREE_v9_Mobile_First'))
import wave_auto_recount as w


def test_automatic_recount_uses_valid_alternative_not_invalid_original(monkeypatch):
    def find(prices,order,lookback):
        if order==5:
            return {'status':'판정 보류','nodes':[],'notes':'invalid'}
        if order==3:
            return {'status':'후보','order':3,'nodes':[{'파동':'3'}], 'notes':'good'}
        return {'status':'판정 보류','nodes':[],'notes':'none'}
    monkeypatch.setattr(w,'estimate_wave',find)
    got=w.estimate_with_recount(None,requested_order=5)
    assert got['status']=='후보'
    assert got['requested_order']==5
    assert got['order']==3 and got['auto_recounted']
    assert got['recount_scales_tried']==[5,3]


def test_no_valid_sensitivity_yields_no_labels(monkeypatch):
    monkeypatch.setattr(w,'estimate_wave',lambda p,order,lookback: {'status':'판정 보류','nodes':[], 'notes':'bad'})
    out=w.estimate_with_recount(None,requested_order=5)
    assert out['status']=='판정 보류'
    assert not out['nodes']
    assert out['recount_scales_tried']==[5,3,8,12]


def test_valid_selected_sensitivity_stays_the_same(monkeypatch):
    def find(prices,order,lookback):
        return {'status':'후보','order':order,'nodes':[{'파동':'3'}],'notes':'valid'}
    monkeypatch.setattr(w,'estimate_wave',find)
    got=w.estimate_with_recount(None,requested_order=8)
    assert got['order']==8 and not got['auto_recounted']
    assert got['recount_scales_tried']==[8]
