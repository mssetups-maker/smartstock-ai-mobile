"""Optional free local LLM via user-run Ollama. Never uploads keys or uses paid API."""
from __future__ import annotations
import json
import urllib.request


def explain_with_ollama(context:str,model:str='gemma3:4b',endpoint='http://localhost:11434') -> str:
    """Only localhost; enforce no public endpoint and no automatic outbound message."""
    if endpoint not in ('http://localhost:11434','http://127.0.0.1:11434'):
        raise ValueError('안전을 위해 로컬 Ollama 주소만 허용합니다.')
    request_data={'model':model,'stream':False,'options':{'temperature':0.2},
          'messages':[{'role':'system','content':
             '너는 금융 분석 근거를 설명한다. 제공된 수치만 사용하고, 부족한 정보는 모른다고 말한다. '
             '모델 상승확률을 확정적 투자 성공률로 표현하지 말 것. 옵션 OI를 포지션 방향으로 단정하지 말 것. '
             '한국어로 기술, 옵션, 뉴스, 재무, 선택한 예측기간의 위험과 데이터 지연 가능성을 요약할 것.'},
                      {'role':'user','content':context}]}
    body=json.dumps(request_data,ensure_ascii=False).encode('utf-8')
    req=urllib.request.Request(endpoint+'/api/chat',data=body,headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=90) as r: answer=json.load(r)
    return str(answer.get('message',{}).get('content','응답이 비어 있습니다.'))
