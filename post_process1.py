'''
train.py에 위치할 코드들

Invisible Brackets
172 \left.
71 \right.

()
229 \left(
137 \right)

{}
175 \left\{
145 \right\}

[]
171 \left[
190 \right]

|
50 \left|
27 \right|

||
73 \left\|
124 \right\|

'''

LEFT_BRAKETS = set([172, 229, 175, 171, 50,  73])
RIGHT_BRAKETS = set([71,  137, 145, 190,  27, 124])
MAX_ADD = 3
    
def add_balance(token, do_post):
    if do_post:
        if token in LEFT_BRAKETS:
            return -1
        if token in RIGHT_BRAKETS:
            return 1
    return 0

def id_to_string(tokens, data_loader, do_eval=0, do_post=0): # 0 Preds 1 -1 -1....
    result = []
    if do_eval:
        eos_id =  data_loader.dataset.token_to_id["<EOS>"]
        pad_id = data_loader.dataset.token_to_id["<PAD>"]
        sos_id = data_loader.dataset.token_to_id["<SOS>"]
        pad_id2 = -1
        ignore_ids = {
            pad_id : 1,
            sos_id : 1,
            pad_id2 : 1,
        }
    for example in tokens:
        string = ""
        balance = 0
        if do_eval:  # 계산 용도 => score 와 관련이 있다.
            for token in example:
                token = token.item()
                if token == eos_id: # <EOS>만나면 종료한다.
                    break
                if token not in ignore_ids: # eos 외 무시할 id들을 체크한다.
                    balance += add_balance(token, do_post)
                    string += data_loader.dataset.id_to_token[token] + " "
        else: # display 용도.
            for token in example:
                token = token.item()
                if token != -1: # 길이 채우기 위한 -1만 무시한다.
                    string += data_loader.dataset.id_to_token[token] + " "
                    balance += add_balance(token, do_post)
        if balance:
            if balance > 0:
                string = '\\left. ' * min(MAX_ADD, balance) + string
            else:
                string += '\\right. ' * min(MAX_ADD, -1 * balance)
        result.append(string)
    return result