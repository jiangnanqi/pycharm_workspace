

def get_description():
    poss = ['rain', 'snow', 'sleet', 'fog', 'sun', 'who knows']
    from random import choice
    return choice(poss)
get_description()