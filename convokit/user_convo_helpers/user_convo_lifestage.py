from convokit.transformer import Transformer

class UserConvoLifestage(Transformer):
    
    '''
		Transformer that, for each user in a conversation, computes the lifestage of the user in that conversation. For instance, if lifestages are 20 conversations long, then the first 20 conversations a user participates in will be in lifestage 0, and the second 20 will be in lifestage 1.

		Assumes that `corpus.organize_user_convo_history` has already been called.

		:param lifestage_size: size of the lifestage 
		:param output_field: name of user conversation attribute to output, defaults to "lifestage"
    '''

    def __init__(self, lifestage_size, output_field='lifestage'):
        self.output_field = output_field
        self.lifestage_size = lifestage_size
        
    def transform(self, corpus):
        for user in corpus.iter_users():
            if 'conversations' not in user.meta:
                continue
            for convo_id, convo in user.meta['conversations'].items():
                corpus.set_user_convo_info(user.name, convo_id, self.output_field,
                                          int(convo['idx']//self.lifestage_size))
        return corpus