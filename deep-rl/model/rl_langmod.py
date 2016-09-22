"""
rl_langmod.py

Core model definition file for a Deep-Q Network Language Model. The state-space input consists of
a bag-of-words context vector, and the action space input consists of a one-hot vector corresponding
to the target word (word following the context).
"""

class RLangmod():
    def __init__(self):
        """

        """