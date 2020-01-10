def warn(text: str):
    # Pre-pends a red-colored 'WARNING: ' to [text].
    # :param text: Warning message
    # :return: 'WARNING: [text]'
    print('\033[91m'+ "WARNING: " + '\033[0m' + text)