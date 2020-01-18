from setuptools import setup, find_packages

setup(
    name="convokit",
    author="Jonathan P. Chang, Caleb Chiam, Liye Fu, Andrew Wang, Justine Zhang, Cristian Danescu-Niculescu-Mizil",
    author_email="cristian@cs.cornell.edu",
    url="https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit",
    description="Cornell Conversational Analysis Toolkit",
    version="2.3.0",
    packages=["convokit",
                "convokit.bag_of_words",
                "convokit.classifier",
                "convokit.coordination",
                "convokit.fighting_words",
                "convokit.forecaster",
                "convokit.hyperconvo",
                "convokit.model",
                "convokit.paired_prediction",
                "convokit.politeness_api",
                "convokit.politeness_api.features",
                "convokit.politenessStrategies",
                "convokit.phrasing_motifs",
                "convokit.prompt_types",
                "convokit.ranker",
                "convokit.text_processing",
                "convokit.userConvoDiversity",
                "convokit.user_convo_helpers",
              ],
    package_data={"convokit": ["data/*.txt"]},
    install_requires=[
        "matplotlib>=3.0.0",
        "pandas>=0.23.4",
        "msgpack-numpy==0.4.3.2",
        "spacy==2.0.12",
        "scipy>=1.1.0",
        "scikit-learn>=0.20.0",
        "nltk>=3.4",
        "dill==0.2.9"
    ],
    extras_require={
        'neuralnetwork': ["torch>=0.12"],
        'cleantext': ["clean-text>=0.1.1"]
    }
)
