# Quotable: Making Data Come Alive!
Quotable is a data assistant that helps consumers and enterprises query their datasets in
natural language. Quotable uses cutting-edge transformers (BERT, Google TAPAS for tabular
data) and layers our innovation called "AQuA" (Augmentation for Question Answering) to
provide simple answers to queries in English.

Try Quotable on our website: [quotableai.com/](http://quotableai.com/)

Generally, you will need to be an expert at Structured Query Language (SQL) to query tables. SQL can be complex and difficult to learn. With Quotable, you can query tables in plain English! It is very simple to use - Just upload your tables, ask your questions, and make your data talk to you. Quotable is perfect for you if you are experiencing "SQL pains" such as below:

* You have to generate instantaneous insights for your execs as a part of an Enterprise BI Team
* You are a lay consumer who wants insights into data, but don't have SQL expertise

### Quotable for Enterprise BI
As an executive in a large enterprise, you want your BI team to provide quick insights on revenues for your product lines. However, every time you ask a question, the BI team needs another week to get back to you. With Quotable, you don't need to wait, and you don't need to know SQL either! Just ask your question in English and get an instantaneous response.

### Quotable for Consumers
You are a basketball enthusiast, say, and no stats expert. You want to know how well Lebron James did in the 2020 season compared to others. You do a Google search, but there are no straight answers. What do you do? With Quotable, you just type your question in, and we do the rest of the magic and get you a straight answer.

## Key Technologies
Quotable uses two main technologies for its product:

1. __Augmentation for Question Answering:__ AQuA is the patentable innovation which prepares user tables for querying. AQuA has 3 major functions: splitting larger tables into smaller ones to feed our NLP model, preprocessing and modification of table rows and columns, and augmenting existing rows and columns with derived rows and columns.

2. __Pretrained BERT model for Question Answering:__  We used a BERT model which is pretrained with table data to perform natural language based Question Answering. For our first version of Quotable, we are using a Google model called TAPAS [1].


## References

[1] Jonathan Herzig, Pawel Krzysztof Nowak, Thomas Müller, Francesco Piccinno, and Julian Eisenschlos. 2020. [TaPas: Weakly supervised table parsing via pre-training](https://www.aclweb.org/anthology/2020.acl-main.398/). In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4320– 4333, Online. Association for Computational Linguistics.
