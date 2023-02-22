# How to create (historical) Dutch transformer models
## A working paper

### Introduction

The rise of transformer language models such as [BERT](https://arxiv.org/abs/1810.04805) has opened up possibilities to use contextualized word embeddings for downstream text processing tasks. This includes applications in humanities research. However, the methods to properly use these models in a humanities and, particularly, a historical context are still very much under development. The aim of this working paper is to present state-of-the-art guidelines to use transformers to study change over time.



| chapter      | Name    |
| ------------ | ------- |
| Introduction |         |
| 1            | Laura   |
| 2            | Carsten |
| 3            | Parisa  |
| 4             |    Flavio     |
|       5       |     Pim    |
| Conclusion            |   |


### 1.	What are suitable corpora to base Dutch historical language models on?
The are various Dutch language models available to train language models on (XXX). However, if the language model will be used to analyse historical texts and to answer questions related to language over different historial time frames, this sets different requirements to the corpus that is used in training the model. In this section, the properties that are required or should be taken into account when training historical Dutch langues models are discussed.

#### Corpus properties
One of the important properties that affect how succesfull a trained model is, is the size of the training corpus. Naturally, the question then rises: what corpus size is required for my model? The answer however, is not straigtforward, as it depends on the task and context of the language model (XXX): the pretraining method, the model architecture and the application. XXX EXAMPLES AND REFERENCES. In the objective considered here the language model will use contextualized word embeddings for pretraining, and a transformer model architecture. As the application of the model is to analyse text in historical context, the corpus requires XXX.
- required size
- balance/bias
- representativeness
- data quality
- stability in word embeddings(?)

#### Available corpora
- which, where,license

*Moderator: Laura Ootes*



Discussion Ideas:

* (Required) Corpus size
    * Requirements for different tasks, contexts
        * Depends on architecture
    * Pre-training from scratch: what is the need to balance different input corpora
        * Transfer-dilution Trade-off and Curse of Multilinguality in multilingual LMs (Conneau et al. 2020): for a fixed model capacity (model size, number of parameters), low-resource languages benefit from related high-resource languages, but/and adding more languages to training decreases performance after a point. 
    * Papers:
        * Li, Z., Wallace, E., Shen, S., Lin, K., Keutzer, K., Klein, D., & Gonzalez, J. (2020, November). Train big, then compress: Rethinking model size for efficient training and inference of transformers. In International Conference on Machine Learning (pp. 5958-5968). PMLR. http://proceedings.mlr.press/v119/li20m/li20m.pdf 
        * ?

* Licensing issues
    * How restricted are relevant corpora
        * Delpher: no redistributing, also no BOW per document (depends on publication date of document)
    * (How) can we work with them anyway?
        * Prefer to not work with sensitive data?
        * what kinds of derived data can be used/distributed?
        * SANE (SURF): 
            * an environment to run your scripts/software on protected data via a secure environment
            * https://www.surf.nl/en/news/sane-secure-data-environment-for-social-sciences-and-humanities
        * Can you share models trained on restricted data?
* Corpus balance/bias and representativeness
    * Diversity of domain, genre, topic, authors, style, time period
    * How to mix? What are important parameters?
    * Requirements/recommendations for metadata
        * Required
            * Time period
            * Language domain - Genre / style (news paper, parlementairy etc.)
            * Size (tokens + bytes)
            * Source
            * Identifying metadata
            * License
        * Recommended: anything that can be inferred from the data
            * Document type
            * Size (number of files, lines)
            * Quality of OCR/HTR
            * URL / DOI
    * Requirements/recommendations for data
        * File format: text + HTR/OCR output format (with e.g. coordinates)
        * Date issued
* List of links of available data sets

| Title | License | URL | Period | Genre | Size (GB) |
|-|-|-|-|-|-|
|Delpher newspapers| Permission needed | https://www.delpher.nl/over-delpher/delpher-open-krantenarchief/wat-zit-er-in-het-delpher-open-krantenarchief | 1618-1879 | newspapers, books, periodicals
|DBNL|Permission needed|https://www.dbnl.org | 1550-present | plays, poetry, novels, letters |
|VOC+WIC+Notarial deeds HTR |CC-BY-4.0|https://doi.org/10.5281/zenodo.6414086 | 1637-1792 | reports, correspondence |
|Huygens Resources||https://resources.huygens.knaw.nl |  | 700-present | correspondence, administrative, ... |
|Resolutions States General || https://republic.huygens.knaw.nl | 1576-1796 | Government decisions, Administrative |
|NederLab||||
|Amsterdam City Archives (HTR + Ground Truth)|PD (?)|https://transkribus.eu/r/amsterdam-city-archives | 1578-1811 | Notarial deeds, Administrative|
| NIBG Radio and television ||| 1870-present | radio transcripts, subtitles |
| SONAR || https://taalmaterialen.ivdnt.org/download/tstc-sonar-corpus/ | | various |
| EuroParl (same as Staten Generaal Digitaal) || https://www.statmt.org/europarl/ | | Political discussions, transcripts, Administrative |
| OSCAR (based on common crawl) ||https://oscar-project.org/||
| TwNC (multimedia news dataset) ||||
| Gysseling? Corpus Middel Nederlands / Corpus Oud Nederlands ||||
| Woordenboek Nederlandse Taal |||... | dictionary |
| Staten Generaal Digitaal | CC-0-1.0 | https://data.overheid.nl/dataset/staten-generaal-digitaal---koninklijke-bibliotheek | 1814-present | State publications |
| Taalmaterialen IvdNT | Varies | https://taalmaterialen.ivdnt.org/document-tag/corpus/ | | |


* How to deal with data of different quality?
    * Depends on the task
    * Does it make sense to train models with multiple sizes of lower quality data? Is it 'garbage in, garbage out', or can this help with messy data?


* References
    * M. van der Wees. [What’s in a Domain?  Towards Fine-Grained Adaptation for Machine Translation](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/van-der-wees-phd-thesis-2017.pdf).  PhD thesis, Universiteit van Amsterdam, 2017
    * [Conneau et al. 2020. Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/pdf/1911.02116v2.pdf)
    * [Abadji et al. 2022. Towards a Cleaner Document-Oriented Multilingual Crawled Corpus](https://arxiv.org/pdf/2201.06642.pdf)
    * [Ortis Suárez et al 2019. Asynchronous Pipeline for Processing Huge Corpora
on Medium to Low Resource Infrastructures](https://arxiv.org/pdf/2201.06642.pdf): OSCAR



These are things that came up in the group on evaluation; Flavio has added them here but just delete them if they do not fit/are already covered
- Oxford English Dictionary
    - detecting word sense
    - word sense disambiguation
    - can dictionaries be used to measure the change in the word sense?
        - unclear what it means if a word is not documented
        - a [plot](https://github.com/emanjavacas/macberth-eval/blob/main/images/period-plots.pdf) documenting the distribution of senses over time based on OED data (possibility to use for lexical change?)
- WNT: Woordenboek der Nederlandsche Taal
- [LChange22](https://languagechange.org/events/2022-acl-lchange/)
- Stability of models [NAACL 2018](https://aclanthology.org/N18-1190/)



### 2. What are suitable model architectures and sizes?

#### Overview

Since the introduction of Transformer-based language models like BERT (Devlin et al., 2019), numerous model variations based on the similar architecture have been developed.
Despite the various differences between Bert, RoBerta (Liu et al., 2019) and their many other derivations (e.g. He et al., 2021), pracitioners oftentimes do not see noticable differences originating from choosing one of those flavours over the other.

The performance of a language model depends on many aspects that are not related to architecture at all, most importantly the input data and the specifics of a downstream task (Manjavacas Arevalo & Fonteyn, 2021).
Training large language models is expensive though, so the potential benefits of optimising the architecture empirically is not expected to justify the additional costs.
Consequently, for instance GysBert (Manjavacas Arevalo & Fonteyn, 2022), have chosen to "closely follow the BERT-base uncased" architecture.

Another consideration against always applying the current state-of-the-art architecture is the long development time: the entire process from data collection via implementation and training can take several years.
During that time new best-performing architectures may have been developed, whereas flaws in current popular architectures might have been uncovered.

Instead, informal approaches have shown to be more practical for finding a suitable architecture. For instance, have there already models been trained for similar use cases and/or on similar data in terms of domain, size etc.?

In the context of historic language models, however, another important question is: should the temporal aspect be encoded into the model explicitly?
Rosin et al. (2022) add an additional attention matrix to encode the chronological information, whereas TempoBERT (Rosin & Radinsky, 2022) apply a standard BERT architecture, but manipulate the text so that it contains temporal information.
They show that this leads to performance improvements downstream tasks such as semantic shift detection and sentence time prediction benefit significantly from temporal information explicitly added to the texts.

Another direction that might be relevant in future research leads towards models developed for other applications.
Stable Diffusion (Rombach et al., 2022) is a generative text-to-image model; similar techniques could be applied for, possibly application-specific language modelling and/or multimodel models.
However, it remains currently unclear how noise can be added to text in a way as required for the diffusion model training works as desired.

In conclusion there is no generic answer in finding the optimal model architecture per use case.
This would require a dedicated large-scale research project, while the practical benefit remains unclear.

#### Pre-processing and Input Data

As outlined above, factors other than model architecture play important roles for the performance of a language model.
For one, the choice of the tokenization in terms of both design and size seems to be important, whereas there has not been specific research on the exact impact of different tokenizers on downstream tasks.
Intuitively, it seems clear though that a tokenizer introduces a specific kind of bias by producing a specific set of tokens on which a model is trained.

Xue et al. (2022) demonstrate that token-free, "byte-level models are competitive with their token-level counterparts", and "characterize the trade-offs in terms of parameter count, training FLOPs, and inference speed."
Clark et al. (2022) tackle the "linguistic pitfalls of tokenization" with a character-based, token-free tokenization approach.

Another related factor is normalization in the pre-processing pipeline.
Especially historic texts, oftentimes digitized automatically through optical character recognition (OCR) or handwritten text recognition (HTR), typically contain numerous errors.
Apart from OCR and HTR errors, spelling variations are very common in historic texts.
Spelling rules for Dutch have established only in 1888, and changed several times since.
Before that, variations are even larger; for instance, capitalization was not consistent and different regions and time periods used to spell according to their own standards.

However, practitioners advice against normalization of input text before training a language model, because that also removes the model's robustness against variations that occur in unseen texts.

Dealing with historic language data includes a specific aspect: almost all larger datasets for Dutch have already been digitized.
Therefore, new data is unlikely to appear in the foreseeable future -- although its quality might improve for instance through improved OCR and HTR technology.

In order to get a sense for the data requirements for specific tasks or domains, it is important to understand of the specifics of the data to be used:

- How homogenous is the data?
- Which domain(s) does it cover?
- What task(s) should the model solve?

Quality measures such as perplexity, lexical variation, and stylistic variation can help understanding the distribution and homogenity of the data.

Conneau et al. (2020) investigate the impact of data distribution particularly considering low-resource languages in the context of multi-lingual models, showing ways to train performant models effectively even for languages and/or domains for which little data is available.

#### Computation

Computational costs for training a model fluctuates heavily, again depending on specifics of model architecture and input data size.
In particular the latter is especially relevant, as pointed out by Kaplan et al., 2020.
They also analyse other aspects and inter-dependent factors regarding their respective impact on model training times and costs.


Smaller input data sets can likely be handled by, for instance, university-owned GPU machines which are, however, too slow (and overloaded) to train large-scale models such as GysBert (Manjavacas Arevalo & Fonteyn, 2022).

Commercial services can handle those large workloads, but are expensive.
Currently, using a TPU machine as provided by the Google Cloud Engine, costs approximately EUR 2000 per week; the approximate duration of training a large-scale language model.
These cost might decrease in the future, while computation speed might increase.
Anyway, training a large-scale language model using commercial offerings is too expensive for the budget of many research projects.

In the context of this project (Semantics of Sustainability), the NL eScience Center currently investigates ways of applying the Dutch National Supercomputer Snellius effectively and efficiently for scientific purposes.

#### References

Clark, Jonathan H., Dan Garrette, Iulia Turc, and John Wieting. “Canine: Pre-Training an Efficient Tokenization-Free Encoder for Language Representation.” Transactions of the Association for Computational Linguistics 10 (January 31, 2022): 73–91. https://doi.org/10.1162/tacl_a_00448.

Conneau, Alexis, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. “Unsupervised Cross-Lingual Representation Learning at Scale.” arXiv, April 7, 2020. https://doi.org/10.48550/arXiv.1911.02116.

Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. “BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding.” ArXiv:1810.04805 [Cs], May 24, 2019. http://arxiv.org/abs/1810.04805.

He, Pengcheng, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. “DeBERTa: Decoding-Enhanced BERT with Disentangled Attention.” arXiv, October 6, 2021. https://doi.org/10.48550/arXiv.2006.03654.

Kaplan, Jared, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. “Scaling Laws for Neural Language Models.” arXiv, January 22, 2020. https://doi.org/10.48550/arXiv.2001.08361.

Liu, Yinhan, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. “RoBERTa: A Robustly Optimized BERT Pretraining Approach.” arXiv, July 26, 2019. https://doi.org/10.48550/arXiv.1907.11692.

Manjavacas Arevalo, Enrique, and Lauren Fonteyn. “MacBERTh: Development and Evaluation of a Historically Pre-Trained Language Model for English (1450-1950).” In Proceedings of the Workshop on Natural Language Processing for Digital Humanities, 23–36. NIT Silchar, India: NLP Association of India (NLPAI), 2021. https://aclanthology.org/2021.nlp4dh-1.4.

Manjavacas Arevalo, Enrique, and Lauren Fonteyn. “Non-Parametric Word Sense Disambiguation for Historical Languages.” In Proceedings of the 2nd International Workshop on Natural Language Processing for Digital Humanities, 123–34. Taipei, Taiwan: Association for Computational Linguistics, 2022. https://aclanthology.org/2022.nlp4dh-1.16.

Rombach, Robin, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. “High-Resolution Image Synthesis with Latent Diffusion Models.” arXiv, April 13, 2022. https://doi.org/10.48550/arXiv.2112.10752.

Rosin, Guy D., Ido Guy, and Kira Radinsky. “Time Masking for Temporal Language Models.” In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, 833–41. Virtual Event AZ USA: ACM, 2022. https://doi.org/10.1145/3488560.3498529.

Rosin, Guy D., and Kira Radinsky. “Temporal Attention for Language Models.” arXiv, May 3, 2022. http://arxiv.org/abs/2202.02093.

Xue, Linting, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. “ByT5: Towards a Token-Free Future with Pre-Trained Byte-to-Byte Models.” Transactions of the Association for Computational Linguistics 10 (March 25, 2022): 291–306. https://doi.org/10.1162/tacl_a_00461.

### 3.	What are (dis)advantages of pre-training vs. fine-tuning?

Learning in most of NLP systems consists of two levels of training. First a Language Model (LM) with millions of parameters is trained on a large unlabeled corpus. Then the representations that are trained in the pretrained model are used in supervised learning for a downstream task, with optional updates (fine-tuning) of the representations and network from the first stage [2]. This yields a question: Do we need to create a language model from scratch, or we can reuse an existing one and fine-tune it (or apply it) for our tasks? To address this question, we describe four different scenarios and analyze their costs and benefits based on the following aspects:

-	Generalizable – Performance on different downstream tasks
-	Required time and resources
-	Required amount of data

#### Scenario 1 - Training from scratch

This is a domain specific pretraining which leads to a performance gain in related downstream tasks [2]. However, we need a large amount of in-domain data as well as computational resources and time.

There are two types of BERT models adapted for a specific domain. In the first type a model is pre-trained with whole-word masking from scratch using domain-specific corpus. Second type of models are pre-train based on the original BERT model using the corpus of that specific domain [1]. Experiments in [1] shows the performance of the former type in most of downstream tasks was better than the latter.

#### Scenario 2 - Continue pretraining a pretrained model (continual learning)

Continual learning may refer to two concepts of domain-adaptive pretraining and task-adaptive pretraining.

Domain-adaptive pretraining (also discussed in scenario 1) means continue pretraining a model on a large corpus of unlabeled domain-specific text [2]. 

Task-adaptive pretraining refers to pre-training on the unlabeled training set for a given task [2]; Other studies e.g. [1] and [3] show its effectiveness. 

Similar to scenario 1, we need a large amount of in-domain data as well as computational resources and time while we gain better performance than the previous scenario.


#### Scenario 3 - Using a pretrained model and apply it to new data (zero-shot)

Some studies show that BERT for general domain might generalize poorly on a specific domain since every domain has its unique knowledge. BERT cannot gain such knowledge without pre- training on the data for specific domain [1]. It might be the case for other existing pre-trained models if their in-domain data differ from our downstream tasks.
In this scenario less time, data, and fewer resources are required. To gain acceptable performance, a suitable pretrained model in similar domain is required.

#### Scenario 4 - Fine-tuning a pretrained (historical or contemporary) model on downstream task
A pre-trained model can be fine- tuned for diverse downstream tasks via supervised training on labeled datasets. It means the parameters of the pre-trained model are updated through the supervised learning process. Costs and benefits of this scenario is similar to scenario 3, while it might gain slightly better performance.


#### Available pretrained models for historical data:
- English
    - MacBERTh

- Dutch
    - GysBERT (Manjavacas & Fonteyn, 2022)
    - RobBERT? (Delobelle et al., 2019) (2020: https://doi.org/10.18653/v1/2020.findings-emnlp.292)
    - Historical Dutch (https://huggingface.co/dbmdz/bert-base-historic-dutch-cased)
    - Bertje? (de Vries et al., 2019)
    - XLM-R (Facebook)


#### Notes:
- Training from scratch if we have enough data, time, and ressources
- If we don't have enough data, time and ressources -> fine-tuning
- Pretraining does not require *labeled* data but it might benefit from weak supervision (see Whisper in audio-to-text domain)
- Evidence from MacBerth and GysBert: Pretrained models perform better than finetuned models on historical data
- Fine-tune pretrained models for downstream tasks
- Contextual language models should account for spelling variations across historical periods
- Transfer distance between pretraining domain and downstream domain should be not too large
- Tip: use stable widely used model architectures (for example: Bert)
- Tip: when training a new model, aim for a release of the model on huggingface.co

#### Brainstorming ideas:

* general language model "knows" the world outside the research corpus
    * (when) is external world knowledge desirable?
* How to make in-domain pre-trained models useful for other tasks and researchers?
* How much computational power is necessary (per data, research question etc.) for pre-training vs. fine-tuning?

#### Reference
[1] https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04642-w
[2] https://aclanthology.org/2020.acl-main.740.pdf
[3] Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. In ACL.

### 4.	What are suitable ways of evaluating Dutch historical language models?

We estimate a language model on a text corpus covering several decades, and then use some of its parameters (word embeddings, ...) to answer the main research question: has the meaning of a word changed over time? We will assume that we can point the model to particular time periods, either because time is explicitly incorporated when estimating the model, or because the same model is estimated for different time periods. 

We propose an approach with three steps.
First, we evaluate the models's performance on unseen text, where we compare the model to existing state of the art models, and assess how it performs on common NLP tasks.
Second, we evaluate the model's performance on unseen semantic shifts---we want a model with both good recall (does it find shifts that actually occurred) and good precision (it does not excessively detect shifts that have not occurred).
Third, we use the model to find a semantic shift "in the wild". We suggest ways to increase the confidence that a conclusion from the estimated model is not an artefact of something else.


#### 4.1 Model goodness of fit 
To compare the estimated model to existing language models, the first option is to calculate the model perplexity: the probability distribution for a word in a particular sentence, conditional on the preceding words. 
The perplexity can then be compared to other models. A challenge is that model perplexities are only comparable for models with the same vocabulary, and are only applicable to models with normalized proability distributions ([Chen, Beeferman, Rosenfeld](https://www.cs.cmu.edu/~roni/papers/eval-metrics-bntuw-9802.pdf)).

The second option is to assess whether the model performs common NLP tasks equally well as other models. Example tasks include filling in the blanks, zero-shot-classification (for instance, classify the sentiment of a given sentence), sentence similarity, and finding the odd item in a list.

A challenge is that for both approaches, one can only compare models from the same language. 



#### 4.2 Detecting a semantic shift on "test" data

To assess whether a model reliably detects a semantic shift, we propose to use a test data set where the semantic shift is under the researcher's control. There are two different types of data available, and a range of tasks with which assess the model.

*Types of data*

The first data type is annotated data with a known semantic shift. Such data have been curated for English, German, Latin and Swedish ([Schlechtweg et al 2020](https://aclanthology.org/2020.semeval-1.1/)) and Russian ([RuShiftEval](https://github.com/akutuzov/rushifteval_public)).
If such data set does not exist for Dutch, it needs to be created.
The idea is then to query the model for specific words for which we know they changed their meaning.

An alternative to annotated data is to create synthetic data with a simulated semantic shift, and check whether the model detects the shift. The model is estimated on the synthetic data, similar to [(EMNLP 2019, section 5)](https://aclanthology.org/D19-1007/) on twitter data. 

The advantage of the first approach is that it is a "real-world" case; the advantage of the second is the option to create a large test data set.

*Tasks*

The simplest---and our preferred---task is to check, for words or sentences that changed their meaning, whether the model associates them with the correct time period. For instance, one can ask the model whether two sentences are from the same time period, or predict the time period a given sentence was written. 

There is a range of alternative tasks. A graphical approach projects the words' embeddings to two dimensions and plots them, similar to existing work by XXX.  <mark>**TODO: Carsten, what is the reference here?**</mark>

Other options are common NLP tasks, as the following example illustrates. 
Suppose word $x$ has changed meaning over time; in period 1, the synonyms are $\{y_1, y_2\}$, and in period 2 the synonyms are $\{z_1, z_2\}$. Then, a model trained on time period 1 should find an odd item in the set $\{z_1, z_2, x\}$ but not in the set $\{y_1, y_2, x\}$. A model trained on time period 2 should find an odd item in $\{y_1, y_2, x\}$ but not in $\{z_1, z_2, x\}$.
A last option is to use generative approaches: If there was a semantic shift for a given word, then the model should generate a different response for the same prompt in two different time periods. 


#### 4.3 Detecting a semantic shift "in the wild" 

Supposing a semantic shift has been found, we suggest ways to probe the robustness of this finding.

On one hand, the result should be consistent with other, simpler NLP approaches. For instance, the words that co-occur with the word "sustainability" should have changed over time. 
On the other hand, the shift could stem from other statistical confounding. For instance, it could reflect the frequency with which a token occurs. 

The first way to address such concerns is to account for time trends that affect the embedding vectors of all tokens in the same way. For instance, if the meaning of a word has truly changed, its embedding vector should have gotten closer to the embedding vector of the new synonym, and further away from the embedding vector of the old synonym. 

The second way is a placebo test: Shuffle the documents from the corpus randomly across time periods, and re-estimate the model and the semantic shift. If the semantic shift also shows up in these data, it is likely an artefact of something else, rather than from the change in meaning. In this case, it is perhaps possible to control for this change in embeddings in the model that uses the non-perturbed data set, but we have not explored this possibility in detail.


Lastly, we note that these steps are not isolated from each other. For instance, if a semantic shift is detected in the last step, then one can apply one of the NLP tasks from section 4.2 to validate the finding. 
Similarly, finding that the word "sustainability" has changed its meaning from A to B will come with some uncertainty, and perplexity can be used to quantify this uncertainty, since it recovers the probability distribution of the next word given the history. For instance, we can calculate the perplexities for the sentence "Sustainability is ..." for different time periods.




#### 4.5 Other ideas where it is unclear if useful
*Leaving them here for the record; feel free to delete/add above*
- Approaches that are used for training
    - Adversarial/competitive networks
        - one network does a task, the other says it is true or false 
    - Electra model: change random word in text to something else. let the model predict whether the word is fake or not
- Generative approaches 
    - train models for different time periods. ask them "what is sustainability?" then use another model to classify the paragraphs -- but how to train such a model? 
    - Paper on evaluating generative capacities of models on the basis of authorship attribution: [Assessing the Stylistic Properties of Neurally Generated Text in Authorship Attribution](https://aclanthology.org/W17-4914/)
- Use model to summarise text?
    - prompt-based approaches (Chat-GPT)? the problem is that these models use way more data than are available for historical research
- current research on explainability: give a task where items differ on very particular dimensions. the answer that the model gives can then be used to infer which dimension has an impact/is important in the model. this is more about what the model has learned/what impacts the model results. not sure how relevant it is here. 
- compare similarity scores using a test setup with paraphrases [(EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.569/)
    - ie formal and informal sentence that expresses the same content
- Interpretability metric: Lime
    - give example sentence to classify. the model iteratively changes some of the words, and sees which of the words change the prediction the most
    - requires specifically curated test set. can be used to detect lexical change, but not necessarily for models that were built for other tasks? 
    - this seems interesting, but I am not sure how it could be used to assess model performance. Would need to read up on it more.
- Focus not on the single word, but on more on the whole corpus
    - unclear what we would want to do specifically
- What is the model using to make the predictions?
    - this is related to the interpretability of the model, but I do not see where this fits into the above.
- a semantic could reflect changes in social attitudes. For instance, the language can reflect historical racism and its changes over time.
    - depends on the use: should the model reflect the historical circumstances? should the model be used for some tasks in society, and biases in the model could harm certain people.
    - similarity meanings / analogies may be able to detect this: ie, "muslim" is close to "terrorist"
    - **NOTE**: I think this is more about any interpretation of a semantic shift that one detects. Leaving it out for now.

### 5.	How can we best account for historicity (change over time) in these types of models?
*Moderator: Pim Huijnen*

The majority of large language models are trained on present-day data. This makes these models principally unfit, or at least problematic, for historical research. After all, any semantic information is necessarily always based on the current-day meaning of, and relation between words. Historical research departs from the exact opposite assumption that you may never just assume that words mean what you think they mean. Whether we are interested in 'democracy', 'health', 'energy', or 'honor': the meaning of (these) words is fundamentally subjected to their context. It is, therefore, essential that large language models in some manner account for the historical context in which words were used to make them fit for the study of history. The question, then, is how to do that. This chapter will discuss the most important considerations that implementing historicity into large language models in our view should take into account.

#### 5.1 What is change?
The first of this considerations is: what do we mean by change? Large language models are able to capture change on two levels: the conceptual or semantic and the linguistic level. Change on the linguistic level includes grammatical change (how a word is written) and syntactic change (how a word is used in a sentence, for example, as an adjective, noun or verb). We can define conceptual change in terms of the onomasiological and semasiological dimensions of concepts (Geeraerts 2010: 27), where the first describes the different representations or manifestations of a concept and the second its different meanings or uses. The two are interrelated, as the following example may clarify. The concept of 'propaganda' has long been a neutral term that was related to 'advertisement'. This changed during the Cold War, with the result that propaganda now has a clear political connotation, related to words like 'proclamation' or 'campaign' more than to 'publicity' or 'commercial'. These semantically related terms constitute the onomasiological dimension of the change in meaning of the term 'propaganda', its increasing political connotation its semasiological one. Research can both focus on onomasiological change, or how the manifestations of a particular concept change over time (like 'advertisement', of which 'propaganda' should - at least in English or Dutch - be an adequate manifestion before WWII, but not anymore), or on semasiological change, or the semantic internal change of a word. Here, Geeraerts distinguishes between the changes of the denotional, referential or connotational meaning of a word (Geeraerts 2010: 26ff).  

To be sure: there are more ways to study conceptual change. An important dimension is the *Pragmatik* that Koselleck distinguishes besides *Semantik*, *Syntax* and *Grammatik* (Koselleck 2006). This aspect of meaning and its change has clear parallels to Skinner's stress on intentionality and context (Skinner 1969). However, these dimensions are out of reach for large language models, because they require a hermeneutical accessapproach to the original texts that underly these models.

#### 5.2 How to incorporate temporal information?


options
* multiple models representing change over time
* build in historicity into the model
* use time-stamps for evaluation tasks outside of the model

How to incorporate temporal information?

    Adapt model architecture to temporal use (explicit)
    Add temporal context to text sequences (implicit)
    Use model and text without changes

##### 5.2.1 Multiple models or one?

Multiple models or one model?
* for multiple models (e.g. same hyperparameters different data, -> e.g. retraining, further pretraining, ...):
    * it works for word embeddings: Shico (2015)
    * distribution of the data over time is skewed, preventing us from seeing contextual information from periods in which data is sparse when using a single model
* against multiple models:
    * which token embeddings to use as seed terms?
    * how to decide periodization?
    * token embeddings already represent historical change

##### 5.2.2 Build in historicity into the model

    Adapt model architecture to temporal use (explicit)
    Add temporal context to text sequences (implicit)

#### 5.3 Evaluation

What is the specific task? It affects the setting, e.g. options, steps, ...
* ideology detection: detect specific sources (news media, political actors, etc.) from a number of different ones
* look at largest perplexity as a proxy for meaning change
* sentence comparison (for example, based on a seed list of interesting words) with time-stamped sentences to
    * find similarities between sentences with the same key word for different time periods (= no change)
    * trace the similarity of key words by studying their similarity for different time periods

Important: how to evaluate historical change?

Literature

Reinhart Koselleck, ‘Stichwort: Begriffsgeschichte’, in: idem., Begriffsgeschichten. Studien zur Semantik und Pragmatik der politischen und sozialen Sprache (Frankfurt a.M. 2006) 99-102.

Skinner

Geeraerts-