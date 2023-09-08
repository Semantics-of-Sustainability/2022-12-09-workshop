# How to create (historical) Dutch transformer models
## A working paper

### Authors

Sophie Arnoult  
&nbsp;&nbsp;*VU Amsterdam*  

Sara Budts  
&nbsp;&nbsp;*Antwerp University*  

Andreas van Cranenburgh  
&nbsp;&nbsp;*Groningen University*

Mirjam Cuper  
&nbsp;&nbsp;*National Library The Hague*

Ronald Dekker  
&nbsp;&nbsp;*KNAW Humanities Cluster*

Pieter Delobelle  
&nbsp;&nbsp;*KU Leuven*

Lauren Fonteyn  
&nbsp;&nbsp;*Leiden University*

Anastasia Giachanou  
&nbsp;&nbsp;*Utrecht University*

Julian Gonggrijp  
&nbsp;&nbsp;*Utrecht University*

Flavio Hafner  
&nbsp;&nbsp;*NL eScience Center Amsterdam*

Pim Huijnen  
&nbsp;&nbsp;*Utrecht University*

Ali Hürriyetoğlu  
&nbsp;&nbsp;*KNAW Humanities Cluster*

Marijn Koolen  
&nbsp;&nbsp;*KNAW Humanities Cluster*

Ken Krige  
&nbsp;&nbsp;*Utrecht University*

Malte Luken  
&nbsp;&nbsp;*NL eScience Center Amsterdam*

Enrique Manjavacas  
&nbsp;&nbsp;*Leiden University*

Dong Nguyen  
&nbsp;&nbsp;*Utrecht University*

Laura Ootes  
&nbsp;&nbsp;*NL eScience Center Amsterdam*

Luka van der Plas  
&nbsp;&nbsp;*Utrecht University*

Carsten Schnober  
&nbsp;&nbsp;*NL eScience Center Amsterdam*

Erik Tjong Kim Sang  
&nbsp;&nbsp;*NL eScience Center Amsterdam*

Stella Verkijk  
&nbsp;&nbsp;*VU Amsterdam*

Arjen Versloot  
&nbsp;&nbsp;*University of Amsterdam*

Leon van Wissen  
&nbsp;&nbsp;*University of Amsterdam*

Parisa Zahedi  
&nbsp;&nbsp;*Utrecht University*

---
  

## Introduction

The rise of transformer language models such as [BERT](https://arxiv.org/abs/1810.04805) has opened up possibilities to use contextualized word embeddings for downstream text processing tasks. This includes applications in humanities research. However, the methods to properly use these models in a humanities context -- and, particularly, for historical research -- are still very much under development. The aim of this working paper is to present guidelines for using transformer language models to study change over time. This paper is based on a [workshop](https://github.com/Semantics-of-Sustainability/2022-12-09-workshop) held at the NL eScience Center Amsterdam on 9 December 2022, which brought together experts in computational analysis of historical text.

---
  

## 1. Which corpora are suitable to base Dutch historical language models on?

There are various Dutch text sources available to train language models, including books, book reviews, news sources, Wikipedia, and Twitter. Yet, if a trained language model will be used to analyse historical texts or to answer questions related to language over different historial time frames, the training corpus should meet specific requirements. In this section, we discuss what should be taken into account when training historical Dutch language models. 


### 1.1. Corpus properties

A comment often made about transformer language models is that they are 'data-hungry', which means that the quality of a trained model is affected the size of the training corpus. However, when it comes to historical text, the number of available resources are not unlimited. Thus, researchers who wish to train a new model often wonder what the (minimum) required amount of training data is. The answer depends on architecture, task and context of the language model. 
<!--- What do you mean by 'the context of the language model'? --->
Furthermore, language models can either be 'pre-trained from scratch', e.g. when a model of a particular language (variety) has been pre-trained on linguistic material from the target language (variety), or an existing pre-trained language model can be fine-tuned or adapted to a new language (variety). Generally speaking, pre-training from scratch requires more input than adaptation. Additionally, when compiling a data set to pre-train a language model, one needs to consider how to balance different input corpora (e.g. in terms of genre, time period, etc.). 
Given that many historical languages could be considered 'low-resource' languages, for which it is not practically feasible to copile a large and balanced data set for pre-training, adaptation may be a solution. It has been argued that, for a fixed model capacity (model size, number of parameters), low-resource languages benefit from related high-resource languages. As such, language models trained on Present-day language varieties could serve as the basis for an adapted historical model. 
<!--- Terms, like 'training from scratch' should be introduced before they are mentioned. I've made some suggestions. --->
<!--- I've removed "However whereas adding more languages to training decreases performance after a point (Conneau et al. 2020, Li et al. (2020))." because I don't know how it fits into teh argument you are trying to make --->

To train a historical Dutch language model the corpus balance, bias, and representiveness of the training data have to be carefully considered. 
<!--- You say that the data has to be balanced, but try to briefly explain why --->
To create a model that is representative of historical Dutch in a broad sense, it is important that the training data is diverse in terms of domain, genre, topic, authors, style, and time period. 
<!--- How is domain different from genre? --->
To make an informed decision, any information that can be inferred from the data is relevant, and ideally the metadata should contain:
<!--- Are these guidelines for training data? As in, are you telling people to ideally only include data for which this information is available? Or are you telling people who compile corpora what sort of meta-data they should include so that there data can be used for pre-training? Perhaps you should be explicit about this? --->
- Time period
- Language domain / genre / style (news paper, parlementairy etc.)
- Size (number of tokens, sentences, files, and data size)
- Source
- Document type
- Date issued
- Quality of OCR/HTR
- URL / DOI
- License

Available training data often consists of data sets with substantial differences in data quality.
<!--- Could you give an example? --->
How to deal with these quality differences depends on the model task. It might make sense to train models with multiple sizes of lower quality data, but it is not clear whether this is an acceptable solution. On the one hand, a model trained on lower quality data may produce output that is of insufficient quality. On the other hand, lower quality training data may also be beneficial, as the target data is often equally messy.

Even when there is sufficient data available, the accompanying licence needs to be carefully evaluated. Some available corpora have licenses that give the user a lot of freedom. Other data is allowed to be accessed, but not redistributed. Corpora may also contain both data with and without copyright restrictions.



### 1.2. Available corpora

Listed in the table below is a non-exhaustive number of available corpora (in alphabetical order) that can be considered for creating historical Dutch transformer models.

<!--- SoNaR is not a historical corpus. I also wouldn't consider Europarl historical, as the earliest texts only go back to 1996. Please define 'historical'. Do you really mean 'historical', or just 'diachronic'? --->

| Title | License | Period | Genre | Size | Reference |  Comments|
|-|-|-|-|-|-|-|
|[Amsterdam City Archives](https://transkribus.eu/r/amsterdam-city-archives)| <b>CC0 1.0 (?)</b> <!-- not sure -->| 1578-1811 | Notarial deeds, administrative | | | Contains both HTR and Ground Truth documents|
|[DBNL](https://www.dbnl.org)|[Permission needed](https://www.kb.nl/onderzoeken-vinden/datasets/dbnl-dataset)| <b>c.a. 1200 <!-- 1550 --> -present (?) </b>| plays, poetry, novels, letters ||
|[Delpher newspapers](https://www.delpher.nl/over-delpher/delpher-open-krantenarchief/wat-zit-er-in-het-delpher-open-krantenarchief)| [Permission needed](https://www.delpher.nl/over-delpher/data-in-delpher) | 1618-1879 | newspapers, books, periodicals| ca. 111-126 GB| |
| [EuroParl](https://www.statmt.org/europarl/) | [Permission needed](https://www.statmt.org/europarl/) | 1996-2011 | Political discussions, transcripts, Administrative | parallel corpus Dutch-English: 190 MB | [Philipp Koehn, MT Summit 2005](http://www.iccs.inf.ed.ac.uk/~pkoehn/publications/europarl-mtsummit05.pdf) | Data is same as Staten Generaal Digitaal |
|[Huygens Resources](https://resources.huygens.knaw.nl)| |   700-present | correspondence, administrative, and more || |Collection of various recourses |
| [Corpus Gysseling](http://hdl.handle.net/10032/tm-a2-j4) |[Non-commercial user agreement](https://taalmaterialen.ivdnt.org/wp-content/uploads/voorwaarden/voorwaarden_corpus-gysseling.pdf)|1200-1300 |Corpus Middel Nederlands| 42 MB| Corpus Gysseling (Version 1.0) (1990) [Data set]. [Available at the Dutch Language Institute](http://hdl.handle.net/10032/tm-a2-j4)| Collection of 13th century texts that served as resources for the Vroegmiddelnederlands dictionary|
| [NederLab](https://www.nederlab.nl/onderzoeksportaal/) |||| | | Web environment for researchers containing historical recourses for Dutch language, literature, and culture. |
| [NIBG Radio and television](https://www.beeldengeluid.nl/kennis/kennisbronnen) | [ Terms of use ](https://files.beeldengeluid.nl/pdf/Gebruikersvoorwaarden_DAAN-BeeldenGeluid.pdf)| 1870-present | Radio transcripts, subtitles ||
| [Resolutions States General](https://republic.huygens.knaw.nl) |[Permission needed](https://republic.huygens.knaw.nl/index.php/republic-huygens-ing/werkwijze/)| 1576-1796 | Government decisions, Administrative | || Currently in progess of development |
| [OSCAR](https://oscar-project.org/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Data avilable [upon request](https://oscar-project.github.io/documentation/accessing/) and via [Huggingface](https://huggingface.co/oscar-corpus)|||180GB (Dutch)| [See publications](https://oscar-project.org/#featured) |Based on common crawl|
| [SONAR](https://taalmaterialen.ivdnt.org/download/tstc-sonar-corpus/) |[Non-commerical user agreement](https://taalmaterialen.ivdnt.org/wp-content/uploads/voorwaarden/voorwaarden_sonar-corpus.pdf)| present-day | various |SoNaR-500: 500 million words, SoNaR-1: 1 million words|SoNaR-corpus (Version 1.2.1) (2015) [Data set]. [Available at the Dutch Language Institute](http://hdl.handle.net/10032/tm-a2-h5)Two data sets are available: SoNaR-500 and SoNaR-1|
| [Staten Generaal Digitaal](https://data.overheid.nl/dataset/staten-generaal-digitaal---koninklijke-bibliotheek) | [CC-0-1.0](https://creativecommons.org/publicdomain/zero/1.0/deed.nl) | 1814-1995 | Reports of meetings in Dutch parliament ||
| [Taalmaterialen IvdNT](https://taalmaterialen.ivdnt.org/document-tag/corpus/) | Check individual data sets | Large range of time spans | Various | | | IvdNT provides a large amount of Dutch historical data sets, data details can be found per data set |
| [Twente Nieuws Corpus (TwNC)](https://research.utwente.nl/en/datasets/twente-nieuws-corpus-twnc) |[Check documentation](https://research.utwente.nl/files/6545509/TwNC-ELRA-final.pdf)|Different per source|Newspaper articles, subtitles and news broadcasts|530M words|[Ordelman, R. J. F., de Jong, F. M. G., van Hessen, A. J., & Hondorp, G. H. W. (2007). TwNC: a Multifaceted Dutch News Corpus. ELRA Newsletter, 12(3-4).](https://research.utwente.nl/en/publications/twnc-a-multifaceted-dutch-news-corpus)|
| [VOC, WIC, and Notarial deeds HTR](https://doi.org/10.5281/zenodo.6414086) |[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode)| 1637-1792 | reports, correspondence | 100.4 GB |DOI: 10.5281/zenodo.6414086|
| [Woordenboek Nederlandse Taal](https://ivdnt.org/woordenboeken/woordenboek-der-nederlandsche-taal/) | [Contact](https://ivdnt.org/profile/roland-de-bonth)  |1500 - 1976|Dutch historical dictionary|95.000 main keywords| | ||

---
  

## 2. Which model architectures and sizes are suitable?

Since the introduction of Transformer-based language models like BERT (Devlin et al., 2019), numerous model variations based on similar architectures have been developed.


Despite the various differences between BERT, RoBERTa (Liu et al., 2019) and their many other derivations (e.g. He et al., 2021), practitioners oftentimes do not see noticable differences originating from choosing one of those flavours over the other.

The performance of a language model depends on many aspects that are not related to its architecture, most importantly the input data and the specifics of a downstream task (Manjavacas Arevalo & Fonteyn, 2021).
Training large language models is expensive, so the potential benefits of finding the optimal architecture as well as hyper-parameter optimization are generally not expected to be worth the additional costs.
Consequently, for instance the developers of GysBERT (Manjavacas Arevalo & Fonteyn, 2022), have chosen to "closely follow the BERT-base uncased" architecture.

For a few reasons, it may be worth simply sticking to established methods and architectures rather than the latest state-of-the-art. First, the performance of a language model depends on many aspects that are not related to its architecture, such as the input data and the specifics of a downstream task (Manjavacas Arevalo & Fonteyn, 2021). Furthermore, because training large language models is expensive, the potential benefits of finding the optimal architecture as well as hyper-parameter optimization are generally not expected to be worth the additional costs. As such, existing historical language models such as Dutch GysBERT (Manjavacas Arevalo & Fonteyn, 2022) and English MacBERTh (Manjavacas Arevalo & Fonteyn, 2021) have been created by "closely follow the BERT-base uncased" architecture. However, there are plans to create at least a cased version of these models, which is expected to be beneficial for tasks such as Named Entity Recognition (NER). We will briefly discuss what sort of pre-processing could help optimize the quality of historical (Dutch) language models in Section 2.1.
Finally, it is also relevant to take development time into account: the entire process from data collection via implementation and training can take several years. During that time new best-performing architectures may have been developed, while flaws in currently popular architectures might be uncovered.
<!--- I don't really understand what point you're trying to make with this last argument. So because things develop really fast it's better to work with something that isn't sota? Can you develop that argument more? And if this is what is addressed in subsection 2.2, this should be made explicit.--->

Finding a suitable architecture can be approached more pragmatically by looking for previous approaches with similar requirements.
It is likely that models have been trained for similar use cases and/or on similar data in terms of domain, size and historical period.
If they have had good results, it may be worth considering a similar architecture, or even re-using the model.
<!--- Why a similar archiecture? Why not just reuse the model? --->

<!--- Here, your version of this draft says "In the context of historic language models, however, another important question is: should the temporal aspect be encoded into the model explicitly? Part 5 below elaborates on this point.". I don't understand why it says 'however', because there is no contrast with the previous statement, and the comment about the temporal dimension also comes out of the blue. --->


Another direction relevant in future research points towards models developed for applications outside of NLP.
Stable Diffusion (Rombach et al., 2022) is a generative text-to-image model; techniques developed there could be adapted for (application-specific) language modelling and/or multimodal models.
However, the specific method of adding noise as implemented in Stable Diffusion is not directly transferable to text data.
<!--- This idea also has to be developed further, or it shouldn't be mentioned at all. --->


### 2.1. Pre-processing and Tokenization

The performance of a language model depends, for a large part, on how its training data is prepared.
For one, the choice of the tokenization in terms of both design and size seems to be important, whereas there has been little specific research on the impact of different tokenizers on downstream tasks.
<!--- Seems to be important? Says who? Can you cite some work here that shows this? --->
Intuitively, it seems clear though that a tokenizer introduces a specific kind of bias by producing a specific set of tokens on which a model is trained.
<!--- Please explain this properly with an example --->

Xue et al. (2022) demonstrate that token-free, "byte-level models are competitive with their token-level counterparts", and "characterize the trade-offs in terms of parameter count, training FLOPs, and inference speed."
Clark et al. (2022) tackle the "linguistic pitfalls of tokenization" with a character-based, token-free tokenization approach.
<!--- This needs to be developed more --->

#### 2.1.1 Normalizing spelling
Normalization in the pre-processing pipeline is a related aspect.
Especially historic texts, oftentimes digitized automatically through optical character recognition (OCR) or handwritten text recognition (HTR), typically contain numerous errors.
Apart from OCR and HTR errors, spelling variation is very common in historic texts. For Dutch, the basis for the official spelling rules was designed by Mathijs Siegenbeek as late as 1804 (Van der Wal & van Bree, 2008), and after that the Dutch spelling system has also been revised on multiple occassions . Before that, capitalization was not consistent and different regions and time periods used to spell according to their own standards. This means that, especially in texts pre-dating the 19th century, even common words (e.g. _onnozel_) could be spelled in multiple ways:


zynen zoon, zoo oopentlyk _onnoozel_ ('his son, so openly innocent'; P.C. Hooft, 1642)


als Godts _onnosel_ Lam ('like God's innocent lamb'; J. Cats, 1657)

However, practitioners advice against normalization of input text before training a language model, because that also removes the model's robustness against variations that occur in unseen texts.
At the same time, (semi-)automatically removing garbage, as opposed to variation, does have a positive impact on the model.



Given such variation, the question arises whether the spelling in the input training data should be normalized. Practitioners advice against this for several reasons. First, a practical consideration is that reducing spelling variation through normalization negatively affects the model's robustness against the orthographic variation that will likely be present in historical texts that were not part of the training corpus. Second, genuine spelling variation (i.e. variant spellings of words that are not introduced because of OCR or HTR errors) is in fact socially meaningful (Nguyen & Grieve 2020) and may, in the case of historical text, encode sociolinguistically relevant information about the author (e.g. regional background, social class, etc.). If it is possible to (semi-)automatically remove orthographic noise (e.g. by refraining from using notoriously 'dirty' OCR text in training) while maintaining any genuine variation, the quality of the model will be positively affected.

Because almost all larger datasets for Dutch have already been digitized, new data is unlikely to appear in the foreseeable future. It is possible, however, that the quality of large databases (e.g. Delpher) improves in the future through improved OCR and HTR technology.
<!--- I have no idea what "Dealing with historic language data includes a specific aspect" means, so I removed it. I also wonder what this comment on the exhaustiveness of digitization is doing here. Is this the most optimal place to put it, in terms of text structure? --->

#### 2.1.1 Representativeness of training data
In order to get a sense for the data requirements for specific tasks or domains, it is important to understand of the specifics of the data to be used:

- How homogenous is the data?
- Which domain(s) does it cover?
- What task(s) should the model solve?

Quality measures such as perplexity (to detect text with problematic OCR/HTR), lexical variation, and stylistic variation can help with understanding the distribution and homogenity of the data.

<!--- As with the extensive discussion on spelling variation, please explain _why_ lexical and stylistic variation are important. Also note that two things are discussed here: (1) how messy the text is in terms of orthography and (2) whether balance should be created in a training corpus. Please indicate more clearly in the text that these are two different considerations. I added subheadings to make this clearer. The link between lexical diversity and tokenization should also be explained better. --->

Conneau et al. (2020) investigate the impact of data distribution, particularly considering low-resource languages in the context of multi-lingual models, showing ways to train performant models effectively even for languages and/or domains for which little data is available.

### 2.2. Computation

Computational costs for training a model fluctuate heavily, again depending on specifics of model architecture and input data size.
The latter is especially relevant, as pointed out by Kaplan et al. (2020).
They also analyse other aspects and inter-dependent factors regarding their respective impact on model training times and costs.

Smaller input data sets can likely be handled by, for instance, university-owned GPU machines and smaller-scale clusters.
However, they often are too slow (and overloaded) to train large-scale models such as GysBERT (Manjavacas Arevalo & Fonteyn, 2022).

Commercial services can handle those large workloads, but are expensive.
The costs of using a TPU machine as provided by the Google Cloud Engine can sum up to thousand of Euros during the course the approximate duration of training a large-scale language model, which can take several weeks.
These costs might decrease in the future, while computation speed might increase.
Anyway, training a large-scale language model using commercial offerings is too expensive for the budget of many research projects.

In the context of this project (Semantics of Sustainability), the NL eScience Center currently investigates ways of applying the Dutch National Supercomputer Snellius effectively and efficiently for scientific purposes.

---
   

## 3.	What are (dis)advantages of pre-training vs. fine-tuning?


Learning in most of NLP systems consists of two levels of training. First a Language Model (LM) with millions of parameters is trained on a large unlabeled corpus. Then the representations that are trained in the pretrained model are used in supervised learning for a downstream task, with optional updates (fine-tuning) of the representations and network from the first stage [2]. This raises the question whether a language model should be trained from scratch, or whether an existing model can be fine-tuned (or applied) to reach our goals. 
<!--- Pre-training from scratch relevant in an earlier section, but it is not explained until here. The order of the various sections may have to be rethought. --->
To address this question, we describe four different scenarios and analyze their costs and benefits based on the following aspects:


-	Generalizable – Performance on different downstream tasks
-	Required time and resources
-	Required amount of data

### 3.1. Scenarios

#### 3.1.1. Scenario 1 - Training from scratch

This is a domain specific pretraining which leads to a performance gain in related downstream tasks [2]. 
<!--- Try to explain what that means more by giving examples. --->
However, it requires a large amount of in-domain data as well as computational resources and time.


There are two types of BERT models adapted for a specific domain. In the first type a model is pre-trained with whole-word masking from scratch using domain-specific corpus. Second type of models are pre-train based on the original BERT model using the corpus of that specific domain [1]. Experiments in [1] shows the performance of the former type in most of downstream tasks was better than the latter.
<!--- Cite sources to back up the latter point. WHich experiments? By whom? --->

#### 3.1.2. Scenario 2 - Continue pretraining a pretrained model (continual learning)

Continual learning may refer to two concepts of domain-adaptive pretraining and task-adaptive pretraining.

Domain-adaptive pretraining (also discussed in scenario 1) means continue pretraining a model on a large corpus of unlabeled domain-specific text [2].
<!--- Again, this needs a more concrete example. Its also a bit odd to define something on second mention... --->

Task-adaptive pretraining refers to pre-training on the unlabeled training set for a given task [2]; Other studies e.g. [1] and [3] show its effectiveness. 
<!--- Needs examples. The studies you cite, what tasks do they perform? --->

Similar to scenario 1, a large amount of in-domain data is needed, as well as computational resources and time while we gain better performance than the previous scenario.


#### 3.1.3. Scenario 3 - Using a pretrained model and apply it to new data (zero-shot)

Some studies show that BERT for general domain might generalize poorly on a specific domain since every domain has its unique knowledge. 
<!--- Cite those studies --->
BERT cannot gain such knowledge without pre- training on the data for specific domain [1]. It might be the case for other existing pre-trained models if their in-domain data differ from our downstream tasks.
In this scenario less time, data, and fewer resources are required. To gain acceptable performance, a suitable pretrained model in similar domain is required.

#### 3.1.4. Scenario 4 - Fine-tuning a pretrained (historical or contemporary) model on downstream task
A pre-trained model can be fine- tuned for diverse downstream tasks via supervised training on labeled datasets. It means the parameters of the pre-trained model are updated through the supervised learning process. Costs and benefits of this scenario is similar to scenario 3, while it might gain slightly better performance.
<!--- Has this been tested by anyone? Please cite. Otherwise, explicitly state that the authors are not aware of anyone testing this. That begs the question why we think it might yield performance improvements, so this should be explained too. --->



### 3.2. Available pretrained models
- Historical English
    - MacBERTh (Manjavacas & Fonteyn, 2021)

- Historical Dutch
    - GysBERT (Manjavacas & Fonteyn, 2022)
    - Historical Dutch (https://huggingface.co/dbmdz/bert-base-historic-dutch-cased) <!--- This model uses Delpher for training without filtering out the illegible texts with poor OCR quality, if I'm not mistaken. --->

- Present-day Dutch
    -  RobBERT? (Delobelle et al., 2019) (2020: https://doi.org/10.18653/v1/2020.findings-emnlp.292) <!--- Not historical --->
    -  Bertje? (de Vries et al., 2019) <!--- Not historical --->
    -  BERT-NL (Brandsen et al., 2019) https://repository.han.nl/han/handle/20.500.12470/1092
    -  XLM-R (Facebook) <!--- Not historical, I'm assuming --->

<!--- There are other historical BERT models for Latin, Italian, French, etc. Should these be listed? Also, there are other pre-trained models that aren't BERT/Transformer-based. I think there's an ELMo model for English used to automatically parse texts (with the PENN parsed data as input), though I'm not sure whether it has been released. https://par.nsf.gov/servlets/purl/10340101 Should these be mentioned? --->



### 3.3. Miscellaneous notes and brainstorming ideas:
- Training from scratch if we have enough data, time, and resources
- If we don't have enough data, time and resources -> fine-tuning
- Pretraining does not require *labeled* data but it might benefit from weak supervision (see Whisper in audio-to-text domain)
- Evidence from MacBERTh and GysBERT: Pretrained models perform better than finetuned models on historical data
- Fine-tune pretrained models for downstream tasks
- Contextual language models should account for spelling variations across historical periods
- Transfer distance between pretraining domain and downstream domain should be not too large
- Tip: use stable widely used model architectures (for example: BERT)
- Tip: when training a new model, aim for a release of the model on huggingface.com
- general language model "knows" the world outside the research corpus
    - (when) is external world knowledge desirable?
- How to make in-domain pre-trained models useful for other tasks and researchers?
- How much computational power is necessary (per data, research question etc.) for pre-training vs. fine-tuning?

---

## 4.	What are suitable ways of evaluating Dutch historical language models?

We estimate a language model on a text corpus covering several decades, and then use some of its parameters (word embeddings, ...) to answer the main research question: has the meaning of a word changed over time? We will assume that we can point the model to particular time periods, either because time is explicitly incorporated when estimating the model, or because the same model is estimated for different time periods.

<!--- Who's research question is this? This is just one type of research question that people could address with models like this, so it's quite odd that this is 'the' research question. Also, evaluating Dutch Historical Language models can be done in many different ways. Shouldn't this section reflect more on what gold standards there are for historical Dutch? For GysBERT, we used WNT data and manually annotated sets. --->

We propose an approach with three steps.
<!--- I'm not sure if what follows can be considered three steps rather than three ways of evaluating a model that could be done in sequence but they could also be done seperately. --->
First, we evaluate the models's performance on unseen text, where we compare the model to existing state of the art models, and assess how it performs on common NLP tasks (Section 4.1).
Second, we evaluate the model's performance on unseen semantic shifts---we want a model with both good recall (does it find shifts that actually occurred) and good precision (it does not excessively detect shifts that have not occurred) (Section 4.2).
Third, we use the model to find a semantic shift "in the wild". We suggest ways to increase the confidence that a conclusion from the estimated model is not an artefact of something else (Section 4.3).
<!--- Have you actually done this? Because the texts presents it as if you've done these evaluations, but then the description in the sections rather suggests that this a 'best practice' reflection? In the latter case, add modal verbs ("We can evaluate" etc.) --->

### 4.1. Model goodness of fit 
To compare the estimated model to existing language models, the first option is to calculate the model perplexity: the probability distribution for a word in a particular sentence, conditional on the preceding words. 
The perplexity can then be compared to other models. A challenge is that model perplexities are only comparable for models with the same vocabulary, and are only applicable to models with normalized probability distributions ([Chen, Beeferman, Rosenfeld](https://www.cs.cmu.edu/~roni/papers/eval-metrics-bntuw-9802.pdf)).

The second option is to assess whether the model performs common NLP tasks equally well as other models. Example tasks include filling in the blanks, zero-shot-classification (for instance, classify the sentiment of a given sentence), sentence similarity, and finding the odd item in a list.

A challenge is that for both approaches, one can only compare models from the same language. 


### 4.2. Detecting a semantic shift on "test" data

To assess whether a model reliably detects a semantic shift, we propose to use a test data set where the semantic shift is under the researcher's control. There are two different types of data available, and a range of tasks with which assess the model.

*Types of data*

The first data type is annotated data with a known semantic shift. Such data have been curated for English, German, Latin and Swedish ([Schlechtweg et al 2020](https://aclanthology.org/2020.semeval-1.1/)) and Russian ([RuShiftEval](https://github.com/akutuzov/rushifteval_public)).
If such data set does not exist for Dutch, it should be created.
The idea is then to query the model for specific words for which we know they changed their meaning.

An alternative to annotated data is to create synthetic data with a simulated semantic shift, and check whether the model detects the shift. The model is estimated on the synthetic data, similar to [(EMNLP 2019, section 5)](https://aclanthology.org/D19-1007/) on twitter data. 

The advantage of the first approach is that it is a "real-world" case; the advantage of the second is the option to create a large test data set.

*Tasks*

The simplest---and our preferred---task is to check, for words or sentences that changed their meaning, whether the model associates them with the correct time period. For instance, one can ask the model whether two sentences are from the same time period, or predict the time period a given sentence was written.

There is a range of alternative tasks. A graphical approach projects the words' embeddings to two dimensions and plots them, facilitating the visualization of clusters that correspond to different meaning to a word.
Kutuzov et al. (2022) provide an overview of current methods for semantic shift detection using contextual language models, illustrated with said visualizations using PCA.

Other options are common NLP tasks, as the following example illustrates. 
Suppose word $x$ has changed meaning over time; in period 1, the synonyms are $\{y_1, y_2\}$, and in period 2 the synonyms are $\{z_1, z_2\}$. Then, a model trained on time period 1 should find an odd item in the set $\{z_1, z_2, x\}$ but not in the set $\{y_1, y_2, x\}$. A model trained on time period 2 should find an odd item in $\{y_1, y_2, x\}$ but not in $\{z_1, z_2, x\}$.
A last option is to use generative approaches: If there was a semantic shift for a given word, then the model should generate a different response for the same prompt in two different time periods. 


### 4.3. Detecting a semantic shift "in the wild" 

Supposing a semantic shift has been found, we suggest ways to probe the robustness of this finding.

On one hand, the result should be consistent with other, simpler NLP approaches. For instance, the words that co-occur with the word "sustainability" should have changed over time. 
On the other hand, the shift could stem from other statistical confounding. For instance, it could reflect the frequency with which a token occurs. 

The first way to address such concerns is to account for time trends that affect the embedding vectors of all tokens in the same way. For instance, if the meaning of a word has truly changed, its embedding vector should have gotten closer to the embedding vector of the new synonym, and further away from the embedding vector of the old synonym. 

The second way is a placebo test: Shuffle the documents from the corpus randomly across time periods, and re-estimate the model and the semantic shift. If the semantic shift also shows up in these data, it is likely an artefact of something else, rather than from the change in meaning. In this case, it is perhaps possible to control for this change in embeddings in the model that uses the non-perturbed data set, but we have not explored this possibility in detail.


Lastly, we note that these steps are not isolated from each other. For instance, if a semantic shift is detected in the last step, then one can apply one of the NLP tasks from section 4.2 to validate the finding. 
Similarly, finding that the word "sustainability" has changed its meaning from A to B will come with some uncertainty, and perplexity can be used to quantify this uncertainty, since it recovers the probability distribution of the next word given the history. For instance, we can calculate the perplexities for the sentence "Sustainability is ..." for different time periods.


### 4.5. Miscellaneous notes and brainstorming ideas
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

---
  

## 5.	How can we best account for historicity (change over time) in these types of models?

The majority of large language models are trained on present-day data. This makes these models principally unfit, or at least problematic, for historical research. After all, any semantic information is necessarily always based on the current-day meaning of, and relation between words. Historical research departs from the exact opposite assumption that you may never just assume that words mean what you think they mean. Whether we are interested in 'democracy', 'health', 'energy', or 'honor': the meaning of (these) words is fundamentally subjected to their context. It is, therefore, essential that large language models in some manner account for the historical context in which words were used to make them fit for the study of history. The question, then, is how to do that. This chapter will discuss the most important considerations that implementing historicity into large language models in our view should take into account.

<!--- I agree with this intro, but what is lacking is evidence that Present-day models fail to work well with historical data -- for instance because they introduce present-day bias / anachronistic readings onto texts. I deal with this in one paper: https://ceur-ws.org/Vol-2723/short15.pdf -- but there should be others that look at lexical phenomena. --->

### 5.1. What is change?


The first of this considerations is: what do we mean by change? Large language models are able to capture change on two levels: the conceptual or semantic and the linguistic level. 
<!--- On the distinction between these two levels, also see https://ceur-ws.org/Vol-2989/long_paper26.pdf and https://arxiv.org/pdf/1606.02821.pdf --->
Change on the linguistic level includes orthographic change (how a word is spelled) and grammatical change (how a word is used in a sentence, for example, as an adjective, noun or verb). 
<!--- A distinction was made in the original text between grammatical change, defined as how a word is written, and syntactic change. This is terminologically quite awkward. First of all, syntax is an aspect of grammar. Second, grammar is about much more than how a word is written. In fact, how a word is written is one of the few thing we wouldn't consider grammar. --->
We can define conceptual change in terms of the onomasiological and semasiological dimensions of concepts (Geeraerts 2010: 27), where the first describes the different representations or manifestations of a concept and the second its different meanings or uses. The two are interrelated, as the following example may clarify. The concept of 'propaganda' has long been a neutral term that was related to 'advertisement'. This changed during the Cold War, with the result that propaganda now has a clear political connotation, related to words like 'proclamation' or 'campaign' more than to 'publicity' or 'commercial'. These semantically related terms constitute the onomasiological dimension of the change in meaning of the term 'propaganda', its increasing political connotation its semasiological one. Research can both focus on onomasiological change, or how the manifestations of a particular concept change over time (like 'advertisement', of which 'propaganda' should - at least in English or Dutch - be an adequate manifestion before WWII, but not anymore), or on semasiological change, or the semantic internal change of a word. Here, Geeraerts distinguishes between the changes of the denotional, referential or connotational meaning of a word (Geeraerts 2010: 26ff).  


To be sure: there are more ways to study conceptual change. An important dimension is the *Pragmatik* that Koselleck distinguishes besides *Semantik*, *Syntax* and *Grammatik* (Koselleck 2006). This aspect of meaning and its change has clear parallels to Skinner's stress on intentionality and context (Skinner 1969). However, these dimensions are out of reach for large language models, because they require a hermeneutical access to the original texts that underly these models.

<!--- The level of theoretical detail here is in stark contrast with other parts of the paper. --->

### 5.2. How to incorporate temporal information?

Two basic strategies exist to use large language models to study change over time:

* create multiple models with data from subsequent time periods to represent change over time
* build in temporal information into one model

These two strategies will be elaborated below. However, it is important to note that there is a third option to use language models for historical research: not to take temporal information into account in the architecture of the model at all. Once the reliability of the model is established, specific tasks can be used to study semantic shifts based on domain knowledge - when the nearest neighbours of 'cell' in an English language model based on historical and current data contain both references to biology and to mobile phones, researchers can infer a semasiological change or broadening from that information alone.

#### 5.2.1. Multiple models or one?

The creation of multiple models with data from subsequent time periods used to be the standard to use word embedding algorithms to study historical change. Hamilton, Leskovec, and Jurafsky (2016) were the first to demonstrate this principle by creating two models, align these using orthogonal Procrustes and calculate the cosine distances of words in both models. The alignment of the models is a crucial factor here, because the embedding spaces of two independently trained models are not automatically comparable: although words might be used similarly in both datasets, their vector representations do not necessarily have to be similar (Wevers and Koolen 2020). Orthogonal Procrustes aligns subsequent models to the first to ensure comparability. The downside of this approach is that it only works if the two (or more) models share the exact same vocabulary. Words that are present in not all models first have to be pruned, which seems a large sacrifice for scholars interested in semantic change (Wevers and Koolen 2020).

Different approaches to use multiple models have been introduced since (Wevers and Koolen 2020). One of these is using models with partly overlapping data, so as to force the algorithm to build upon earlier models, but also to represent both change *and* continuity in the models (Kenter et al. 2015). This method enables scholars to not just compare to distinct periods in time, but to follow shifts in meaning from one period to another.

Some considerations for the use of multiple models to study semantic change:

* pro multiple models:
    * it works (for word embeddings)
    * the distribution of data over time is often skewed. This prevents us from seeing sufficient contextual information from periods in which data is sparse when using a single model

* against multiple models:
    * periodization (slicing a dataset into multiple, diachronic subsets) is necessarily arbitrary 
    * token embeddings already represent historical change, which takes away the necessity to create multiple models
    * it will be difficult to establish which of the token embeddings should be used as seed terms

#### 5.2.2. Build in historicity into the model

Multiple approaches have demonstrated that building in historicity directly into the language model is possible. Rosin et al. (2022) add an additional attention matrix to encode the chronological information. TempoBERT (Rosin & Radinsky, 2022), on the other hand, applies a standard BERT architecture -- after manipulating the text in the training data so that it contains temporal information. They show that downstream tasks such as semantic shift detection and sentence time prediction benefit significantly from this method.

<!--- This tempoBERT should be mentioned with the available pre-trained models, I suppose. --->

### 5.3. Evaluation

The decision for representing historicity via multiple models or within one large model, in the end, heavily depends on the specific tasks the model has to do. Some suggestions for tasks:

* ideology detection: detect specific sources (news media, political actors, etc.) from a number of different ones
* look at largest perplexity as a proxy for meaning change
* sentence comparison (for example, based on a seed list of interesting words) with time-stamped sentences to
    * find similarities between sentences with the same key word for different time periods (= no change)
    * trace the similarity of key words by studying their similarity for different time periods

For all these tasks it is crucial that the model 'works'. This means that the outcomes of these tasks are reliable in the sense that they represent history more than biases in the data, artefacts of the model's architecture, etc. A number of examples in the form of tokens or sentences, of which the historical trajectories are known, can help evaluating the model. 

---
  

## 6. Bibliography

Abadji, Julien, Pedro Ortiz Suarez, Laurent Romary, Benoît Sagot. “Towards a cleaner document-oriented multilingual crawled corpus.” arXiv, January 17, 2022. https://doi.org/10.48550/arXiv.2201.06642 

Clark, Jonathan H., Dan Garrette, Iulia Turc, and John Wieting. “Canine: Pre-Training an Efficient Tokenization-Free Encoder for Language Representation.” Transactions of the Association for Computational Linguistics 10 (January 31, 2022): 73–91. https://doi.org/10.1162/tacl_a_00448.

Conneau, Alexis, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. “Unsupervised Cross-Lingual Representation Learning at Scale.” arXiv, April 7, 2020. https://doi.org/10.48550/arXiv.1911.02116.

Conneau, Alexis, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. “Unsupervised Cross-Lingual Representation Learning at Scale.” arXiv, April 7, 2020. https://doi.org/10.48550/arXiv.1911.02116.

Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. “BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding.” ArXiv:1810.04805 [Cs], May 24, 2019. http://arxiv.org/abs/1810.04805.

Geeraerts, Dirk. Theories of Lexical Semantics. Oxford and New York, 2010.

Gururangan, Suchin, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. “Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks.” In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 8342–8360, Online. Association for Computational Linguistics, 2020. https://doi.org/10.18653/v1/2020.acl-main.740 

Hamilton, Wiliam L., Jure Leskovec, and Dan Jurafsky. “Cultural shift or linguistic drift? comparing two computational measures of semantic change.” In Proceedings of the Conference on Empirical Methods in Natural Language Processing. Conference on Empirical Methods in Natural Language Processing, 2116-2121. NIH Public Access, 2016. https://doi.org/10.18653/v1/d16-1229 

He, Pengcheng, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. “DeBERTa: Decoding-Enhanced BERT with Disentangled Attention.” arXiv, October 6, 2021. https://doi.org/10.48550/arXiv.2006.03654.

Howard, Jeremy and Sebastian Ruder. “Universal Language Model Fine-tuning for Text Classification.” In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 328–339. Melbourne, Australia: Association for Computational Linguistics, 2018. https://doi.org/10.18653/v1/P18-1031 

Kaplan, Jared, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. “Scaling Laws for Neural Language Models.” arXiv, January 22, 2020. https://doi.org/10.48550/arXiv.2001.08361.

Kenter, Tom, Melvin Wevers, Pim Huijnen, and Maarten de Rijke. “Ad Hoc monitoring of vocabulary shifts over time.” In Proceedings of the 24th ACM International on Conference on Information and Knowledge Management, 1191–1200. New York: Association for Computing Machinery, 2015. https://doi.org/10.1145/2806416.2806474 

Koselleck, Reinhart. “Stichwort: Begriffsgeschichte“, in: Begriffsgeschichten. Studien zur Semantik und Pragmatik der politischen und sozialen Sprache. Frankfurt a.M,  2006. 99-102.

Kutuzov, Andrey, Erik Velldal, and Lilja Øvrelid. “Contextualized Embeddings for Semantic Change Detection: Lessons Learned.” Northern European Journal of Language Technology 8, no. 1 (August 26, 2022). https://doi.org/10.3384/nejlt.2000-1533.2022.3478.

Li, Zhuohan, Eric Wallace, Sheng Shen, Kevin Lin, Kurt Keutzer, Dan Klein, Joseph E. Gonzalez. (2020, November). Train big, then compress: Rethinking model size for efficient training and inference of transformers. In 37th International Conference on Machine Learning, 5958–5968. PLMR 119, Online, 2020. http://proceedings.mlr.press/v119/li20m/li20m.pdf

Liu, Yinhan, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. “RoBERTa: A Robustly Optimized BERT Pretraining Approach.” arXiv, July 26, 2019. https://doi.org/10.48550/arXiv.1907.11692.

Manjavacas Arevalo, Enrique, and Lauren Fonteyn. “MacBERTh: Development and Evaluation of a Historically Pre-Trained Language Model for English (1450-1950).” In Proceedings of the Workshop on Natural Language Processing for Digital Humanities, 23–36. NIT Silchar, India: NLP Association of India (NLPAI), 2021. https://aclanthology.org/2021.nlp4dh-1.4.

Manjavacas Arevalo, Enrique, and Lauren Fonteyn. “Non-Parametric Word Sense Disambiguation for Historical Languages.” In Proceedings of the 2nd International Workshop on Natural Language Processing for Digital Humanities, 123–34. Taipei, Taiwan: Association for Computational Linguistics, 2022. https://aclanthology.org/2022.nlp4dh-1.16.

Nguyen, Dong and Jack Grieve. Do Word Embeddings Capture Spelling Variation?. In Proceedings of the 28th International Conference on Computational Linguistics, pages 870–881, Barcelona, Spain (Online). International Committee on Computational Linguistics, 2020. https://aclanthology.org/2020.coling-main.75/

Rombach, Robin, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. “High-Resolution Image Synthesis with Latent Diffusion Models.” arXiv, April 13, 2022. https://doi.org/10.48550/arXiv.2112.10752.

Rosin, Guy D., and Kira Radinsky. “Temporal Attention for Language Models.” arXiv, May 3, 2022. http://arxiv.org/abs/2202.02093.

Rosin, Guy D., Ido Guy, and Kira Radinsky. “Time Masking for Temporal Language Models.” In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, 833–41. Virtual Event AZ USA: ACM, 2022. https://doi.org/10.1145/3488560.3498529.

Schönemann, Peter H. “A generalized solution of the orthogonal Procrustes problem.” Psychometrika, 31, no. 1 (March 1966): 1–10. https://doi.org/10.1007/BF02289451 

Skinner, Quentin. "Meaning and Understanding in the History of Idea." History and Theory 8, no. 1 (1969): 3-53.

Su, Peng and Vijay-Shanker, K. “Investigation of improving the pre-training and fine-tuning of BERT model for biomedical relation extraction.” BMC Bioinformatics 23, no. 120 (April 4, 2022): 1–20. https://doi.org/10.1186/s12859-022-04642-w 

Suárez, Pedro Javier Ortiz, Benoît Sagot, and Laurent Romary. “Asynchronous pipeline for processing huge corpora on medium to low resource infrastructures.” 7th Workshop on the Challenges in the Management of Large Corpora (CMLC-7). Leibniz-Institut für Deutsche Sprache, 2019. https://doi.org/10.14618/IDS-PUB-9021  

Van der Wees, Marlies. “What’s in a Domain? Towards Fine-Grained Adaptation for Machine Translation”. PhD thesis, Universiteit van Amsterdam, 2017.

Van der Wal, Marijke & van Bree, Cor. "Geschiedenis van het Nederlands". Houten: Spectrum, 2008.

Wevers, Melvin and Marijn Koolen. “Digital begriffsgeschichte: Tracing semantic change using word embeddings. Historical Methods.” Historical Methods: A Journal of Quantitative and Interdisciplinary History 53, no. 4 (May 13, 2020): 226–243. https://doi.org/10.1080/01615440.2020.1760157

Xue, Linting, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. “ByT5: Towards a Token-Free Future with Pre-Trained Byte-to-Byte Models.” Transactions of the Association for Computational Linguistics 10 (March 25, 2022): 291–306. https://doi.org/10.1162/tacl_a_00461
