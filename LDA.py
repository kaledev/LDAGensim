# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
from nltk.corpus import stopwords
import pandas as pd

class LDAClassification:
    """
    A class that returns a Pandas Dataframe specifying LDA classification
    of documents, given a list of texts as input and the number of topics requested. Utilizes Gensim library.
    
    Code adopted initially from Selva Prabhakaran, 2018. Modified by Mike Kale, 2022.
    https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    https://www.mikekale.com
    ...

    Args
        df (DataFrame) : DataFrame of body text of the document
        stop_words_extend (str) : Add words into the stop words for exclusion
    """
    def __init__(self, stop_words_extend=None):
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(stop_words_extend)
        self.nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
        
        self.texts = None
        self.corpus = None
        self.id2word = None
        self.lda_model = None
        
    def train_find_optimal_topics(self, text_list, limit, start=2, step=3, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True):
        """
        Compute c_v coherence for various number of topics.
        Utilized to find the optimal number of topics with the provided training set.

        Args
            text_list (list of str) : string list of texts
            limit (int)
            start (int)
            step (int)
            random_state (int)
            update_every (int)
            chunksize (int)
            passes (int)
            alpha (str)
            per_word_topics (bool)

        Returns
            model_list (list of LDA) : List of LDA topic models
            coherence_values (list of int) : Coherence values corresponding to the LDA model with respective number of topics
        """
        
        texts, corpus = self.preprocessing(text_list, training=True)
        
        perplexity_values = []
        coherence_values = []
        model_list = []
        
        # Run through number of topics and generate gensim model
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=self.id2word,
                                                   num_topics=num_topics, 
                                                   random_state=random_state,
                                                   update_every=update_every,
                                                   chunksize=chunksize,
                                                   passes=passes,
                                                   alpha=alpha,
                                                   per_word_topics=per_word_topics)
            model_list.append(model)
            
            # Generate perplexity and coherence
            
            perplexitymodel = model.log_perplexity(corpus)
            perplexity_values.append(perplexitymodel)
            
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=self.id2word, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, perplexity_values, coherence_values
        
    def train(self, text_list, num_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True):
        """
        Compute c_v coherence for set number of topics.
        Utilized to find the optimal number of topics with the provided training set.

        Args
            text_list (list of str) : string list of texts
            limit (int)
            start (int)
            step (int)
            random_state (int)
            update_every (int)
            chunksize (int)
            passes (int)
            alpha (str)
            per_word_topics (bool)
        """
        
        self.texts, self.corpus = self.preprocessing(text_list, training=True)

        # Build LDA model
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                   id2word=self.id2word,
                                                   num_topics=num_topics, 
                                                   random_state=random_state,
                                                   update_every=update_every,
                                                   chunksize=chunksize,
                                                   passes=passes,
                                                   alpha=alpha,
                                                   per_word_topics=per_word_topics)
        
        # Print the keywords found for the n topics
        # print(lda_model.print_topics())
        
        # Compute Perplexity
        perplexity_lda = self.lda_model.log_perplexity(self.corpus) # a measure of how good the model is. lower the better.
        print('Perplexity: ', perplexity_lda)  
        
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.texts, dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('Coherence Score: ', coherence_lda)
        
    def get_topics(self):
        """
        Print the keywords found for the n topics
        """
        
        print(self.lda_model.print_topics())
    
    def predict(self, text_list):
        """
        Predict utilizing training data (or other)

        Args
            text_list (list of str) : string list of texts

        Returns
            df (DataFrame) : DataFrame of topic predictions
        """
        
        texts, corpus = self.preprocessing(text_list)
        df = self.get_dataframe_results(corpus, texts)
        return df
        
    def preprocessing(self, text_list, training=False):
        """
        Preprocessing text_list data for training / predictions

        Args
            text_list (list of str) : string list of texts
            training (bool) : this is training data, in which case the dictionary should be static

        Returns
            texts (list of str) : data lemmatized
            corpus (list of lists) : term document frequency
        """
        
        # Rename
        data = text_list

        # Simple cleansing - puncuation, etc.
        data_words = list(self.simple_cleanse(data))

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # See trigram example
        # print(trigram_mod[bigram_mod[data_words[0]]])
        
        # Remove Stop Words
        data_words_nostops = self.remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = self.make_bigrams(bigram_mod, data_words_nostops)
        
        # Form Trigrams (as alternative)
        # data_words_bigrams = self.make_trigrams(trigram_mod, bigram_mod, data_words_nostops)
        
        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        # View
        # print(data_lemmatized[:1])

        # Create Dictionary
        if training:
            # Dictionary should be static for the trained model
            self.id2word = corpora.Dictionary(data_lemmatized)

        # Create Texts
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [self.id2word.doc2bow(text) for text in texts]

        # View
        # print(corpus[:1])

        # Human readable format of corpus (term-frequency)
        # [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
        
        return texts, corpus

    def simple_cleanse(self, text_list):
        """
        # Simple cleansing - puncuation, etc.

        Args
            text_list (list of str) : string list of texts

        Yields
            str : cleansed data
        """
        
        for text in text_list:
            yield(gensim.utils.simple_preprocess(str(text), deacc=True))  # deacc=True removes punctuations
            
    def remove_stopwords(self, texts):
        """
        Remove stopwords from text

        Args
            texts (list of str) : string list of texts

        Returns
            list of lists : texts with stopwords removed
        """
        return [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words] for doc in texts]

    def make_bigrams(self, bigram_mod, texts):
        """
        Make bigrams out of texts

        Args
            texts (str) : string list of texts
            bigram_mod (Gensim) : bigram model

        Returns
            list of lists : bigrams of texts
        """
        
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(self, trigram_mod, bigram_mod, texts):
        """
        Make trigrams out of texts

        Args
            texts (str) : string list of texts
            bigram_mod (Gensim) : bigram model
            trigram_mod (Gensim) : trigram model

        Returns
            list of lists : trigrams of texts
        """
        
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """
        Text lemmatization

        Args
            texts (list of str) : string list of texts
            allowed_postags (list of str) : 

        Returns
            texts_out (list of str) : lemmatized texts
        """

        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    
    def get_dataframe_results(self, corpus, texts):
        """
        Results for returning predictions - DataFrame

        Args
            texts (list of str) : data lemmatized
            corpus (list of lists) : term document frequency

        Returns
            df_dominant_topic (DataFrame) : Dominant topic info by texts
            df_topic_distributions (DataFrame) : Distribution of topics by texts
        """
        
        #--------------------
        # Topic Dominance
        #--------------------
        
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(self.lda_model[corpus]):
            row = row_list[0] if self.lda_model.per_word_topics else row_list            
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.lda_model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    
        # Format
        df_dominant_topic = sent_topics_df.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        
        #--------------------
        # Topic Distributions
        #--------------------
        
        # Number of Documents for Each Topic
        topic_counts = df_dominant_topic['Dominant_Topic'].value_counts()

        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts/topic_counts.sum(), 4)

        # Topic Number and Keywords
        topic_num_keywords = df_dominant_topic.groupby(['Dominant_Topic', 'Keywords']).size().reset_index(name='Freq')
        topic_num_keywords.drop('Freq', axis=1, inplace=True)

        # Concatenate Column wise
        df_topic_distributions = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

        # Change Column names
        df_topic_distributions.columns = ['Topic', 'Keywords', 'Num_Documents', 'Perc_Documents']


        return df_dominant_topic, df_topic_distributions