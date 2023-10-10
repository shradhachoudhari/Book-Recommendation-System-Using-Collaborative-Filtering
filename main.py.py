#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from scipy import sparse

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[3]:


book=pd.read_csv('/Users/shraddhalipane/Desktop/MyProject/MyBooks/MyBooks/Books.csv', sep=',',encoding="ISO-8859-1",index_col= None, low_memory= False , dtype={'book_author': str})
rating=pd.read_csv('/Users/shraddhalipane/Desktop/MyProject/MyBooks/MyBooks/Ratings.csv')
user=pd.read_csv('/Users/shraddhalipane/Desktop/MyProject/Users.csv')


# In[4]:


print(book.shape)
print(rating.shape)
print(user.shape)


# In[5]:


book.head()


# # Data Preprocessing

# In[6]:


book.isnull().sum()


# In[7]:


missing_publishers = book[book['Publisher'].isnull()]
df_cleaned = book.dropna(subset=['Publisher'])
df_filled = book.fillna({'Publisher': 'Unknown'})
book['Publisher'].fillna(book['book_author'], inplace=True)
book.isnull().sum()


# In[8]:


missing_author = book[book['book_author'].isnull()]
ds_cleaned = book.dropna(subset=['book_author'])
df_filled_with= book.fillna({'book_author': 'Unknown'})
book['book_author'].fillna(book['Publisher'], inplace=True)

book.isnull().sum()


# In[9]:


book.dropna(inplace=True)
book.isnull().sum()


# In[10]:


rating.isnull().sum()


# In[11]:


user.isnull().sum()


# In[12]:


user.count()


# In[13]:


user= pd.DataFrame(user)
user=user.fillna(0)
user.isnull().sum() 


# In[14]:


user.count()


# In[ ]:





# In[15]:


book.to_csv('book.csv',index=False)


# In[16]:



rating.to_csv('rating.csv',index=False)


# In[17]:


book.publication_year.value_counts(dropna=False).sort_index().plot(kind='barh',figsize=(15,16))
plt.show()


# In[18]:


combined_book_rating=pd.merge(book,rating,on='isbn')
columns= ['publication_year','Publisher','book_author','image_url_s','image_url_m','image_url_l']
combined_book_rating = combined_book_rating.drop(columns, axis=1)
combined_book_rating.head()


# In[ ]:





# In[ ]:





# In[19]:


combined_book_rating.groupby('book_title')['book_rating'].mean().sort_values(ascending=False).head()


# In[20]:


combined_book_rating.groupby('book_title')['book_rating'].count().sort_values(ascending=False).head()


# In[21]:


#we will calculate the average rating using mean function

#avg_rating_df =combined_book_rating.groupby('book_title').mean()['book_rating'].reset_index()
#avg_rating_df.rename(columns={'book_rating':'avg_rating'},inplace=True)
#avg_rating_df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


#combined book data with rating data

combine_book_rating = pd.merge(rating, book, on='isbn',how='inner')
columns= ['publication_year','Publisher','book_author','image_url_s','image_url_m','image_url_l']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
combine_book_rating.head()


# In[23]:


book_totalratingCount=(combined_book_rating.groupby(by=['book_title'])['book_rating'].
                  count().
                  reset_index().
                  rename(columns={ 'book_rating':'totalRatingCount'})
                  [['book_title','totalRatingCount']]
                     )
book_totalratingCount.head()


# In[24]:


rating_with_totalRatingCount = combine_book_rating.merge(book_totalratingCount, left_on = 'book_title', right_on = 'book_title', how = 'left')
rating_with_totalRatingCount.head()


# In[25]:


pd.set_option('display.float_format', lambda x:'%.3f' %x)
print(book_totalratingCount['totalRatingCount'].describe())


# In[26]:


print(book_totalratingCount['totalRatingCount'].quantile(np.arange(.9, 1,.01))),


# In[27]:


popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
rating_popular_book.head()


# In[28]:


book.duplicated().sum()     


# In[29]:


#us and canada city

combined = rating_popular_book.merge(user, left_on = 'user_id', right_on = 'user_id', how = 'left')
combined_df=combined[combined['location'].str.contains('usa|canada')]
combined_df=combined_df.drop('age',axis=1)
combined_df.head()


# In[ ]:





# In[30]:


#aggregate rating.

agg_rating=combined.groupby('book_title').agg(mean_rating=('book_rating','mean'), number_of_rating=('book_rating','count')).reset_index()
                                              


# In[31]:


agg_ratings=agg_rating[agg_rating['number_of_rating']>100]
agg_ratings.info()


# In[32]:


agg_ratings.sort_values(by='number_of_rating',ascending=False).head()


# In[33]:


sns.jointplot(x='mean_rating', y='number_of_rating',data=agg_ratings)


# In[34]:


df=pd.merge(combined, agg_ratings[['book_title']],on='book_title',how='inner')
df.info()


# In[35]:


df


# In[36]:


#number of users
print('rating dataset has',df['user_id'].nunique(),'unique users')

#number of books
print('rating dataset has',df['book_title'].nunique(),'unique books')

#number of ratings
print('rating dataset has',df['book_rating'].nunique(),'unique rating')

#list unique rating
print('unique rating are',sorted(df['book_rating'].unique()))


# # user- book matrix

# In[37]:


combined_df.head()


# In[38]:


from scipy.sparse import csr_matrix

combined_df=combined_df.drop_duplicates(['user_id','book_title'])
matrix=combined_df.pivot_table(index='book_title',columns='user_id',values='book_rating').fillna(0)
matrix_df = csr_matrix(matrix.values)

matrix.head()


# # Applying Cosine Similarity

# # item similarity matrix using cosine similarity
# 

# In[39]:


from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(matrix)
similarity_scores.shape


# In[40]:


def recommend(book_name):
    # index fetch
    index = np.where(matrix.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = book[book['book_title'] == matrix.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('book_title')['book_title'].values))
        item.extend(list(temp_df.drop_duplicates('book_title')['book_author'].values))
        item.extend(list(temp_df.drop_duplicates('book_title')['image_url_m'].values))
        
        data.append(item)
    
    return data


# In[41]:


recommend('1984')


# In[42]:


item_similarity = cosine_similarity(matrix.T)


# In[43]:


# Convert the similarity matrix into a DataFrame
item_similarity_df = pd.DataFrame(item_similarity, index=matrix.columns, columns=matrix.columns)


# # Applying KNN Algorithm

# In[44]:


knn_model=NearestNeighbors( metric ='cosine', algorithm='brute',n_neighbors=7, n_jobs=-1)
knn_model.fit(matrix_df)


# In[45]:


print("Algorithm:", knn_model.algorithm)
print("Metric:", knn_model.metric)
print("Leaf size:", knn_model.leaf_size)
print("Number of neighbors:", knn_model.n_neighbors)
print("P:", knn_model.p)
print("Radius:", knn_model.radius)


# In[46]:


# Get top 10 nearest neighbors 
indices=knn_model.kneighbors(matrix.loc[['1984']], 10, return_distance=False)

# Print the recommended books
print("Recommended Books:")

for index, value in enumerate(matrix.index[indices][0]):
    print((index+1),". ",value)


# In[47]:


combined_df.head()


# In[ ]:




