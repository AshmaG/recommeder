import time

## Create predictions for all movies and users
st = time.time()
preds_ = []
movieids = np.sort(df_ratings_XS['Movie_Id'].unique())
userids = np.sort(df_ratings_XS['Cust_Id'].unique())

for mo in movieids:
    preds_.append([svd.predict(cc,mo).est for cc in userids])

elapsed_ = time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-st))
print("Predictions completed in: {}".format(elapsed_))

## Transform predictions into dataframe only for missing ratings
st = time.time()

# Identify movies rated already
df_ratings_XS['K'] = df_ratings_XS['Cust_Id'].astype(str)+'_'+df_ratings_XS['Movie_Id'].astype(str)
Keys_rated = df_ratings_XS['K'].unique()
Keys_rated = set(Keys_rated)

# Identify missing ratings
preds_df_data = []
for i in range(len(preds_)):
    for j in range(len(preds_[i])):
        if str(userids[j])+'_'+str(movieids[i]) not in Keys_rated:
            preds_df_data.append([movieids[i], userids[j], preds_[i][j]])
preds_df = pd.DataFrame(data=preds_df_data, columns=['Movie_Id', 'Cust_Id', 'Predicted_Rating'])

elapsed_ = time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-st))
print("Dataframe creation completed in: {}".format(elapsed_))

print(preds_df.head(3))