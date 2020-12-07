import tensorflow.compat.v1 as tf
import os
import subprocess
import shutil
import csv
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import IPython

tf.get_logger().setLevel('ERROR')

from tapas.utils import tf_example_utils
from tapas.protos import interaction_pb2
from tapas.utils import number_annotation_utils
from tapas.scripts import prediction_utils

#i = 0


#i=25
#i = 30
#i = 50
i = ""

while os.path.exists('results'+str(i)):
	shutil.rmtree(r'results'+str(i))

print("BEFORE:",i)

os.makedirs('results'+str(i)+'/sqa/tf_examples', exist_ok=True)
os.makedirs('results'+str(i)+'/sqa/model', exist_ok=True)
with open('results'+str(i)+'/sqa/model/checkpoint', 'w') as f:
  f.write('model_checkpoint_path: "model.ckpt-0"')
for suffix in ['.data-00000-of-00001', '.index', '.meta']:
  shutil.copyfile(r'tapas_sqa_base/model.ckpt'+suffix, r'results'+str(i)+'/sqa/model/model.ckpt-0'+suffix)

max_seq_length = 512
vocab_file = "tapas_sqa_base/vocab.txt"
config = tf_example_utils.ClassifierConversionConfig(
	vocab_file=vocab_file,
	max_seq_length=max_seq_length,
	max_column_id=max_seq_length,
	max_row_id=max_seq_length,
	strip_column_names=False,
	add_aggregation_candidates=False,
	drop_rows_to_fit=True
)
converter = tf_example_utils.ToClassifierTensorflowExample(config)

def convert_interactions_to_examples(tables_and_queries):
  """Calls Tapas converter to convert interaction to example."""
  for idx, (table, queries) in enumerate(tables_and_queries):
    interaction = interaction_pb2.Interaction()
    for position, query in enumerate(queries):
      question = interaction.questions.add()
      question.original_text = query
      question.id = f"{idx}-0_{position}"
    for header in table[0]:
      interaction.table.columns.add().text = header
    for line in table[1:]:
      row = interaction.table.rows.add()
      for cell in line:
        row.cells.add().text = cell
    number_annotation_utils.add_numeric_values(interaction)
    for i in range(len(interaction.questions)):
      try:
        yield converter.convert(interaction, i)
      except ValueError as e:
        print(f"Can't convert interaction: {interaction.id} error: {e}")

def write_tf_example(filename, examples):
  with tf.io.TFRecordWriter(filename) as writer:
    for example in examples:
      writer.write(example.SerializeToString())

def predict(table_data, queries):
  table = [list(map(lambda s: s.strip(), row.split("|")))
           for row in table_data.split("\n") if row.strip()]
  examples = convert_interactions_to_examples([(table, queries)])
  write_tf_example("results"+str(i)+"/sqa/tf_examples/test.tfrecord", examples)
  write_tf_example("results"+str(i)+"/sqa/tf_examples/random-split-1-dev.tfrecord", [])
  subprocess.call(r'python tapas/tapas/run_task_main.py --task="SQA" --output_dir="results'+str(i)+'" --noloop_predict --test_batch_size='+str(len(queries))+' --tapas_verbosity="ERROR" --compression_type= --init_checkpoint="tapas_sqa_base/model.ckpt" --bert_config_file="tapas_sqa_base/bert_config.json" --mode="predict"', shell=True)

  results_path = "results"+str(i)+"/sqa/model/test_sequence.tsv"
  all_coordinates = []
  df = pd.DataFrame(table[1:], columns=table[0])
  print("FOLDER",results_path)
  with open(results_path) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
      coordinates = prediction_utils.parse_coordinates(row["answer_coordinates"])
      all_coordinates.append(coordinates)
      answers = ', '.join([table[row + 1][col] for row, col in coordinates])
      position = int(row['position'])

  return answers

acro = {}
acro['GP'] = "Games Played"
acro['MIN'] = "Minutes Played"
acro['PTS'] = "Points"
acro['FGM'] = "Field Goals Made"
acro['FGA'] = "Field Goals Attempted"
acro['FG%'] = "Field Goal Percentage"
acro['3PM'] = "Three Pointers Made"
acro['3PA'] = "Three Pointers Attempted"
acro['3P%'] = "Three Pointer Percentage"
acro['FTM'] = "Free Throws Made"
acro['FTA'] = "Free Throws Attempted"
acro['FT%'] = "Free Throws Percentage"
acro['OREB'] = "Offensive Rebounds"
acro['DREB'] = "Defensive Rebounds"
acro['REB'] = "Total Number of Rebounds"
acro['AST'] = "Assists"
acro['STL'] = "Steals"
acro['BLK'] = "Total Number of Blocks"
acro['TOV'] = "Turnovers"
acro['EFG%'] = "Effective Field Goal Percentage"
acro['TS%'] = "True Shooting Percentage"
acro['3-pt'] = "Three Pointer"
acro['#'] = "Rank"

def AQuA_v0(query_string,table,table_type,output,out_file_name):

    if table_type == "json_file":
        df = pd.read_json(table)
    elif table_type == "dataframe":
        df = table
    elif table_type == "csv_file":
        df = pd.read_json(tables[0].json())
    else:
        print("table_type not recognized.")

    mod = CountVectorizer(ngram_range=(1,3))
    query = [query_string]
    mod.fit_transform(query)
    query_list = mod.get_feature_names()

    for i in range(0,len(df)):
        row = df.loc[i,:]

        row_string = ""
        for j in row.values:
            row_string = row_string + " " + str(j)

        ans = mod.fit_transform([row_string])
        mod.get_feature_names()
        row_list = mod.get_feature_names()
        df.loc[i,'score_xyz'] = len(set(query_list)&set(row_list))
    content_snapshot = df.sort_values('score_xyz',ascending=False)[0:3]

    if output == "dataframe":
        return content_snapshot
    elif output == "csv_file":
        content_snapshot.to_csv(out_file_name,index=False,sep='|')


def AQuA_v1(query_string,table,table_type,output,out_file_name):
    #input specification
    if table_type == "json_file":
        df = pd.read_json(table)
    elif table_type == "dataframe":
        df = table
    elif table_type == "csv_file":
        df = pd.read_json(tables[0].json())
    else:
        print("table_type not recognized.")

    #replace accroynms in columns and rows with longer descriptions
    df = df.rename(columns = acro)
    df = df.replace('-',np.NaN)

    #analyze query and create trigram list
    mod = CountVectorizer(ngram_range=(1,3))
    query = [query_string]
    mod.fit_transform(query)
    query_list = mod.get_feature_names()
    print(query_list)

    #analyze table and create trigram list for each column
    #cut table only columns that match
    for i in df.columns:
        try:
            df[i] = df[i].astype(str).str.replace(',','').astype(float)
        except:
            pass
        col = df.loc[:,i]


        col_string = str(i)
        for j in col.values:
            col_string = col_string + " " + str(j)

        ans = mod.fit_transform([col_string])
        mod.get_feature_names()
        col_list = mod.get_feature_names()
        df.loc["sort",i] = len(set(query_list)&set(col_list))
    df.iloc[-1,2] = 1
    df.iloc[-1,1] = 1
    df1 = df.loc[:,df.loc['sort']>0].drop("sort",axis = 0).copy()

    #sort rows by trigram similarity and sort dataframe
    for i in range(0,len(df1)):
        row = df1.loc[i,:]

        row_string = ""
        for j in row.values:
            row_string = row_string + " " + str(j)

        ans = mod.fit_transform([row_string])
        mod.get_feature_names()
        row_list = mod.get_feature_names()
        #print(row_list)
        df1.loc[i,'score_xyz'] = len(set(query_list)&set(row_list))

    #df2 holds the rows where row trigram similarity is > 0
    try:
        df2 = df1.loc[df1['score_xyz']>0,:].drop('score_xyz',axis = 1)

    except:
        df2 = pd.DataFrame()

    #df3 holds all rows where row trigram similarity is = 0
    #df4 will keep top and bottom 3 for each column.

    df3 = df1.loc[df1['score_xyz']==0.0,:].drop('score_xyz',axis = 1)

    df4 = pd.DataFrame()
    for col in df3.columns:
        df4 = pd.concat([df4,df3.sort_values(col,ascending=False)[0:1]],axis=0)
        df4 = pd.concat([df4,df3.sort_values(col,ascending=True)[0:1]],axis=0)

    content_snapshot = pd.concat([df2,df4],axis=0)

    content_snapshot = content_snapshot.drop_duplicates()

    #output
    if output == "dataframe":
        return content_snapshot
    elif output == "csv_file":
        print(query_string)
        content_snapshot = content_snapshot.astype(str)
        content_snapshot = content_snapshot.replace(to_replace = "\.0+$",value = "", regex = True)
        display(content_snapshot)
        content_snapshot.to_csv(out_file_name,index=False,sep='|')


def AQuA_v2(query_string,table,table_type,output,out_file_name):
    #input specification
    if table_type == "json_file":
        df = pd.read_json(table)
    elif table_type == "dataframe":
        df = table
    elif table_type == "csv_file":
        df = pd.read_json(tables[0].json())
    else:
        print("table_type not recognized.")

    #replace accroynms in columns and rows with longer descriptions
    df = df.rename(columns = acro)
    df = df.replace('-',np.NaN)
    query_string = ' '.join([acro.get(word,word) for word in query_string.split(" ")])
    mod = CountVectorizer(ngram_range=(1,3))

    #check for "per" in the query
    if " per " in query_string:
        after = re.sub(r'[^\w\s]', '', query_string.split(" per ")[1])
        before = re.sub(r'[^\w\s]', '', query_string.split(" per ")[0].split(" ")[-1])

        mod.fit_transform([before])
        before_query_list = mod.get_feature_names()

        mod.fit_transform([after])
        after_query_list = mod.get_feature_names()

        #search table for before and after columns to apply divide to
        for i in df.columns:
            try:
                df[i] = df[i].astype(str).str.replace(',','').astype(float)
            except:
                pass
            col = df.loc[:,i]


            col_string = str(i)
            for j in col.values:
                col_string = col_string + " " + str(j)

            ans = mod.fit_transform([col_string])
            col_list = mod.get_feature_names()
            df.loc["sort",i] = len(set(before_query_list)&set(col_list))
            df.loc["sort2",i] = len(set(after_query_list)&set(col_list))
            df.loc["sort3",i] = len(set(before_query_list)&set(col_list)) + len(set(after_query_list)&set(col_list))
        df.iloc[-1,2] = 1
        df.iloc[-1,1] = 1
        df_before = df.loc[:,df.loc['sort']==df.loc['sort'].max()].drop(["sort","sort2"],axis = 0).copy()
        df_after = df.loc[:,df.loc['sort2']==df.loc['sort2'].max()].drop(["sort","sort2"],axis = 0).copy()
        df = df.drop(["sort2","sort3"],axis=0)

        df['_per_'] = df[df_before.columns[0]]/df[df_after.columns[0]]
        query_string = " ".join("who has the most points per games played?".split(" per ")[0].split(" ")[0:-1]) + " _per_"#     print(query_string)

    query = [query_string]
    mod.fit_transform(query)
    query_list = mod.get_feature_names()

    #analyze table and create trigram list for each column
    #cut table only columns that match
    for i in df.columns:
        try:
            df[i] = df[i].astype(str).str.replace(',','').astype(float)
        except:
            pass
        col = df.loc[:,i]


        col_string = str(i)
        for j in col.values:
            col_string = col_string + " " + str(j)

        ans = mod.fit_transform([col_string])
        mod.get_feature_names()
        col_list = mod.get_feature_names()
        df.loc["sort",i] = len(set(query_list)&set(col_list))
    df.iloc[-1,2] = 1
    df.iloc[-1,1] = 1
    col_name = str(df.loc['sort',df.loc['sort']==df.loc['sort',:].max()].index[0])
    df1 = df.loc[:,df.loc['sort']>0].drop("sort",axis = 0).copy()

    #sort rows by trigram similarity and sort dataframe
    for i in range(0,len(df1)):
        row = df1.loc[i,:]

        row_string = ""
        for j in row.values:
            row_string = row_string + " " + str(j)

        ans = mod.fit_transform([row_string])
        mod.get_feature_names()
        row_list = mod.get_feature_names()
        df1.loc[i,'score_xyz'] = len(set(query_list)&set(row_list))

    #df2 holds the rows where row trigram similarity is > 0
    try:
        df2 = df1.loc[df1['score_xyz']>0,:].drop('score_xyz',axis = 1)
    except:
        df2 = pd.DataFrame()

    #df3 holds all rows where row trigram similarity is = 0
    #df4 will keep top and bottom 3 for each column.
    df3 = df1.loc[df1['score_xyz']==0.0,:].drop('score_xyz',axis = 1)

    df4 = pd.DataFrame()
    for col in df3.columns:
        df4 = pd.concat([df4,df3.sort_values(col,ascending=False)[0:1]],axis=0)
        df4 = pd.concat([df4,df3.sort_values(col,ascending=True)[0:1]],axis=0)
    content_snapshot = pd.concat([df2,df4],axis=0)
    try:
        content_snapshot = content_snapshot.sort_values(col_name,ascending=False)
    except:
        pass
    content_snapshot = content_snapshot.drop_duplicates()

    if output == "dataframe":
        return [content_snapshot,query_string]
    elif output == "csv_file":
        content_snapshot = content_snapshot.astype(str)
        content_snapshot = content_snapshot.replace(to_replace = "\.0+$",value = "", regex = True)
        display(content_snapshot)
        content_snapshot.to_csv(out_file_name,index=False,sep='|')


def AQuA_v3(query_string,table,table_type,output,out_file_name):
    #input specification
    if table_type == "json_file":
        df = pd.read_json(table)
    elif table_type == "dataframe":
        df = table
    elif table_type == "csv_file":
        df = pd.read_json(tables[0].json())
    else:
        print("table_type not recognized.")

    #replace accroynms in columns and rows with longer descriptions
    df = df.rename(columns = acro)
    df = df.replace('-',np.NaN)
    query_string = ' '.join([acro.get(word,word) for word in query_string.split(" ")])

    #analyze query and create trigram list
    mod = CountVectorizer(ngram_range=(1,3))

    #check for "per" in the query
    if " per " in query_string:
        after = re.sub(r'[^\w\s]', '', query_string.split(" per ")[1])
        before = re.sub(r'[^\w\s]', '', query_string.split(" per ")[0].split(" ")[-1])

        mod.fit_transform([before])
        before_query_list = mod.get_feature_names()

        mod.fit_transform([after])
        after_query_list = mod.get_feature_names()

        #search table for before and after columns to apply divide to
        for i in df.columns:
            try:
                df[i] = df[i].astype(str).str.replace(',','').astype(float)
            except:
                pass
            col = df.loc[:,i]


            col_string = str(i)
            for j in col.values:
                col_string = col_string + " " + str(j)

            ans = mod.fit_transform([col_string])
            col_list = mod.get_feature_names()
            df.loc["sort",i] = len(set(before_query_list)&set(col_list))
            df.loc["sort2",i] = len(set(after_query_list)&set(col_list))
            df.loc["sort3",i] = len(set(before_query_list)&set(col_list)) + len(set(after_query_list)&set(col_list))
        df.iloc[-1,0] = 1
        df.iloc[-1,1] = 1
        df_before = df.loc[:,df.loc['sort']==df.loc['sort'].max()].drop(["sort","sort2"],axis = 0).copy()
        df_after = df.loc[:,df.loc['sort2']==df.loc['sort2'].max()].drop(["sort","sort2"],axis = 0).copy()
        df = df.drop(["sort2","sort3"],axis=0)

        df['_per_'] = df[df_before.columns[0]]/df[df_after.columns[0]]
        query_string = " ".join("who has the most points per games played?".split(" per ")[0].split(" ")[0:-1]) + " _per_"

    query = [query_string]
    mod.fit_transform(query)
    query_list = mod.get_feature_names()

    #analyze table and create trigram list for each column
    #cut table only columns that match
    for i in df.columns:
        try:
            df[i] = df[i].astype(str).str.replace(',','').astype(float)
        except:
            pass
        col = df.loc[:,i]


        col_string = str(i)
        for j in col.values:
            col_string = col_string + " " + str(j)

        ans = mod.fit_transform([col_string])
        mod.get_feature_names()
        col_list = mod.get_feature_names()
        df.loc["sort",i] = len(set(query_list)&set(col_list))
    df.iloc[-1,0] = 1
    df.iloc[-1,1] = 1
    col_name = str(df.loc['sort',df.loc['sort']==df.loc['sort',:].max()].index[0])
    df1 = df.loc[:,df.loc['sort']>0].drop("sort",axis = 0).copy()

    #sort rows by trigram similarity and sort dataframe
    for i in range(0,len(df1)):
        row = df1.loc[i,:]

        row_string = ""
        for j in row.values:
            row_string = row_string + " " + str(j)

        ans = mod.fit_transform([row_string])
        mod.get_feature_names()
        row_list = mod.get_feature_names()
        #print(row_list)
        df1.loc[i,'score_xyz'] = len(set(query_list)&set(row_list))

    #df2 holds the rows where row trigram similarity is > 0
    try:
        df2 = df1.loc[df1['score_xyz']>0,:].drop('score_xyz',axis = 1)

    except:
        df2 = pd.DataFrame()

    #df3 holds all rows where row trigram similarity is = 0
    #df4 will keep top and bottom 3 for each column.
    df3 = df1.loc[df1['score_xyz']==0.0,:].drop('score_xyz',axis = 1)

    df4 = pd.DataFrame()
    for col in df3.columns:
        df4 = pd.concat([df4,df3.sort_values(col,ascending=False)[0:1]],axis=0)
        df4 = pd.concat([df4,df3.sort_values(col,ascending=True)[0:1]],axis=0)
    content_snapshot = pd.concat([df2,df4],axis=0)
    content_snapshot = content_snapshot.drop_duplicates()

    #output
    if output == "dataframe":
        return [content_snapshot,query_string]
    elif output == "csv_file":
        content_snapshot = content_snapshot.astype(str)
        content_snapshot = content_snapshot.replace(to_replace = "\.0+$",value = "", regex = True)
        display(content_snapshot)
        content_snapshot.to_csv(out_file_name,index=False,sep='|')


def AQuA_v4(query_string,table_list,table_type,output,out_file_name):

    #pick the table to use
    mod = CountVectorizer(ngram_range=(1,3))

    score_list = []
    for items in table_list:
        mod.fit_transform(items.split("_"))
        table_count_list = mod.get_feature_names()
        query = [query_string]
        mod.fit_transform(query)
        query_count_list = mod.get_feature_names()
        score = len(set(query_count_list)&set(table_count_list))
        score_list.append(score)

    survivor_list = []
    for table,score in zip(table_list,score_list):
        if score == max(score_list):
            survivor_list.append(table)

    if len(survivor_list) == 1:
        table = survivor_list[0]
    else:
        #need to figure out best table to use is multiple pass first check
        max_score = 0
        table_name = survivor_list[0]
        for item in survivor_list:
            df = pd.read_csv(item)
            for i in df.columns:
                try:
                    df[i] = df[i].astype(str).str.replace(',','').astype(float)
                except:
                    pass
                col = df.loc[:,i]

                col_string = str(i)
                for j in col.values:
                    col_string = col_string + " " + str(j)

                ans = mod.fit_transform([col_string])
                mod.get_feature_names()
                col_list = mod.get_feature_names()
                score = len(set(query_count_list)&set(col_list))
                if score > max_score:
                    table_name = item
                    max_score = score
        table = table_name

	#input specification
    if table_type == "json_file":
        df = pd.read_json(table)
    elif table_type == "dataframe":
        df = table
    elif table_type == "csv_file":
        df = pd.read_csv(table)
    else:
        print("table_type not recognized.")

    #replace accroynms in columns and rows with longer descriptions
    df = df.rename(columns = acro)
    df = df.replace('-',np.NaN)
    query_string = ' '.join([acro.get(word,word) for word in query_string.split(" ")])

    #analyze query and create trigram list
    mod = CountVectorizer(ngram_range=(1,3))

    #check for "per" in the query
    if " per " in query_string:
        after = re.sub(r'[^\w\s]', '', query_string.split(" per ")[1])
        before = re.sub(r'[^\w\s]', '', query_string.split(" per ")[0].split(" ")[-1])

        mod.fit_transform([before])
        before_query_list = mod.get_feature_names()

        mod.fit_transform([after])
        after_query_list = mod.get_feature_names()

        #search table for before and after columns to apply divide to
        for i in df.columns:
            try:
                df[i] = df[i].astype(str).str.replace(',','').astype(float)
            except:
                pass
            col = df.loc[:,i]


            col_string = str(i)
            for j in col.values:
                col_string = col_string + " " + str(j)

            ans = mod.fit_transform([col_string])
            col_list = mod.get_feature_names()
            df.loc["sort",i] = len(set(before_query_list)&set(col_list))
            df.loc["sort2",i] = len(set(after_query_list)&set(col_list))
            df.loc["sort3",i] = len(set(before_query_list)&set(col_list)) + len(set(after_query_list)&set(col_list))
        df.iloc[-1,0] = 1
        df.iloc[-1,1] = 1
        df_before = df.loc[:,df.loc['sort']==df.loc['sort'].max()].drop(["sort","sort2"],axis = 0).copy()
        df_after = df.loc[:,df.loc['sort2']==df.loc['sort2'].max()].drop(["sort","sort2"],axis = 0).copy()
        df = df.drop(["sort2","sort3"],axis=0)

        df['_per_'] = df[df_before.columns[0]]/df[df_after.columns[0]]
        query_string = " ".join("who has the most points per games played?".split(" per ")[0].split(" ")[0:-1]) + " _per_"

    query = [query_string]
    mod.fit_transform(query)
    query_list = mod.get_feature_names()

    #analyze table and create trigram list for each column
    #cut table only columns that match
    for i in df.columns:
        try:
            df[i] = df[i].astype(str).str.replace(',','').astype(float)
        except:
            pass
        col = df.loc[:,i]


        col_string = str(i)
        for j in col.values:
            col_string = col_string + " " + str(j)

        ans = mod.fit_transform([col_string])
        mod.get_feature_names()
        col_list = mod.get_feature_names()
        df.loc["sort",i] = len(set(query_list)&set(col_list))
    df.iloc[-1,0] = 1
    df.iloc[-1,1] = 1
    col_name = str(df.loc['sort',df.loc['sort']==df.loc['sort',:].max()].index[0])
    df1 = df.loc[:,df.loc['sort']>0].drop("sort",axis = 0).copy()

    #sort rows by trigram similarity and sort dataframe
    for i in range(0,len(df1)):
        row = df1.loc[i,:]

        row_string = ""
        for j in row.values:
            row_string = row_string + " " + str(j)

        ans = mod.fit_transform([row_string])
        mod.get_feature_names()
        row_list = mod.get_feature_names()
        #print(row_list)
        df1.loc[i,'score_xyz'] = len(set(query_list)&set(row_list))

    #df2 holds the rows where row trigram similarity is > 0
    try:
        df2 = df1.loc[df1['score_xyz']>0,:].drop('score_xyz',axis = 1)
    except:
        df2 = pd.DataFrame()

    #df3 holds all rows where row trigram similarity is = 0
    #df4 will keep top and bottom 3 for each column.
    df3 = df1.loc[df1['score_xyz']==0.0,:].drop('score_xyz',axis = 1)

    df4 = pd.DataFrame()
    for col in df3.columns:
        df4 = pd.concat([df4,df3.sort_values(col,ascending=False)[0:1]],axis=0)
        df4 = pd.concat([df4,df3.sort_values(col,ascending=True)[0:1]],axis=0)

    content_snapshot = pd.concat([df2,df4],axis=0)
    content_snapshot = content_snapshot.drop_duplicates()

    #output
    if output == "dataframe":
        return [content_snapshot,query_string]
    elif output == "csv_file":
        content_snapshot = content_snapshot.astype(str)
        content_snapshot = content_snapshot.replace(to_replace = "\.0+$",value = "", regex = True)
        display(content_snapshot)
        content_snapshot.to_csv(out_file_name,index=False,sep='|')

def Table_Selector_v4(query_string,table_list,table_type,output,out_file_name):

    #pick the table to use
    mod = CountVectorizer(ngram_range=(1,3))

    score_list = []
    for items in table_list:
        mod.fit_transform(items.split("_"))
        table_count_list = mod.get_feature_names()
        query = [query_string]
        mod.fit_transform(query)
        query_count_list = mod.get_feature_names()
        score = len(set(query_count_list)&set(table_count_list))
        score_list.append(score)

    survivor_list = []
    for table,score in zip(table_list,score_list):
        if score == max(score_list):
            survivor_list.append(table)

    if len(survivor_list) == 1:
        table = survivor_list[0]
    else:
        #need to figure out best table to use is multiple pass first check
        max_score = 0
        table_name = survivor_list[0]
        for item in survivor_list:
            df = pd.read_csv(item)
            for i in df.columns:
                try:
                    df[i] = df[i].astype(str).str.replace(',','').astype(float)
                except:
                    pass
                col = df.loc[:,i]

                col_string = str(i)
                for j in col.values:
                    col_string = col_string + " " + str(j)

                ans = mod.fit_transform([col_string])
                mod.get_feature_names()
                col_list = mod.get_feature_names()
                score = len(set(query_count_list)&set(col_list))
                if score > max_score:
                    table_name = item
                    max_score = score
        table = table_name

	#input specification
    if table_type == "json_file":
        df = pd.read_json(table)
    elif table_type == "dataframe":
        df = table
    elif table_type == "csv_file":
        df = pd.read_csv(table)
    else:
        print("table_type not recognized.")

    content_snapshot = df

    #output
    if output == "dataframe":
        return [content_snapshot,query_string]
    elif output == "csv_file":
        content_snapshot = content_snapshot.astype(str)
        content_snapshot = content_snapshot.replace(to_replace = "\.0+$",value = "", regex = True)
        display(content_snapshot)
        content_snapshot.to_csv(out_file_name,index=False,sep='|')

def AQuA_v5(query_string,table_list,table_type,output,out_file_name):

    #pick the table to use
    mod = CountVectorizer(ngram_range=(1,3))

    score_list = []
    for items in table_list:
        mod.fit_transform(items.split("_"))
        table_count_list = mod.get_feature_names()
        query = [query_string]
        mod.fit_transform(query)
        query_count_list = mod.get_feature_names()
        score = len(set(query_count_list)&set(table_count_list))
        score_list.append(score)

    survivor_list = []
    for table,score in zip(table_list,score_list):
        if score == max(score_list):
            survivor_list.append(table)

    if len(survivor_list) == 1:
        table = survivor_list[0]
    else:
        #need to figure out best table to use is multiple pass first check
        max_score = 0
        table_name = survivor_list[0]
        for item in survivor_list:
            df = pd.read_csv(item)
            for i in df.columns:
                try:
                    df[i] = df[i].astype(str).str.replace(',','').astype(float)
                except:
                    pass
                col = df.loc[:,i]

                col_string = str(i)
                for j in col.values:
                    col_string = col_string + " " + str(j)

                ans = mod.fit_transform([col_string])
                mod.get_feature_names()
                col_list = mod.get_feature_names()
                score = len(set(query_count_list)&set(col_list))
                if score > max_score:
                    table_name = item
                    max_score = score
        table = table_name

	#input specification
    if table_type == "json_file":
        df = pd.read_json(table)
    elif table_type == "dataframe":
        df = table
    elif table_type == "csv_file":
        df = pd.read_csv(table)
    else:
        print("table_type not recognized.")

    #replace accroynms in columns and rows with longer descriptions
    df = df.rename(columns = acro)
    df = df.replace('-',np.NaN)
    query_string = ' '.join([acro.get(word,word) for word in query_string.split(" ")])

    #analyze query and create trigram list
    mod = CountVectorizer(ngram_range=(1,3))

    #check for "per" in the query
    if " per " in query_string:
        after = re.sub(r'[^\w\s]', '', query_string.split(" per ")[1])
        before = re.sub(r'[^\w\s]', '', query_string.split(" per ")[0].split(" ")[-1])

        mod.fit_transform([before])
        before_query_list = mod.get_feature_names()

        mod.fit_transform([after])
        after_query_list = mod.get_feature_names()

        #search table for before and after columns to apply divide to
        for i in df.columns:
            try:
                df[i] = df[i].astype(str).str.replace(',','').astype(float)
            except:
                pass
            col = df.loc[:,i]


            col_string = str(i)
            for j in col.values:
                col_string = col_string + " " + str(j)

            ans = mod.fit_transform([col_string])
            col_list = mod.get_feature_names()
            df.loc["sort",i] = len(set(before_query_list)&set(col_list))
            df.loc["sort2",i] = len(set(after_query_list)&set(col_list))
            df.loc["sort3",i] = len(set(before_query_list)&set(col_list)) + len(set(after_query_list)&set(col_list))
        df.iloc[-1,0] = 1
        df.iloc[-1,1] = 1
        df_before = df.loc[:,df.loc['sort']==df.loc['sort'].max()].drop(["sort","sort2"],axis = 0).copy()
        df_after = df.loc[:,df.loc['sort2']==df.loc['sort2'].max()].drop(["sort","sort2"],axis = 0).copy()
        df = df.drop(["sort2","sort3"],axis=0)

        df['_per_'] = df[df_before.columns[0]]/df[df_after.columns[0]]
        query_string = " ".join("who has the most points per games played?".split(" per ")[0].split(" ")[0:-1]) + " _per_"

    query = [query_string]
    mod.fit_transform(query)
    query_list = mod.get_feature_names()

    #analyze table and create trigram list for each column
    #cut table only columns that match
    for i in df.columns:
        try:
            df[i] = df[i].astype(str).str.replace(',','').astype(float)
        except:
            pass
        col = df.loc[:,i]


        col_string = str(i)
        for j in col.values:
            col_string = col_string + " " + str(j)

        ans = mod.fit_transform([col_string])
        mod.get_feature_names()
        col_list = mod.get_feature_names()
        df.loc["sort",i] = len(set(query_list)&set(col_list))
    df.iloc[-1,0] = 10
    df.iloc[-1,1] = 10
    col_name = str(df.loc['sort',df.loc['sort']==df.loc['sort',:].max()].index[0])
    cut_off = df.loc['sort',:].astype("int").nlargest(5).min()
    df1 = df.loc[:,df.loc['sort']>cut_off].drop("sort",axis = 0).copy()
    #print(df1)

    #sort rows by trigram similarity and sort dataframe
    for i in range(0,len(df1)):
        row = df1.loc[i,:]

        row_string = ""
        for j in row.values:
            row_string = row_string + " " + str(j)

        ans = mod.fit_transform([row_string])
        mod.get_feature_names()
        row_list = mod.get_feature_names()
        #print(row_list)
        df1.loc[i,'score_xyz'] = len(set(query_list)&set(row_list))

    #df2 holds the rows where row trigram similarity is > 0
    try:
        df2 = df1.loc[df1['score_xyz']>0,:].drop('score_xyz',axis = 1)
    except:
        df2 = pd.DataFrame()

    #df3 holds all rows where row trigram similarity is = 0
    #df4 will keep top and bottom 3 for each column.
    df3 = df1.loc[df1['score_xyz']==0.0,:].drop('score_xyz',axis = 1)

    df4 = pd.DataFrame()
    for col in df3.columns:
        df4 = pd.concat([df4,df3.sort_values(col,ascending=False)[0:1]],axis=0)
        df4 = pd.concat([df4,df3.sort_values(col,ascending=True)[0:1]],axis=0)

    content_snapshot = pd.concat([df2,df4],axis=0)
    content_snapshot = content_snapshot.drop_duplicates()

    #output
    if output == "dataframe":
        return [content_snapshot,query_string]
    elif output == "csv_file":
        content_snapshot = content_snapshot.astype(str)
        content_snapshot = content_snapshot.replace(to_replace = "\.0+$",value = "", regex = True)
        display(content_snapshot)
        content_snapshot.to_csv(out_file_name,index=False,sep='|')
