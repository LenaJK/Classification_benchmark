import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from tqdm import tqdm
import pickle
from numpy.random import PCG64, SeedSequence, Generator


from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.decomposition import PCA

import matplotlib.pylab as pylab
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from Bio import pairwise2
from Bio import SeqIO
from Bio import Align
from Bio.SubsMat.MatrixInfo import blosum62
from Bio.Align import substitution_matrices

from pyxdameraulevenshtein import damerau_levenshtein_distance_ndarray as DLD
from scipy.spatial import distance
from sklearn.manifold import MDS

from gensim.models import word2vec
from gensim.models import doc2vec
from pyfasta import Fasta




class PfamClassifier(object):    
############################ INPUT VARIABLES ##################################

    def __init__(self, families,familypath=None, embedding=None, seed=None, initial_entropy=None):
# families have to be provided in form of fasta files inside a folder with path familypath

# If seed is None, the initial entropy will be generated automatically and randomly.
# Reproducible results can be obtained by passing the same seed. Alternatively, the initial
# entropy itself can be passed. This may be useful in the case when no seed was passed, the
# class was used, and then you wish to reproduce its results later. In that case, the 
# initial_entropy variable will be available as a class attribute and can be passed to a new
# instance.
        if initial_entropy is None:
            self.initial_entropy = np.random.SeedSequence(seed)
        else:
            self.initial_entropy = initial_entropy
        self.rng = Generator(PCG64(self.initial_entropy))


# If no path exists proteinsequences becomes the default path containting the 78 Pfam families
        if familypath is None:
            familypath = 'proteinsequences'
                
        self.families = families
        self.embedding = embedding
        self.seed = seed
        self.familypath = familypath       
        
# Load the training set for the chosen embedding into self.pv     
        # if self.embedding == 'word2vec_swissprot':
            
        self.pv = load_protvec(str(embedding))
 #        elif self.embedding == 'word2vec_uniref'
#             from gensim.models import word2vec
#             self.pv = load_protvec('uniref50_size200_window25.model')
            
# Turn family sequences into proteinvectors and store inside proteins dataframe
# proteins[columns = ['protein_name', 'sequence', family_name','n_seqs','seq_length','protein_vector','labels']]
        self.proteins = self.input_df_generator()

# Generate results directories
        if not os.path.exists(os.path.join('results',str(self.embedding),'Classification')):
            os.makedirs(os.path.join('results',str(self.embedding),'Classification'))

    
# Define initialisation methods
# Function that generates protein dataframe containing the sequences, 
# n_seq, seq_length, proteinvector, label of all proteins
    def input_df_generator(self):
        dfs = []
# go through every provided family and transform its sequences to proteinvectors
        for name in self.families:
            fasta = Fasta(os.path.join(self.familypath,name+'_seed.txt'))
            keys = []
            values = []
            for k in fasta.keys():
                keys.append(k)
                values.append(str(fasta[k]).upper())
            df = pd.DataFrame(data={'protein_name': keys, 'sequence': values})
            df['family_name'] = name.split('_')[0]
            df['n_seqs'] = len(df)
            df['seq_length'] = df['sequence'].apply(len)
# proteinvectors get generated and stores in proteins dataframe
#             df['protein_vector'] = df['sequence'].apply(self.pv.get_vector)
            dfs.append(df)
        proteins = pd.concat(dfs, ignore_index=True)
        proteins['labels']=proteins.index


# pass it the protein name you wish to modify and the list of replacement characters
# sequence_modifyer function will modify the dataframe
        def sequence_modifyer(protein, replace_with):
            s = str(proteins[proteins['protein_name']==protein]['sequence'].values[0])
            for r in replace_with:
                s = s.replace('X', r, 1)
            proteins.loc[proteins[proteins['protein_name']==protein].index,'sequence'] = s

# Replace all the X in the sequences 
# For 'DPO3E_HAEIN/8-173', replace the first X with F and the second with V

# When using default 78 families it replaces the X in sequences with other letters:
#     if any(proteins['protein_name'].values=='DPO3E_HAEIN/8-173'):
        if 'DPO3E_HAEIN/8-173' in proteins['protein_name'].values: 
            sequence_modifyer('DPO3E_HAEIN/8-173', ['F','V'])
    # For 'Q7DBE1_ECO57/9-110', in order of appearance: A I L A
        if 'Q7DBE1_ECO57/9-110' in proteins['protein_name'].values:
            sequence_modifyer('Q7DBE1_ECO57/9-110', ['A','I','L','A'])
    # For 'A0A0R3QTM0_9BILA/1-112' - KWG.WPE.D.VWFHVK
        if 'A0A0R3QTM0_9BILA/1-112' in proteins['protein_name'].values:
            sequence_modifyer('A0A0R3QTM0_9BILA/1-112', ['K','W','G','W','P','E','D','V','W','F','H','V','D','K'])

        proteins['protein_vector'] = proteins['sequence'].apply(self.pv.get_vector)
        
# Return dataframe proteins
        return proteins
    
    
    def   family_statistics(self): 
#         ????????????????????????????????????????
        # these are the Pfam identifiers the user needs to supply
        # separately, users need to supply the seed sequences in fasta files!
        pfam_ids = ['PFNNN', 'PFNNN2']
        # ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/database_files/pfamA.txt.gz
        # ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/database_files/clan_membership.txt.gz
        clans = pd.read_csv('clan_membership.txt', sep='\t', header=None)
        pfam = pd.read_csv('pfamA.txt', sep='\t', header=None)
        pfam.rename(columns={0: 'acc', 
                         1: 'ID', 
                         23: 'num_seed', 
                         24: 'num_full', 
                         28: 'num_archs', 
                         29: 'num_species', 
                         30: 'num_structs', 
                         33: 'seq_len', 
                         34: 'percent_id', 
                         35: 'coverage', 
                         40: 'num_uniprot'}, inplace=True)
        pfam.drop([x for x in range(0,46) if x in pfam.columns], axis=1, inplace=True)
        pfam = pd.merge(pfam, clans, left_on='acc', right_on=1, how='left')
        pfam.rename(columns={0: 'clan'}, inplace=True)
        pfam.drop([1], axis=1, inplace=True)
        pfam = pfam[['acc', 'ID', 'clan', 'num_seed', 'seq_len', 'percent_id', 'coverage', 'num_full', 'num_uniprot', 'num_archs', 'num_species', 'num_structs']]
        pfam = pfam[pfam.acc.isin(pfam_ids)].sort_values(['num_seed', 'seq_len', 'percent_id'])
        
        return pfam
    
############################ CLASSIFICATION PROCESSES & OUTPUTS ##################################

############################ run_classification ##################################
    def run_classification(self, familyselection='all', clustering='kmeans', k=1, DM=None):

#   ------------------------------ INPUT ------------------------------
# 1.familyselection: 'all' or list of at least 2 family names e.g. ['PF08213','PF01134',...]
# - If not provided it automatically selects all families in self.families
# - Otherwise it choses the proteins from the list of family names provided in familyselection
# 2. clustering: kmeans or kNN
# - If'kmeans' (default): perform kmeans clustering (then can't provide distancematrix or k)
# - If kNN': perform k nearest neighbour clustering 
#   3. k: 1,2,3,4...
#   - Amount of nearest neighbours provided to kNN: k=1 by default
#   4. DM (distancematrix): (normed) PairwiseAlignment distance'PA''nPA', (normed) Damerau-Levenshtein distance'DA''nDA'
#   - pass the filename (as a string) that stores the distance matrix (either ending in .npy or .pkl)
#   - provides precomputed distance matrix to kNN

#   ------------------------------ OUTPUTS  ------------------------------
# 1. CMDF: Dataframe that stores the outputs of the confusion matrix for each pair of familes
# - Familyname A, Familyname B, AA,AB,BA,BB
# 2. Mclass_score: Calculates the means misclassification score of all 
#                  pairwise family comparisons Mclass_score
# 3. PF_misclf: Dataframe that stores all misclassified proteins and the families 
#               they got wrongly identified as
# 4. Misclf_proteins: List of misclassified proteins
# 5. Saves the heatmaps that show the misclassified families (AB and BA) 
#    in results/embedding/Clustering


#   ---------------------------- INITIALISATION ---------------------------------
# Familyselector is automatically set to all to include all families 
# You can also specifiy a list of families by setting familyselector
        if familyselection=='all':
            famlist = self.families
# If familyselection is a list then only select proteins from that list in variable selector            
        if isinstance(familyselection, list):
            famlist = familyselection
        selector = self.proteins['family_name'].isin(famlist)    
        
# Cross validated between family comparison for  performance (measured via confusion matrix)
        CMDF = pd.DataFrame(columns = ['familyA', 'familyB', 'AA','AB','BA','BB'])
        PF_misclf = pd.DataFrame(0, index=self.proteins[selector]['protein_name'].unique(),columns=self.proteins[selector]['family_name'].unique())
# np array of all families compare
        families = self.proteins[selector]['family_name'].unique()

# This function needs at least 2 families to do pairwise classification
        if len(families)<2:
            raise ValueError('You need to provide more than 1 family in familysection to this function')

# set nr of crossvalidations to the smallest family size
        crossval = min(self.proteins[selector]['n_seqs'].values)

#   ---------------------------- CLUSTERING = KMEANS  ---------------------------------
# clustering can be 'kmeans' or 'kNN' (can specify k - default k value is 1)
        if clustering == 'kmeans':
            if DM is not None:
                raise ValueError('Kmeans does not accept a precomputed distance matrix - use kNN instead')

# Pairwise family comparison:
            for i,fam1 in enumerate(families):
                for j,fam2 in enumerate(families[i+1:]): 
# Select crossval random proteinvectors as seeds from each of the 2 families
                    seeds1 = self.rng.choice(self.proteins[self.proteins['family_name']==fam1].index.values,
                                              crossval,replace=False)
                    seeds2 = self.rng.choice(self.proteins[self.proteins['family_name']==fam2].index.values,
                                              crossval,replace=False)
# vals contains the proteinvectors of the 2 chosen families
                    selected_rows = (self.proteins['family_name']==fam1) | (self.proteins['family_name']==fam2)
                    vals = np.vstack(self.proteins[selected_rows]['protein_vector'].values)
# calculate the kmeans accuracy crossval times and take the average
                    CM_average = []
                    for n in range(crossval):
# store the 2 random cluster seeds for each validation in ndarray and pass it to kmeans
                        seeds = np.vstack([self.proteins.loc[seeds1[n],'protein_vector'],
                                           self.proteins.loc[seeds2[n],'protein_vector']])
                        clust = KMeans(n_clusters=2, init=seeds, n_init=1).fit(vals)
# Identify which family name is which cluster label via linear sum assignment
# First define a benefit function M and then maximize it
                        clusterlabels = np.unique(clust.labels_)
                        M = np.zeros([2, 2])
                        for p,fam in enumerate([fam1, fam2]):
                            for q,lab in enumerate(clusterlabels):
                                M[p,q] = sum((self.proteins[selected_rows]['family_name'].values==fam)
                                            & (clust.labels_==lab))
# Identify which clusters belong to which families by maximising befit_fct M
                        ii, jj = linear_sum_assignment(M, maximize=True)
# Create 2 dictionaries that map family names to cluster labels and vice versa
                        family_dict = {}
                        label_dict = {}
                        for p, q in zip(ii,jj):
                            family_dict[[fam1, fam2][p]] = clusterlabels[q]
                            label_dict[clusterlabels[q]] = [fam1, fam2][p]

# Calculate the confusion matrix by passing it the true labels and the matching kmeans labels
                        Truelabels = self.proteins[selected_rows]['family_name'].values

                        misclf_proteins = self.proteins[selected_rows]['protein_name'].values
# Dataframe PF_misclf stores which proteins get misclassified in which families
                        for p, tl, cl in zip(misclf_proteins, Truelabels, [label_dict[l] for l in clust.labels_]):
                            if tl != cl:                            
                                PF_misclf.loc[p,cl] += 1
                        self.PF_misclf = PF_misclf
# This would calculate amount of misclassified proteins and store it in another column of proteins
#                         proteins.loc[selected_rows,'Misclassified'] += int([Truelabels == [label_dict[l] for l in clust.labels_]])

# Calculate the confusion matrix CM
                        CM = confusion_matrix(Truelabels, [label_dict[l] for l in clust.labels_],normalize='true',labels=[fam1, fam2])
                        CM_average.append(CM.flatten())

# Store the average confusion matrix values (4 for comparing 2 families) in CMDF
# AA = True positive, AB = False negative, BA = False positive, BB = True negative
                    aa,ab,ba,bb = np.mean(np.vstack(CM_average),axis=0)
# Store result in CMDF dataframe
                    CMDF = CMDF.append({'familyA':fam1, 'familyB':fam2, 'AA':aa, 
                                        'AB':ab, 'BA':ba, 'BB':bb},ignore_index=True)
# Calculate the misclassification class score 
            Mclass_score = np.mean(CMDF['AB'].values+CMDF['BA'].values)
            self.CMDF = CMDF
#             CMDF.to_pickle("./ConfusionMatrix.pkl",protocol=3)

            heatmapname = 'Kmeans_betweenfamily_misclassified.png'
            clust_method = 'Kmeans'
            heatmap_title = 'Kmeans pairwise misclassification'
        
#   ---------------------------- CLUSTERING = KNN  ---------------------------------
        elif clustering == 'kNN':
# generate dictionaries that transform family names to numbers and back
            fam_to_nr = {l:i for i,l in enumerate(families)}
            nr_to_fam = {v:k for k,v in fam_to_nr.items()}
#   ---------------------------- NO DISTANCE MATRIX  ---------------------------------
# If we don't pass a precomputed distance matrix to the function
            if DM is None:
# Compare all families with each other
                for i,fam1 in enumerate(families):
                    for j,fam2 in enumerate(families[i+1:]): 
# Initiate true and predicted family labels
                        true_labels = []
                        pred_labels = []
# Calculate the proteinvectors of the 2 chosen families in vals
                        selected_rows = (self.proteins['family_name']==fam1) | (
                            self.proteins['family_name']==fam2)
                        vals = np.vstack(self.proteins[selected_rows]['protein_vector'].values)
# Turn family names into numbers
                        nr = self.proteins[selected_rows]['family_name'].map(
                            lambda x: fam_to_nr[x]).values

# Go through every selected protein vector (of the 2 families) and calculate which family it
# gets classified into if all other vectors keep their labels
# k can be modified to ask more than 1 neighbour
                        for v in range(vals.shape[0]):
                            neigh = kNN(n_neighbors=k)
# Forget the family label of the selected protein
                            X = np.vstack([vals[:v,:], vals[v+1:,:]])
                            y = np.hstack([nr[:v], nr[v+1:]])
                            neigh.fit(X, y)
                            y_pred_nr = neigh.predict(vals[v,:].reshape([1,-1]))[0]
                            y_pred_lab = nr_to_fam[y_pred_nr]
                            y_true_nr = nr[v]
                            y_true_lab = nr_to_fam[y_true_nr]
                            # cm_df.loc[y_true_lab, y_pred_lab] += 1
                            true_labels.append(y_true_lab)
                            pred_labels.append(y_pred_lab)

                        misclf_proteins = self.proteins[selected_rows]['protein_name'].values

# Calcualte which proteins get misclassified into which families and store info in PF_misclf
                        for p, tl, cl in zip(misclf_proteins, true_labels, pred_labels):
                            if tl != cl:                            
                                PF_misclf.loc[p,cl] += 1

                        CM = confusion_matrix(true_labels, pred_labels,
                                              normalize='true',labels=[fam1, fam2])
# Store the average confusion matrix values (4 for comparing 2 families) in CMDF
# AA = True positive, AB = False negative, BA = False positive, BB = True negative
                        aa,ab,ba,bb = CM.flatten()
                        CMDF = CMDF.append({'familyA':fam1, 'familyB':fam2, 'AA':aa,
                                            'AB':ab, 'BA':ba, 'BB':bb},ignore_index=True)
                
                Mclass_score = np.mean(CMDF['AB'].values+CMDF['BA'].values)
                heatmapname = 'KNN_k'+str(k)+'_betweenfamily_misclassified.png'
                clust_method = 'KNN_k'+str(k)
                heatmap_title = 'KNN pairwise misclassification'
                # CMDF.to_pickle("./ConfusionMatrix.pkl",protocol=3)
            
#   ---------------------------- PROVIDED DISTANCE MATRIX  ---------------------------------            
# If a precomputed distance matrix is passed to the function
            else:
#  Read in provided distance matrix 
                if DM[-4:] == '.npy':
                    distancematrix = np.load(DM)
                elif DM[-4:] == '.pkl':
                    distancematrix = pd.read_pickle(DM).values
                distancematrix_name = str(DM)[:-4]
                
                
# Create dataframe containing a copy of the selected proteins only           
                selected_prots = self.proteins[selector].copy()
# Column that reassigns label for new dataframe (can transfer between old and new labels)
                selected_prots['dist_labels'] = np.arange(len(selected_prots))

# Compare all families with each other
                for i,fam1 in enumerate(families):
                    for j,fam2 in enumerate(families[i+1:]): 
                        selected_rows = (selected_prots['family_name']==fam1) | (
                            selected_prots['family_name']==fam2)
                        idx = selected_prots.loc[selected_rows,'dist_labels'].values
                        vals = distancematrix[np.ix_(idx, idx)]
#  Set the diagonal of the distance matrix to a large value to
# avoid it being its own nearest point
                        vals[np.diag_indices_from(vals)] = 1e18

                        nr = selected_prots[selected_rows]['family_name'].map(
                            lambda x: fam_to_nr[x]).values
# Use precomputed metric if you provide a distance matrix
                        neigh = kNN(n_neighbors=k, metric = 'precomputed')
                        neigh.fit(vals, nr)
                        y_pred_nr = neigh.predict(vals)
                        y_pred_lab = list(map(lambda x: nr_to_fam[x], y_pred_nr))
                        y_true_nr = nr
                        y_true_lab = selected_prots[selected_rows]['family_name'].values
# ?????????????????? Not sure we can pass this y_true_lab and y_pred_lab - used to be a vector
                        misclf_proteins = selected_prots[selected_rows]['protein_name'].values
                        for p, tl, cl in zip(misclf_proteins, y_true_lab, y_pred_lab):
                            if tl != cl:                            
                                PF_misclf.loc[p,cl] += 1

                        CM = confusion_matrix(y_true_lab, y_pred_lab,
                                          normalize='true',labels=[fam1, fam2])
# Store the average confusion matrix values (4 for comparing 2 families) in CMDF
                        aa,ab,ba,bb = CM.flatten()
                        CMDF = CMDF.append({'familyA':fam1, 'familyB':fam2, 'AA':aa, 
                                            'AB':ab, 'BA':ba, 'BB':bb},ignore_index=True)
                Mclass_score = np.mean(CMDF['AB'].values+CMDF['BA'].values)
                heatmapname = 'KNN_k'+str(k)+'_Distancematrix_'+str(DM)\
                +'_betweenfamily_misclassified.png'
                clust_method = 'KNN_k'+str(k)
                heatmap_title = 'KNN pairwise misclassification with distance matrix '\
                +str(DM)
            
#   ---------------------------- PLOT HEATMAP  ---------------------------------
        cm_hm = pd.concat([
            CMDF[['familyA','familyB','AB']].rename(columns={
                'familyA': 'Family A',
                'familyB': 'Family B',
                'AB': 'Misclassified'
            }),
            CMDF[['familyA','familyB','BA']].rename(columns={
                'familyA': 'Family B',
                'familyB': 'Family A',
                'BA': 'Misclassified'
            })
        ])


        sns.set_context("paper")  
        fig,ax = plt.subplots(1,1,figsize = [20,20])
        sns.heatmap(cm_hm.pivot_table(index='Family A', columns='Family B', 
                    values='Misclassified'), cbar_kws={'label': clust_method+
                    ' misclassified','shrink': 0.85},cmap='RdPu',ax=ax, square=True)
        ax.set_title(heatmap_title)
        fig.savefig('results/'+str(self.embedding)+'/Classification/'+heatmapname,
                    bbox_inches='tight',dpi=300)

        Misclf_proteins = PF_misclf[(PF_misclf>0).any(axis=1)].index
        
        return CMDF, Mclass_score, PF_misclf, Misclf_proteins

    
    

    
                                                        
                                      
                                      
############################ plot tSNE for all families ##################################  
    def plot_tSNE(self, familyselection='all', kmeans=False):
#   ------------------------------ INPUT ------------------------------
# 1.familyselection: list of at least 2 family names e.g. ['PF08213','PF02237',...]
# - If not provided it automatically selects all families 
# - Otherwise it choses the proteins from the list of family names provided in familyselection
# 2. kmeans: True or False (default)
# - False (default): Only plots t-SNE clustering
# - True: Plots Kmeans clustering and t-SNE clustering in one image to compare and provides a
#         misclassification dataframe misclf_prot_fam (for all protein vs. all families) and
#         returns the list of misclassified proteins in Misclf_proteins

# Familyselector is automatically set to all to include all families 
# You can also specifiy a list of families by seeting familyselector 
        if familyselection=='all':
            famlist = self.families
# If familyselection is a list then only select proteins from that list in variable selector            
        if isinstance(familyselection, list):
            famlist = familyselection
# Plot only provided families in familyselection      
        selector = self.proteins['family_name'].isin(famlist)    
        

        if kmeans and len(famlist)<2:
            raise ValueError('You need to provide more than 1 family in families to this function if kmeans=True')


# Define a colourpalette with lightness depending on the amount of familes
# ncol tells us how many columns the labels will require in the plot
        if len(famlist)<=20:
            colours = sns.husl_palette(len(famlist), l=0.65)
            ncol=1
        elif len(famlist)<=40:
            colours = sns.husl_palette(int(np.ceil(len(famlist)/2)), l=0.65)+sns.husl_palette(int(np.ceil(len(famlist)/2)), l=0.4)
            ncol=2
        elif len(famlist)<=60:
            colours = sns.husl_palette(int(np.ceil(len(famlist)/3)), l=0.8)+sns.husl_palette(
            int(np.ceil(len(famlist)/3)), l=0.65)+sns.husl_palette(int(np.ceil(len(famlist)/3)), l=0.4)
            ncol=3
        else:
            colours = sns.husl_palette(int(np.ceil(len(famlist)/4)), l=0.8)+sns.husl_palette(
            int(np.ceil(len(famlist)/4)),l=0.6)+ sns.husl_palette(int(np.ceil(len(famlist)/4)), l=0.4
            )+sns.husl_palette(int(np.ceil(len(famlist)/4)), l=0.2)
            ncol=4
                                   

# Perform T-SNE clustering to 2 dimensions using all selected embedded vectors    
        X_embedded = TSNE(n_components=2,random_state=self.rng.integers(0,2147483647)).fit_transform(np.vstack(
        self.proteins[selector]['protein_vector'].values))


#   ---------------------------- PERFORM KMEANS COMPARISON  ---------------------------------
# If kmeans=True: compare Kmeans clustering with T-SNE clustering                                               
        if kmeans:
# Perform Kmeans clustering to len(famlist) clusters
            clust = KMeans(n_clusters=len(famlist), random_state=self.rng.integers(0,2147483647)).fit(
                np.vstack(self.proteins[selector]['protein_vector'].values))

# fam2_misclf stores which proteins get misclassified into which families
            misclf_prot_fam = pd.DataFrame(0, index=self.proteins[selector
            ]['protein_name'].unique(),columns=self.proteins[selector]['family_name'].unique())
          
# Identify which family name is which cluster label via linear sum assignment
# First define a benefit function M and then maximize it
            clusterlabels = np.unique(clust.labels_)
            M = np.zeros([len(famlist), len(clusterlabels)])
            for i,fam in enumerate(famlist):
                for j,lab in enumerate(clusterlabels):
                    M[i,j] = sum((self.proteins[selector]['family_name'].values==fam)
                                        & (clust.labels_==lab))

        
# Identify which clusters belong to which families by maximising befit_fct M
            ii, jj = linear_sum_assignment(M, maximize=True)
# Create 2 dictionaries that map family names to cluster labels and vice versa
            family_dict = {}
            label_dict = {}
            for i, j in zip(ii,jj):
                family_dict[famlist[i]] = clusterlabels[j]
                label_dict[clusterlabels[j]] = famlist[i]

# Calculate the confusion matrix by passing it the true labels and the matching kmeans labels
            Truelabels = self.proteins[selector]['family_name'].values

            prots = self.proteins[selector]['protein_name'].values

            for p, tl, cl in zip(prots, Truelabels, [label_dict[l] for l in clust.labels_]):
                if tl != cl:                            
                    misclf_prot_fam.loc[p,cl] += 1
            self.misclf_prot_fam = misclf_prot_fam

            Misclf_proteins = misclf_prot_fam[(misclf_prot_fam>0).any(axis=1)].index

#   ---------------------------- PLOT T-SNE AND KMEANS CLUSTERINGS  -------------------------
            sns.set_context('talk')
            fig, (ax1,ax2) = plt.subplots(2, 1, figsize=[6,17])
            sns.scatterplot(data=pd.DataFrame(data = {
                't-SNE reduced dimension 1': X_embedded[:,0],
                't-SNE reduced dimension 2': X_embedded[:,1],
                'Kmeans': ['Cluster {:d}'.format(l) for l in clust.labels_]
            }), x='t-SNE reduced dimension 1', y='t-SNE reduced dimension 2', 
                            hue='Kmeans', 
                            hue_order=['Cluster {:d}'.format(l) for l in np.unique(
                            clust.labels_)], ax=ax1)
            ax1.legend(bbox_to_anchor=(1.05, 0.5), loc='center left',ncol=ncol)

            sns.scatterplot(data=pd.DataFrame(data = {
                            't-SNE reduced dimension 1': X_embedded[:,0],
                            't-SNE reduced dimension 2': X_embedded[:,1],
                            'Family': self.proteins[selector]['family_name']
                            }), x='t-SNE reduced dimension 1', y='t-SNE reduced dimension 2', 
                            hue='Family', ax=ax2)
            ax2.legend(bbox_to_anchor=(1.05, 0.5), loc='center left',ncol=ncol)
            ax1.set_title('Kmeans:')
            ax2.set_title('T-SNE:')
            if len(famlist)==2:
                fig.savefig('results/'+str(self.embedding)+
                '/Classification/T-SNE_'+famlist[0]+'_'+famlist[1]+'.png', bbox_inches='tight',dpi=300)
            else:  
                fig.savefig('results/'+str(self.embedding)+
                        '/ClassificationT-SNE_Kmeans_compare.png', bbox_inches='tight')

            return misclf_prot_fam, Misclf_proteins
                                        
#   ---------------------------- PLOT ONLY T-SNE CLUSTERINGS  ------------------------------
        else:                  
            sns.set_context('talk')
            fig, ax = plt.subplots(1, 1, figsize=[8,8])
            sns.scatterplot(data=pd.DataFrame(data = {
                't-SNE reduced dimension 1': X_embedded[:,0],
                't-SNE reduced dimension 2': X_embedded[:,1],
                'Families': self.proteins[selector]['family_name']
            }), x='t-SNE reduced dimension 1', y='t-SNE reduced dimension 2', 
                            hue='Families', 
                            ax=ax, palette=colours[:len(famlist)], s=25, linewidth=0)
            ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', ncol=ncol)
            if len(famlist)<2:
                ax.set_title(famlist)
                fig.savefig('results/'+str(self.embedding)+
                '/Classification/T-SNE_'+famlist[0]+'.png', bbox_inches='tight',dpi=300)
            else:
                fig.savefig('results/'+str(self.embedding)+
                '/Classification/T-SNE_allfamilies.png', bbox_inches='tight',dpi=300)

            

        
        
   
        
        
                  
                                                        
                                                        
                                                        
                                                        
############################ FAMILY ANALYSIS PROCESSES & OUTPUTS  ##################################    
    def pairwise_distances(self,familyselection='all',emb_metric='Euclidean', seq_metric='PA', norm=True, hist=False):
#   ------------------------------ INPUT ------------------------------
# 1. familyselection: list of at least 2 family names e.g. ['PF08213','PF02237',...]
# - If not provided it automatically selects all families 
# - Otherwise it choses the proteins from the list of family names provided in familyselection
# 2. emb_metric:   'Euclidean' or 'Cosine'
# - default is 'Euclidean'
# - Defines the distance metric used to calculate distances in the embedding space
# 3. seq_metric: 'DL' for Damerau-Levenshtein or 'PA' for Pairwise Alignment distance
#  - Default is 'PA'
#  - Defines the distance metric used to calculate distances in the sequence space
# 4. norm: 'True' or 'False'
#  - Default is True
#  - If True it norms the embedding distance matrix and the sequence distance matrix by the average sequence length
#  - Exception: If seq_metric is 'PA' the distance is already normed and no further actions are taken
# 5. hist: 'True' or 'False'
#  - Default is 'False'
#  - If True it saves the histograms of the distribution of embedding distances between 2 families

#   ------------------------------ OUTPUT ------------------------------
# 1. clusters: Dataframe with columns = ['family1', 'family2', 'Edis_mean','Edis_std', 
#             'Sdis_mean','Sdis_std','centroid','centroid_distances','CD_mean','CD_std'])
#  - Name of family 1 and 2, mean and std of the embedding distances between the proteins of 2 families 
#    or within 1 family (if family1=famiy2) mean and std of the sequence distances between the proteins 
#    of 2 families or within 1 family (if family1=famiy2), centroid (between 2 families or within 1 family)
#    coordiantes in the embedding space, distances of each protein to the centroid,
#    mean and std of the centroid distances
# 2. Seq_dis: Distance matrix of all proteins of each selected family to each the other in sequence space
# 2. Emb_dis: Distance matrix of all proteins of each selected family to each the other in embedding space





# Familyselector is automatically set to all to include all families 
# You can also specifiy a list of families by seeting familyselector 
        if familyselection=='all':
            famlist = self.families
# If familyselection is a list then only select proteins from that list in variable selector            
        if isinstance(familyselection, list):
            famlist = familyselection
            
# Raise error if fewer than 2 families are passed            
        if len(famlist)<2:
            raise ValueError('You need to provide more than 1 family in familyselection to this function')

# Plot only provided families in famlist      
        selector = self.proteins['family_name'].isin(famlist)    
        
# sort list of families by name   
        famlist = sorted(famlist)
# Calculate average sequence length to norm distances if norm is True
        if norm:
            slengths = self.proteins[selector]['seq_length'].values
            SN = np.zeros([len(slengths),len(slengths)])
            for i,s1 in enumerate(slengths):
                for j,s2 in enumerate(slengths):
                    if j<=i:
                        SN[i,j] = (s1+s2)/2
                        SN[j,i] = SN[i,j]

#   ---------------------------- SEQUENCE DISTANCES ------------------------------                        
# Calculate sequence distances (e.g.'DL' or 'PA')                    
        sequences = self.proteins[selector]['sequence'].values
        Seq_dis = np.zeros([len(sequences),len(sequences)])
        if seq_metric == 'DL':
            for i,seq in enumerate(sequences):
# Calculate the Damerau-Levenshtein distance between all sequences (takes a long time)
                Seq_dis[i,i:]=DLD(seq,sequences[i:])
                Seq_dis[i:,i]=Seq_dis[i,i:] 

                        
# Calculate the Pairwise Alignment distance between all sequences                      
        elif seq_metric == 'PA':
            aligner = Align.PairwiseAligner()
            aligner.mode = 'global' 
            matrix = substitution_matrices.load("BLOSUM62")
            aligner.substitution_matrix = matrix
            aligner.open_gap_score = -11
            aligner.extend_gap_score = -1
# Calculate best alignment score of 2 sequences using biopython
            for i,seq1 in enumerate(sequences):
                for j,seq2 in enumerate(sequences):
# Calculate lower triangle of matrix and mirror it up
                    if j<=i:
                        Seq_dis[i,j]=aligner.score(seq1,seq2)
                        Seq_dis[j,i]=Seq_dis[i,j]
# Transform pairwise alignment scores into distances
            AA, BB = np.meshgrid(Seq_dis.diagonal(), Seq_dis.diagonal())
            Seq_dis = 1-2*Seq_dis/(AA+BB)

#   ---------------------------- EMBEDDING DISTANCES ------------------------------   
        proteinvectors = self.proteins[selector]['protein_vector'].values
        Emb_dis = np.zeros([len(proteinvectors),len(proteinvectors)])

        for i,vec1 in enumerate(proteinvectors):
            for j,vec2 in enumerate(proteinvectors):
# Calculate lower triangle of matrix and mirror it up
                if j<=i:
                    if emb_metric == 'Euclidean':
                        Emb_dis[i,j] = np.linalg.norm(vec1-vec2)
                        Emb_dis[j,i] = Emb_dis[i,j]
                    elif emb_metric == 'Cosine':
                        Emb_dis[i,j] = distance.cosine(vec1,vec2)
                        Emb_dis[j,i] = Emb_dis[i,j]
                    
# If norm is true we normalize the embedding distances by average sequence length SN                  
        if norm:
            Emb_dis = Emb_dis/SN
            if seq_metric == 'DL':
                Seq_dis = Seq_dis/SN

# Create dataframe containing a copy of the selected proteins only           
        selected_prots = self.proteins[selector].copy()
# Column that reassigns label for new dataframe (can transfer between old and new labels)
        selected_prots['dist_labels'] = np.arange(len(selected_prots))

        clusters = pd.DataFrame(columns = ['family1', 'family2', 'Edis_mean','Edis_std', 'Sdis_mean','Sdis_std','centroid','centroid_distances','CD_mean','CD_std'])

        for i,family1 in enumerate(famlist):
            for family2 in famlist[i:]:
#   ---------------------------- WITHIN FAMILY DISTANCES ------------------------------    
                if family1 == family2:
                    idx = selected_prots[selected_prots['family_name']==family1]['dist_labels'].values
                    idy = selected_prots[selected_prots['family_name']==family2]['dist_labels'].values
                    Fam_Emb = Emb_dis[np.ix_(idx,idy)]
                    Fam_Seq = Seq_dis[np.ix_(idx,idy)]
                    Fam_Emb[np.diag_indices_from(Fam_Emb)] = np.nan
                    Fam_Seq[np.diag_indices_from(Fam_Seq)] = np.nan

                    Edis_mean = np.nanmean(Fam_Emb.flatten())
                    Edis_std = np.nanstd(Fam_Emb.flatten()) 
                    Sdis_mean = np.nanmean(Fam_Seq.flatten())
                    Sdis_std = np.nanstd(Fam_Seq.flatten())
                    
                    proteinvectors = selected_prots[(selected_prots['family_name']==family1) | 
                    (selected_prots['family_name']==family2)]['protein_vector'].values
                    centroid = np.vstack(proteinvectors).mean(axis=1) 
                    centroid_distances = proteinvectors-centroid
                    means_centroid_distances = np.mean(centroid_distances)
                    std_centroid_distances = np.std(centroid_distances)                    

#   ---------------------------- BETWEEN FAMILY DISTANCES ------------------------------ 
                
                else:
                    idx = selected_prots[selected_prots['family_name']==family1]['dist_labels'].values
                    idy = selected_prots[selected_prots['family_name']==family2]['dist_labels'].values
                    Fam_Emb = Emb_dis[np.ix_(idx,idy)]
                    Fam_Seq = Seq_dis[np.ix_(idx,idy)]

                    Edis_mean = np.nanmean(Fam_Emb.flatten())
                    Edis_std = np.nanstd(Fam_Emb.flatten())
                    
                    Sdis_mean = np.nanmean(Fam_Seq.flatten())
                    Sdis_std = np.nanstd(Fam_Seq.flatten())
                    
                    proteinvectors = selected_prots[(selected_prots['family_name']==family1) | 
                    (selected_prots['family_name']==family2)]['protein_vector'].values
                    centroid = np.vstack(proteinvectors).mean(axis=1) 
                    centroid_distances = proteinvectors-centroid
                    means_centroid_distances = np.mean(centroid_distances)
                    std_centroid_distances = np.std(centroid_distances)  
                    
                    if hist:
                        # How is the euclidean distance distributed? 
                        tmp = Emb_dis[np.triu_indices(len(slengths),1)].flatten()
                        plt.hist(tmp[tmp>0],bins=100)
                        plt.xlabel('Embedding distances')
                        plt.ylabel('Counts')
                        plt.title('Distribution of euclidean distances '+str(family1)+'_'+str(family2))
                        plt.savefig('results/Classification/hist_Neucl_families_'+str(family1)+'_'+str(family2)+'.png', bbox_inches='tight',dpi=300)
                                
                                
                                
                clusters = clusters.append({'family1':family1, 'family2':family2, 
                'Edis_mean':Edis_mean, 'Edis_std':Edis_std, 'Sdis_mean':Sdis_mean,
                'Sdis_std':Sdis_std,'centroid':centroid,'centroid_distances':centroid_distances,
                'CD_mean':means_centroid_distances,'CD_std':std_centroid_distances},ignore_index=True)
                



        return clusters, Seq_dis, Emb_dis

#################################### PROTVEC AND DPROTVEC CLASSES AND FUNCTIONS ####################################################         
# Borrowed very heavily from
# https://github.com/jowoojun/biovec
# https://github.com/kyu999/biovec
# But with some important modifications
# Most importantly, normalisation on the sum of three vectors is implemented


#   ---------------------------- PROTVEC ------------------------------ 

class ProtVec(word2vec.Word2Vec):

    def __init__(self, fasta_fname=None, corpus=None, n=3, size=100, corpus_fname="corpus.txt",  sg=1, window=25, min_count=1, workers=20):
        """
        Either fname or corpus is required.
        fasta_fname: fasta file for corpus
        corpus: corpus object implemented by gensim
        n: n of n-gram
        corpus_fname: corpus file path
        min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
        """

        self.n = n
        self.size = size
        self.fasta_fname = fasta_fname

        if corpus is None and fasta_fname is None:
            raise Exception("Either fasta_fname or corpus is needed!")

        if fasta_fname is not None:
            print('Generate Corpus file from fasta file...')
            generate_corpusfile(fasta_fname, n, corpus_fname)
            corpus = word2vec.Text8Corpus(corpus_fname)

        word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, workers=workers)

    def to_vecs(self, seq):
        """
        convert sequence to three n-length vectors
        e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
        """
        ngram_patterns = split_ngrams(seq, self.n)

        protvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self[ngram])
                except:
                    raise Exception("Model has never trained this n-gram: " + ngram)
            protvecs.append(sum(ngram_vecs))
        return protvecs
    
    def get_vector(self, seq):
        """
        sum and normalize the three n-length vectors returned by self.to_vecs
        """
#         return normalize(sum(self.to_vecs(seq)))
        return sum(self.to_vecs(seq))
    
    
    
#  Load doc2vec or prot2vec - depending on the modelname
def load_protvec(model_fname):
    if 'doc2vec' in model_fname:
        return DProtVec.load(model_fname)
    else:
        return ProtVec.load(model_fname)                                                
                                             

def split_ngrams(seq, n):
    """
    'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    """
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams

def generate_corpusfile(fasta_fname, n, corpus_fname):
    '''
    Args:
        fasta_fname: corpus file name
        n: the number of chunks to split. In other words, "n" for "n-gram"
        corpus_fname: corpus_fnameput corpus file path
    Description:
        Protvec uses word2vec inside, and it requires to load corpus file
        to generate corpus.
    '''
    f = open(corpus_fname, "w")
    fasta = Fasta(fasta_fname)
    for record_id in tqdm(fasta.keys(), desc='corpus generation progress'):
        r = fasta[record_id]
        seq = str(r)
        ngram_patterns = split_ngrams(seq, n)
        for ngram_pattern in ngram_patterns:
            f.write(" ".join(ngram_pattern) + "\n")
    f.close()
    
'''
Binary representation of amino acid residue and amino acid sequence
e.g.
    'A' => [0, 0, 0, 0, 0]
    'AGGP' => [[0, 0, 0, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 1]]
'''

AMINO_ACID_BINARY_TABLE = {
    'A': [0, 0, 0, 0, 0],
    'C': [0, 0, 0, 0, 1],
    'D': [0, 0, 0, 1, 0],
    'E': [0, 0, 0, 1, 1],
    'F': [0, 0, 1, 0, 0],
    'G': [0, 0, 1, 0, 1],
    'H': [0, 0, 1, 1, 0],
    'I': [0, 0, 1, 1, 1],
    'K': [0, 1, 0, 0, 0],
    'L': [0, 1, 0, 0, 1],
    'M': [0, 1, 0, 1, 0],
    'N': [0, 1, 0, 1, 1],
    'P': [0, 1, 1, 0, 0],
    'Q': [0, 1, 1, 0, 1],
    'R': [0, 1, 1, 1, 1],
    'S': [1, 0, 0, 0, 0],
    'T': [1, 0, 0, 0, 1],
    'V': [1, 0, 0, 1, 0],
    'W': [1, 0, 0, 1, 1],
    'Y': [1, 0, 1, 0, 0]
}

def convert_amino_to_binary(amino):
    '''
    Convert amino acid to 1-dimentional 5 length binary array
    "A" => [0, 0, 0, 0, 0]
    '''
    if not AMINO_ACID_BINARY_TABLE.has_key(amino):
        return None
    return AMINO_ACID_BINARY_TABLE[amino]


def convert_amino_acid_sequence_to_vector(sequence):
    '''
    "AGGP" => [[0, 0, 0, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 1]]
    '''
    binary_vector = [convert_amino_to_binary(amino) for amino in sequence]
    if None in binary_vector:
        return None
    return binary_vector

def normalize(x):
    return x / np.sqrt(np.dot(x, x))



def _combine(vectors, k):
    """ Combine vectors. """
    embeds = np.zeros((vectors.shape[0] // k, vectors.shape[1]))
    for i in range(k):
        embeds += vectors[i::k, :]
    return embeds
def _normalize(vectors):
    """ Normalize vectors (in rows) to length 1. """
    norms = np.sqrt(np.sum(vectors ** 2, axis=1))
    vectors /= norms.reshape((len(norms), 1))
    return vectors
    
#   ---------------------------- DPROTVEC ------------------------------     
    
    
    
class DProtVec(doc2vec.Doc2Vec):
    def __init__(self, corpus=None, size=200, epochs=25, dm=1, window=25, min_count=0, workers=20):
        doc2vec.Doc2Vec.__init__(self, corpus, vector_size=size, epochs=epochs, dm=dm, window=window, 
                         min_count=min_count, workers=workers)
    def get_embeddings(self, seqs, k=3, overlap=False, norm=False, epochs=50):
        """ Infer embeddings in one pass using a gensim doc2vec model.
        Parameters:
            doc2vec_file (str): file pointing to saved doc2vec model
            seqs (iterable): sequences to infer
            k (int) default 3
            overlap (Boolean) default False
            norm (Boolean) default True
            steps (int): number of steps during inference. Default 5.
        Returns:
            numpy ndarray where each row is the embedding for one sequence.
        """
        #model = Doc2Vec.load(doc2vec_file)
        as_kmers = seqs_to_kmers(seqs, k=k, overlap=overlap)
        vectors = np.array([self.infer_vector(doc, epochs=epochs)
                            for doc in as_kmers])
        if overlap:
            embeds = vectors
        else:
            embeds = _combine(vectors, k)
        if norm:
            embeds = _normalize(embeds)
        return embeds
    def get_embeddings_new(doc2vec_file, seqs, k=3, overlap=False, passes=100):
        """ Infer embeddings by averaging passes using a gensim doc2vec model.
        Make passes through the sequences, normalizing and averaging the results
        after every pass.
        Parameters:
            doc2vec_file (str): file pointing to saved doc2vec model
            seqs (iterable): sequences to infer
            k (int) default 3
            overlap (Boolean) default False
            passes (int): number of passes during inference. Default 100.
        Returns:
            numpy ndarray where each row is the embedding for one sequence.
        """
        #model = Doc2Vec.load(doc2vec_file)
        as_kmers = seqs_to_kmers(seqs, k=k, overlap=overlap)
        old_embeds = None
        order = [i for i in range(len(seqs))]
        for p in range(passes):
            random.shuffle(order)
            if not overlap and k > 1:
                shuffled_inds = list(chain.from_iterable(([k*i + kk for
                                                           kk in range(k)]
                                     for i in order)))
            else:
                shuffled_inds = order
            shuffled_kmers = [as_kmers[i] for i in shuffled_inds]
            vectors = np.array([self.infer_vector(doc, steps=1)
                                for doc in shuffled_kmers])
            if overlap:
                embeds = vectors
            else:
                embeds = _combine(vectors, k)
            embeds = _normalize(embeds)
            unshuffle_order = [order.index(i) for i in range(len(order))]
            embeds = embeds[unshuffle_order]
            if old_embeds is not None:
                old_embeds = (p * old_embeds + embeds) / (p + 1)
                old_embeds = _normalize(old_embeds)
            else:
                old_embeds = embeds[:]
        return old_embeds
    def get_vector(self, seq):
        return self.get_embeddings([seq])[0]
    def get_vector_norm(self, seq):
        return self.get_embeddings([seq], norm=True)[0]
        
                         
def seq_to_kmers(seq, k=3, overlap=False, **kwargs):
    """ Divide a string into a list of kmer strings.
    Parameters:
        seq (string)
        k (int), default 3
        overlap (Boolean), default False
    Returns:
        List containing 1 list of kmers (overlap=True) or k lists of
            kmers (overlap=False)
    """
    N = len(seq)
    if overlap:
        return [[seq[i:i+k] for i in range(N - k + 1)]]
    else:
        return [[seq[i:i+k] for i in range(j, N - k + 1, k)]
                for j in range(k)]
def seqs_to_kmers(seqs, k=3, overlap=False, **kwargs):
    """Divide a list of sequences into kmers.
    Parameters:
        seqs (iterable) containing strings
        k (int), default 3
        overlap (Boolean), default False
    Returns:
        List of lists of kmers
    """
    as_kmers = []
    for seq in seqs:
        as_kmers += seq_to_kmers(seq, k=k, overlap=overlap)
    return as_kmers
    
class Corpus(object):
    """ An iteratable for training seq2vec models. """
    def __init__(self, fasta_fname, kmer_hypers):
        self.fasta = Fasta(fasta_fname)
        self.kmer_hypers = kmer_hypers
    def __iter__(self):
        for doc in self.get_documents():
            yield doc
    def df_to_kmers(self):
        for record_id in self.fasta.keys():
            r = self.fasta[record_id]
            seq = str(r)
            kmers = seq_to_kmers(seq, **self.kmer_hypers)
            if self.kmer_hypers['overlap']:
                yield kmers
            else:
                for km in kmers:
                    yield km
    def get_documents(self):
        if self.kmer_hypers['merge']:
            return (TaggedDocument(doc, [i // self.kmer_hypers['k']])
                    for i, doc in enumerate(self.df_to_kmers()))
        return (TaggedDocument(doc, [i]) for i,
                doc in enumerate(self.df_to_kmers()))    

