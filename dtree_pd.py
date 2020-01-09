# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:51:42 2019
Categorical Decision Tree

@author: lfinco
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy

#test data
data1 = pd.read_csv("cdata1.csv")
features = ['Age', 'Education', 'Income', 'MaritalStatus']
# set the target column
clas = 'Purchase'

#data2 = pd.read_csv("cdata2.csv")
#features = ['a', 'b', 'c', 'd', 'e']
#clas = 'cls'


#data = data1[data1['Purchase']=="will buy"]
#data = data1[0:3]

# Categorical Only

#entropy(data2[clas].value_counts())
#data2[clas].value_counts()
#feature = "Income"

def bestsplit(data, clas):
    # holder lists for split tests
    feat, val, gn, splittype = [], [], [], []
    
    # use ID3 Iterative Dichotomiser
    # get best split
    
    # tests
    if len(data)<5:
        return('', '', '', '', '', '')
    
    # calculate entropy of data before split
    entr = entropy(data[clas].value_counts())
    
    # grab the features
    features = data.columns.tolist()
    features.remove(clas)
    
    for feature in features:
        if np.issubdtype(data[feature].dtype, np.object_) or np.issubdtype(data[feature].dtype, np.bool_):
            
            # search through feature vlaues for best info gain
            for value in data[feature].unique().tolist():
                
                # categorical split on feature=value
                left = data[data[feature]==value]
                right = data[data[feature]!=value]
                
                # calculate new entropies
                entr_l = entropy(left[clas].value_counts())
                entr_r = entropy(right[clas].value_counts())
                
                # calculate info gain
                weight = len(data[data[feature]==value]) / len(data)
                entr_split = (weight*entr_l) + ((1-weight)*entr_r)
                gain = entr - entr_split
                
                # save results
                splittype.append("categorical")
                feat.append(feature)
                val.append(value)
                gn.append(gain)
                
        elif np.issubdtype(data[feature].dtype, np.integer) or np.issubdtype(data[feature].dtype, np.float64):
            
            splittype.append("continuous")
            
            # use binary split, average of max & min
            value = data[feature].mean()
            
            # continuous split on feature < value
            left = data[data[feature]<value]
            right = data[data[feature]>=value]
            
            # calculate new entropies
            entr_l = entropy(left[clas].value_counts())
            entr_r = entropy(right[clas].value_counts())
            
            # calculate info gain
            weight = len(data[data[feature]<value]) / len(data)
            entr_split = (weight*entr_l) + ((1-weight)*entr_r)
            gain = entr - entr_split
            
            # save results
            feat.append(feature)
            val.append(value)
            gn.append(gain)
            
        else:
            print("Dtype not supported", data[feature].dtype)
            return('', '', '', '', '', '')
    
    # return split for best gain
    if np.var(gn) != 0:
        # if info gain, return the split criteria and the split datasets
        idx = np.argmax(gn)
        if np.issubdtype(data[feat[idx]].dtype, np.object_) or np.issubdtype(data[feat[idx]].dtype, np.bool_):
            left = data[data[feat[idx]]==val[idx]]
            right = data[data[feat[idx]]!=val[idx]]
        elif np.issubdtype(data[feat[idx]].dtype, np.integer):
            left = data[data[feat[idx]]<val[idx]]
            right = data[data[feat[idx]]>=val[idx]]
        
        if len(left) == 0 and len(right) == 0:
            return('', '', '', '', '', '')
        else:
            print(feat[idx], val[idx], gn[idx], len(left), len(right), splittype[idx])
            return(feat[idx], val[idx], gn[idx], left, right, splittype[idx])
    else:
        # if no info gain, return empty strings
        return('', '', '', '', '', '')


class Node:
    """
    Represents a split in the tree.
    """
    def __init__(self, feature='', value='', gain='', count=0, minsamples=5):
        
        # feature for the split
        self.feat = feature
        
        # value for the split
        self.val = value
        
        # information gain for the split
        self.gn = gain
        
        # is this node a leaf node?
        self.leaf_node = True
        
        # predicted most probable category for node
        self.leaf_category = ''
        
        # samples in node before split
        self.samples = count
        
        # minimum samples needed to split a node
        self.minsamples = minsamples
        
        # type of split, either "categorical" or "continuous"
        self.node_type = None
        
        # each of the two subnodes - assume a leaf node before testing using bestsplit
        self.sub_node_l = None
        self.sub_node_r = None
    
    def train(self, data, clas):
        
        # train this node on data submitted
        splits = 0  # tracker for number of splits created
        self.samples = len(data)
        
        # test for minimum number of samples
        if self.samples == 0:
            
            # if less than minimum number of samples, make it a leaf node
            self.leaf_node = True
            self.samples = len(data)
            self.leaf_category = ''
            self.feat, self.val, self.gn, self.sub_node_l, self.sub_node_r = ('', '', '', '', '')
            print("exit", self.leaf_category, self.feat)
            return
        
        if self.samples < self.minsamples:
            
            # if less than minimum number of samples, make it a leaf node
            self.leaf_node = True
            self.samples = len(data)
            self.leaf_category = ''
            self.feat, self.val, self.gn, self.sub_node_l, self.sub_node_r = ('', '', '', '', '')
            print("exit", self.leaf_category, self.feat)
            return
        
        # get best split
        split_result = bestsplit(data, clas)
        
        if split_result[0] != '':
            
            # if there is a good split result, save the feature and the value
            # and kick off 2 sub-nodes
            splits = splits + 1
            self.leaf_node = False
            self.leaf_category = split_result[5]
            self.feat = split_result[0]
            self.val = split_result[1]
            self.gn = split_result[2]
            split1 = split_result[3]
            split2 = split_result[4]
            self.node_type = split_result[5]
            
            # call each subnode to train on their split
            self.sub_node_l = Node()
            self.sub_node_r = Node()
            self.sub_node_l.train(split1, clas)
            self.sub_node_r.train(split2, clas)
        else:
            # if no best split, make it a leaf node
            #print("not splitting")
            self.leaf_node = True
            self.leaf_category = data[clas].value_counts().index[0]
        #print(splits)
    
    def pred(self, data):
        """
        Return predicted category from leaf recursive
        data is now one row :-(
        """
        if self.leaf_node:
            return(self.leaf_category)
        else:
            if self.node_type == "categorical":
                if data[self.feat] == self.val:
                    return self.sub_node_l.pred(data)
                else:
                    return self.sub_node_r.pred(data)
            elif self.node_type == "continuous":
                if data[self.feat] < self.val:
                    return self.sub_node_l.pred(data)
                else:
                    return self.sub_node_r.pred(data)
            else:
                print("Not Categorical or Continuous")
                return
    
    def draw(self, g, label):
        if self.leaf_node:
            return g
        else:
            l_label = label+"A"
            r_label = label+"B"
            g.node(l_label, self.sub_node_l.feat+" - "+self.sub_node_l.val+"\nSamples:"+str(self.sub_node_l.samples))
            g.node(r_label, self.sub_node_r.feat+" - "+self.sub_node_r.val+"\nSamples:"+str(self.sub_node_r.samples))
            g.edge(label, l_label)
            g.edge(label, r_label)
            self.sub_node_l.draw(g, l_label)
            self.sub_node_r.draw(g, r_label)
            return g


class Tree:
    """
    Represents one decision tree of recursive nodes
    """
    def __init__(self, feature='', value=''):
        self.head_node = Node()
    
    def fit(self, data, clas):
        """
        Train the decision tree on data provided
        Parameters:
        
        data := Dataset to train the decision tree on.  Type must be pandas Dataframe.
        clas := 
        """
        self.head_node.train(data, clas)
    
    def graph(self, g):
        head_label = 'A'
        #g = Digraph('Decision Tree', filename='dtree.gv')
        if self.head_node.leaf_node:
            g.node(head_label, self.head_node.category+" "+str(self.head_node.samples))
            return g
        else:
            l_label = head_label+"A"
            r_label = head_label+"B"
            g.node(head_label, self.head_node.feat+" "+self.head_node.val+" "+str(self.head_node.samples))
            g.node(l_label, self.head_node.sub_node_l.feat+" "+self.head_node.sub_node_l.val+" "+str(self.head_node.sub_node_l.samples))
            g.node(r_label, self.head_node.sub_node_r.feat+" "+self.head_node.sub_node_r.val+" "+str(self.head_node.sub_node_r.samples))
            g.edge(head_label, l_label)
            g.edge(head_label, r_label)
            self.head_node.sub_node_l.draw(g, l_label)
            self.head_node.sub_node_r.draw(g, r_label)
            return g
    
    def predict(self, data):
        return self.head_node.pred(data)




# usage
dtree = Tree()
dtree.fit(data1, 'Purchase')
#dtree.fit(data2, 'cls')

from graphviz import Digraph

g = Digraph('Decision Tree', filename='dtree.gv')

g = dtree.graph(g)

#g.node('A', dtree.head_node.feat+" "+str(dtree.head_node.samples))
#g.node('B', dtree.head_node.sub_node_l.feat)
#g.node('C', dtree.head_node.sub_node_r.feat)

#g.edges(['AB', 'AC'])
g.view()
# then run ("C:\Program Files (x86)\Graphviz2.38\bin\dot.exe" -Tpdf dtree.gv -o dtree.pdf) at command line

dtree.predict(data1.iloc[0])
dtree.predict(data1.iloc[1])

data1['pred'] = data1.apply(lambda x: dtree.predict(x), axis=1)


raw = pd.read_csv('U:/CleanHarbors/clean-harbors-dataset.csv',encoding = "ISO-8859-1", header=0, low_memory=False)

topfeatures = ['genrtr','profil_cnt','vendor_id','vendor_profil_no','status','status_date','vendor_email','expiration_check_indcr','nw_chmcl_indcr','profil_cnt.1','profil_fee_indcr','row_version','genrtr.1','profil_creatn_usrnm','undisclosed_hazards','wst_classfctn_cd','profil_creatn_dt','profil_mdfctn_dt','status.1','profil_no','profil_type','genrtr_profil_no','expiration_date','dscrpn','core_profil_indcr','creatn_dt','no_other_const_indcr','sk_indcr','contains_organic_with_vp2','creatn_usrnam','service_provider_cd','no_rcra_metal_indcr','mdfctn_usrnam','provincial_wastecodes_indcr','prohibited_from_land_disp','profil_cnt.2','state_wastecodes_indcr','mdfctn_dt','const_variance_basis','wst_form_cd','us_epa_hazardous_waste_indcr','green_indcr','wastewater_or_non','labpack_waste_indcr','hocs','tsca','f006_or_f019_sludge','profil_mdfctn_usrnm','ldr_dspsl_category','genrtr.2','contains_vocs','strong_or_mild_odor','src_cd','ash','cercla_regulated','copied_from_template_indcr','unvrsl_wst','row_version.1','powdered_metalwaste_indcr','subject_to_neshap_indcr','human_waste_included_indcr','approval_date','color','genrc_profile','containerized','bulk_solid','bulk_liquid','waste_generation_process_descr','pcbs','shipment_frequency','physical_state','specific_gravity','lcr_routing_path_lov_id','hold_indcr','total_organic_carbon','cpg_owner','btus','sterilization_applied_indcr','waste_notexposed_infectious_indcr','texas_exempt_indcr','spec_aprvl_hndlg','osha_chemical_indcr','ph','pharm_production_indcr','hon_rule_indcr','first_time_approval_indcr','total_halogens','vendor_usrnm','profil_source','profil_submitter_user_type','profil_submitter','profil_submitter_email','accnt_rep','red_indcr','tolling_indcr','max_cntnr_qty','min_cntnr_qty','order_request_waste_indcr','cntnr_type_cd','profil_creatn_user_type','cntnr_size','dot_shpng_cd','const_basis_of_knowledge','cntnr_storage_capacity','contain_organic_with_vp1','sampl_takn.1','flash_point','conditionally_exempt_genrtr_indcr','boiling_point','viscosity','number_of_layers','metal_objects','melting_point', 'wst_classfctn_cd']
culledfeatures = ['rad_nuc_form','rad_chem_form','total_benzene_GT1Mg_indcr','flash_point_value_max','knowledge','ph_value','lbs_per_gal_or_cu_ft','latest_genrtr_transmitter','ph_value_max','tab_quantity','btus_value','is_knowledge_or_test_indcr','btus_value_max','other_solid_vehicle_type','liquid_vehicle_type','hin_no','special_reqmts','undisclosed_hazards_descr','solid_vehicle_type','density','odor','expected_number_liquid_loads','expected_number_solid_loads','solid_storage_capacity','tank_size','min_solid_qty','max_solid_qty','solid_ton_or_yd','other_cntnr_matl_descr','specific_disposal_info','metal_objects_descr','wst_cd_basis_of_knowledge','strong_odor_indcr','nfpa_hzrd_class_cd','other_drum_type_descr','battery_requirements_indcr','total_annual_benzene_indcr','percent_middle_layer','dspsl_tchngy','orign_cd','asbestos_waste_wetted_indcr','trtmnt_cd','sampl_takn','is_from_sic_indcr','variance_text','min_liquid_qty','max_liquid_qty','max_suspended_solid','min_suspended_solid','mgmt_mthd_cd','vapor_pressure','labpack_packng_matrl_indcr','labpack_shipng_cntnr_indcr','inorganic_waste_indcr','genrtr_rspbl_party','percent_top_layer','default_vendor','percent_bottom_layer','rcra_exempt_indcr','min_settled_solid','max_settled_solid','max_free_liquid','min_free_liquid','dot_hzrd_class','cntnr_matl_cd','other_process_gen_waste_descr','other_waste_source','ozone_depl','waste_source_cd','waste_mgmt_method','wst_cd_variance_basis','comingled_waste','exempt_waste','epa_cd_assist_reqd','sorbent_added','trtmnt_cd_assist_reqd','shpng_name_assist_reqd','entered_via_lab_pack','process_generating_waste_cd','from_drums','from_tanks','point_source_category_cd','regulated_under_benzene_neshap','uhcs_present','subject_to_cat_pre_disch_stds','ww_or_nww','contains_benzene_indcr','melting_point','metal_objects','number_of_layers','viscosity','boiling_point','conditionally_exempt_genrtr_indcr','flash_point','sampl_takn.1','contain_organic_with_vp1','cntnr_storage_capacity','cntnr_size','profil_creatn_user_type','cntnr_type_cd','order_request_waste_indcr','max_cntnr_qty','min_cntnr_qty','tolling_indcr','red_indcr','profil_submitter_user_type','profil_source','first_time_approval_indcr','pharm_production_indcr','hon_rule_indcr','ph','osha_chemical_indcr','texas_exempt_indcr','spec_aprvl_hndlg','sterilization_applied_indcr','waste_notexposed_infectious_indcr','btus','total_organic_carbon','cpg_owner','hold_indcr','lcr_routing_path_lov_id','specific_gravity','physical_state','pcbs','shipment_frequency','genrc_profile','bulk_liquid','bulk_solid','containerized','provincial_wastecodes_indcr','state_wastecodes_indcr','us_epa_hazardous_waste_indcr','green_indcr','unvrsl_wst','copied_from_template_indcr','human_waste_included_indcr','profil_type','ldr_dspsl_category','subject_to_neshap_indcr','powdered_metalwaste_indcr','no_rcra_metal_indcr','no_other_const_indcr','expiration_check_indcr','labpack_waste_indcr','sk_indcr','core_profil_indcr','service_provider_cd','contains_vocs','nw_chmcl_indcr','const_variance_basis','status.1','ash','strong_or_mild_odor','tsca','hocs','undisclosed_hazards','src_cd','cercla_regulated','prohibited_from_land_disp','wastewater_or_non','f006_or_f019_sludge','profile_aprvl_type','profil_fee_indcr','contains_organic_with_vp2','wst_classfctn_cd']
features = list(set(topfeatures).intersection(set(culledfeatures)))
wccs = ['FB1','CNO','CNOS','A22K','RORV','FB2','LCCRQ','A32V','A31','CCRK','CCRC','FB5','LCCRC','CBP','CCSS','CCRKS','CCS','LLF','D23','FB3','FB1','A32','LFB1','LBLA','CCRN','CNOSV','B35','LCCRD','A31','CNOS','LPTP','CFL1','CNO','CBPS','LCCRB','A99DB','LPTN','LBD2','RXHZ','CFL4','CCC','EEE','LBD1','LBD','LCCRA','FB1E','CCRKR','B26B','CCSM','FB4']
df2 = raw[raw['wst_classfctn_cd'].isin(wccs)].iloc[:1000]
df3 = df2[features]


#setup decision tree and data
# remove dates, usernames, emails, and null columns
#features = ['rad_nuc_form','rad_chem_form','total_benzene_GT1Mg_indcr','sampled_by','flash_point_value','flash_point_value_max','knowledge','ph_value','lbs_per_gal_or_cu_ft','latest_genrtr_transmitter','ph_value_max','tab_quantity','btus_value','is_knowledge_or_test_indcr','btus_value_max','other_solid_vehicle_type','liquid_vehicle_type','hin_no','special_reqmts','undisclosed_hazards_descr','solid_vehicle_type','density','odor','expected_number_liquid_loads','expected_number_solid_loads','solid_storage_capacity','tank_size','min_solid_qty','max_solid_qty','solid_ton_or_yd','other_cntnr_matl_descr','specific_disposal_info','metal_objects_descr','wst_cd_basis_of_knowledge','strong_odor_indcr','nfpa_hzrd_class_cd','other_drum_type_descr','battery_requirements_indcr','total_annual_benzene_indcr','percent_middle_layer','dspsl_tchngy','contact_name','orign_cd','asbestos_waste_wetted_indcr','trtmnt_cd','sampl_takn','is_from_sic_indcr','variance_text','min_liquid_qty','max_liquid_qty','max_suspended_solid','min_suspended_solid','mgmt_mthd_cd','vapor_pressure','labpack_packng_matrl_indcr','labpack_shipng_cntnr_indcr','inorganic_waste_indcr','genrtr_rspbl_party','percent_top_layer','default_vendor','percent_bottom_layer','rcra_exempt_indcr','signature_title','min_settled_solid','max_settled_solid','signature_name','max_free_liquid','min_free_liquid','dot_hzrd_class','cntnr_matl_cd','other_process_gen_waste_descr','other_waste_source','ozone_depl','waste_source_cd','waste_mgmt_method','wst_cd_variance_basis','comingled_waste','exempt_waste','epa_cd_assist_reqd','sorbent_added','trtmnt_cd_assist_reqd','shpng_name_assist_reqd','entered_via_lab_pack','process_generating_waste_cd','from_drums','from_tanks','point_source_category_cd','regulated_under_benzene_neshap','uhcs_present','subject_to_cat_pre_disch_stds','ww_or_nww','contains_benzene_indcr','melting_point','metal_objects','number_of_layers','viscosity','boiling_point','conditionally_exempt_genrtr_indcr','flash_point','sampl_takn.1','contain_organic_with_vp1','cntnr_storage_capacity','cntnr_size','profil_creatn_user_type','cntnr_type_cd','order_request_waste_indcr','max_cntnr_qty','min_cntnr_qty','tolling_indcr','red_indcr','profil_submitter_user_type','profil_source','total_halogens','first_time_approval_indcr','pharm_production_indcr','hon_rule_indcr','ph','osha_chemical_indcr','texas_exempt_indcr','spec_aprvl_hndlg','sterilization_applied_indcr','waste_notexposed_infectious_indcr','btus','total_organic_carbon','cpg_owner','hold_indcr','lcr_routing_path_lov_id','specific_gravity','physical_state','pcbs','shipment_frequency','genrc_profile','bulk_liquid','bulk_solid','containerized','color','provincial_wastecodes_indcr','state_wastecodes_indcr','us_epa_hazardous_waste_indcr','green_indcr','unvrsl_wst','copied_from_template_indcr','human_waste_included_indcr','profil_type','ldr_dspsl_category','subject_to_neshap_indcr','powdered_metalwaste_indcr','no_rcra_metal_indcr','no_other_const_indcr','expiration_check_indcr','labpack_waste_indcr','sk_indcr','core_profil_indcr','service_provider_cd','contains_vocs','nw_chmcl_indcr','const_variance_basis','status.1','ash','strong_or_mild_odor','tsca','hocs','undisclosed_hazards','src_cd','cercla_regulated','wst_form_cd','prohibited_from_land_disp','wastewater_or_non','f006_or_f019_sludge','profile_aprvl_type','profil_fee_indcr','contains_organic_with_vp2','wst_classfctn_cd']
#dtset = df1[features]

dtree = Tree()
%time dtree.fit(df3, 'wst_classfctn_cd')


